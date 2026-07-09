"""Event-driven trigger for adaptive trajectory prediction (3-factor and funnel modes)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# =====================================================================
# Mode 0: Always-on trigger (fallback)
# =====================================================================

class SimpleTrigger(nn.Module):
    """Always-on trigger (fallback)."""
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, trajectory, intent_logits, intent_history=None):
        B = trajectory.shape[0]
        return {
            'trigger_decision': torch.ones(B, dtype=torch.bool, device=trajectory.device),
            'trigger_score': torch.ones(B, device=trajectory.device),
            'maneuver_score': torch.zeros(B, device=trajectory.device),
            'stage_results': {},
        }


# =====================================================================
# Mode 1: Simple learned 3-factor trigger
# =====================================================================

class EventDrivenTrigger(nn.Module):
    """3-factor weighted trigger: 0.3*threat + 0.3*intent + 0.4*spatial."""
    def __init__(
        self,
        feature_dim: int = 6,
        num_intent_classes: int = 5,
        threat_weight: float = 0.3,
        intent_weight: float = 0.3,
        spatial_weight: float = 0.4,
        trigger_threshold: float = 0.5,
    ):
        super().__init__()
        self.threat_weight = threat_weight
        self.intent_weight = intent_weight
        self.spatial_weight = spatial_weight
        self.trigger_threshold = trigger_threshold

        self.threat_scorer = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.intent_scorer = nn.Sequential(
            nn.Linear(num_intent_classes, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.spatial_scorer = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, trajectory, intent_logits, intent_history=None):
        B, T, C = trajectory.shape

        vel = trajectory[:, -1, 3:6]
        threat = self.threat_scorer(vel).squeeze(-1)

        intent_probs = F.softmax(intent_logits, dim=-1)
        intent_score = self.intent_scorer(intent_probs).squeeze(-1)

        all_vel = trajectory[:, :, 3:6]
        speed = torch.norm(all_vel, dim=-1)
        mean_vel = all_vel.mean(dim=1)
        max_speed = speed.max(dim=1, keepdim=True)[0]
        vel_std = all_vel.std(dim=1)
        speed_change = speed[:, -1:] - speed[:, :1].mean(dim=1, keepdim=True)
        spatial_input = torch.cat([mean_vel, max_speed, vel_std, speed_change], dim=-1)
        spatial = self.spatial_scorer(spatial_input).squeeze(-1)

        trigger_score = (
            self.threat_weight * threat +
            self.intent_weight * intent_score +
            self.spatial_weight * spatial
        )
        trigger_decision = trigger_score > self.trigger_threshold

        maneuver_score = speed.std(dim=1) / (speed.mean(dim=1) + 1e-8)

        return {
            'trigger_decision': trigger_decision,
            'trigger_score': trigger_score,
            'maneuver_score': maneuver_score,
            'stage_results': {'mode': 'simple'},
        }


# =====================================================================
# Mode 2: Full Funnel Trigger
# =====================================================================

class FunnelTrigger(nn.Module):
    """
    Six-stage funnel filter trigger.

    Stages:
      1. Spatiotemporal boundary filter: CTE vs R_protect
      2. Cross-track error (CTE) computation
      3. Motion intent: heading angle / longitudinal / lateral stability
      4. Buffer management: Buffer Size >= W?
      5. Threat timing: D_t < D_warn OR TTA_t < T_warn?
      6. Attitude confirmation: aligned with zone + stable flight -> predict

    Args:
        feature_dim: input feature dim
        num_intent_classes: number of intent classes
        buffer_size: sliding window size W (frames)
        r_protect: protected zone radius (m)
        d_warn: distance warning threshold (m)
        t_warn: time warning threshold (s)
        cte_threshold: cross-track error threshold (m)
        heading_tolerance: heading alignment tolerance (deg)
        stability_threshold: stability threshold (velocity variance)
    """
    def __init__(
        self,
        feature_dim: int = 6,
        num_intent_classes: int = 5,
        # Funnel thresholds
        buffer_size: int = 10,          # W: min observation window
        r_protect: float = 500.0,       # R_protect: protected zone radius (m)
        d_warn: float = 200.0,          # D_warn: distance warning threshold (m)
        t_warn: float = 30.0,           # T_warn: time warning threshold (s)
        cte_threshold: float = 50.0,    # CTE threshold (m)
        heading_tolerance: float = 30.0, # heading tolerance (deg)
        stability_threshold: float = 2.0,# velocity variance stability threshold
        dt: float = 0.1,                # time step (s)
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.r_protect = r_protect
        self.d_warn = d_warn
        self.t_warn = t_warn
        self.cte_threshold = cte_threshold
        self.heading_tolerance = heading_tolerance
        self.stability_threshold = stability_threshold
        self.dt = dt

        # Learnable scorers (for threshold fine-tuning)
        self.stability_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.heading_encoder = nn.Sequential(
            nn.Linear(3 + num_intent_classes, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.threat_encoder = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Sliding window buffer (per-sample state; not registered to avoid grad issues)
        self.register_buffer('_buf_ptr', torch.zeros(1, dtype=torch.long))
        self._buffer_store = {}  # {batch_idx: trajectory_buffer}

    def _compute_cte(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Stage 1-2: compute cross-track error (CTE).

        CTE = perpendicular distance from current position to reference route.
        Reference route = start-to-end line of history (simplified).

        Returns:
            cte: (B,) cross-track error (m)
            distance_to_protected: (B,) distance to protected zone (m)
        """
        B, T, C = trajectory.shape
        pos = trajectory[:, :, :3]   # (B, T, 3)

        # Reference route: start -> end line of history
        start_pos = pos[:, 0, :]      # (B, 3)
        end_pos = pos[:, -1, :]       # (B, 3)
        route_vec = end_pos - start_pos  # (B, 3)
        route_len = torch.norm(route_vec, dim=-1).clamp(min=1e-6)  # (B,)

        # Project last position onto route
        last_pos = pos[:, -1, :]      # (B, 3)
        to_end = last_pos - start_pos  # (B, 3)

        # Projection param t = dot(to_end, route_vec) / |route_vec|^2
        t = torch.sum(to_end * route_vec, dim=-1) / (route_len ** 2)  # (B,)
        t = t.clamp(0.0, 1.0)  # clamp to segment

        # Projection point
        proj = start_pos + t.unsqueeze(-1) * route_vec  # (B, 3)
        cte = torch.norm(last_pos - proj, dim=-1)  # (B,) cross-track error

        # Distance to protected zone (simplified: route end as zone center)
        distance_to_protected = route_len * (1.0 - t)  # (B,) remaining along route

        return cte, distance_to_protected

    def _compute_heading_intent(
        self, trajectory: torch.Tensor, intent_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: compute heading intent angle + longitudinal/lateral stability.

        Returns:
            heading_to_zone: (B,) angle between heading and zone direction (deg)
            heading_aligned: (B,) whether aligned with zone (bool)
        """
        B, T, C = trajectory.shape
        pos = trajectory[:, :, :3]
        vel = trajectory[:, :, 3:6]

        # Current velocity direction (mean of last 3 frames to denoise)
        recent_vel = vel[:, -3:, :].mean(dim=1)  # (B, 3)
        vel_norm = recent_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel_dir = recent_vel / vel_norm  # (B, 3) avoid F.normalize zero vec -> NaN

        # Zone direction (route forward direction)
        route_vec = pos[:, -1, :3] - pos[:, 0, :3]
        route_norm = route_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        route_dir = route_vec / route_norm  # (B, 3)

        # Angle
        cos_angle = (vel_dir * route_dir).sum(dim=-1).clamp(-1, 1)  # (B,)
        heading_to_zone = torch.acos(cos_angle) * 180.0 / math.pi  # (B,) deg
        heading_aligned = heading_to_zone < self.heading_tolerance

        return heading_to_zone, heading_aligned

    def _compute_stability(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: compute longitudinal/lateral stability.

        Longitudinal = coefficient of variation of speed (low = stable)
        Lateral = variance of lateral acceleration (low = stable)

        Returns:
            long_stability: (B,) longitudinal stability [0,1] (high=stable)
            lat_stability: (B,) lateral stability [0,1] (high=stable)
        """
        B, T, C = trajectory.shape
        vel = trajectory[:, :, 3:6]  # (B, T, 3)

        # Speed magnitude
        speed = torch.norm(vel, dim=-1)  # (B, T)
        speed_mean = speed.mean(dim=-1)  # (B,)
        speed_std = speed.std(dim=-1)    # (B,)

        # Longitudinal stability: speed coefficient of variation
        cv = speed_std / (speed_mean + 1e-6)  # coefficient of variation
        long_stability = torch.exp(-cv * 5.0)  # [0,1], low CV = high stability

        # Lateral stability: acceleration perpendicular to velocity
        speed_3d = vel  # (B, T, 3)
        # Direction vectors (avoid F.normalize zero vec -> NaN)
        vel_curr = speed_3d[:, 1:, :]   # (B, T-1, 3)
        vel_prev = speed_3d[:, :-1, :]  # (B, T-1, 3)
        vel_dir = vel_curr / vel_curr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel_dir_prev = vel_prev / vel_prev.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Direction change rate (curvature approximation)
        lat_acc = torch.norm(vel_dir - vel_dir_prev, dim=-1)  # (B, T-1)
        lat_std = lat_acc.std(dim=-1)  # (B,)
        lat_stability = torch.exp(-lat_std * 10.0)  # [0,1]

        return long_stability, lat_stability

    def _compute_tta(self, distance: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """
        Stage 5: compute time-to-arrival (TTA) at protected zone.

        Args:
            distance: (B,) distance to protected zone (m)
            speed: (B,) current speed (m/s)
        Returns:
            tta: (B,) estimated time to arrival (s)
        """
        tta = distance / (speed + 1e-6)
        return tta

    def forward(
        self,
        trajectory: torch.Tensor,
        intent_logits: torch.Tensor,
        intent_history: Optional[torch.Tensor] = None,
        protected_zone_pos: Optional[torch.Tensor] = None,  # (B, 3) zone center (optional)
    ) -> Dict[str, torch.Tensor]:
        """
        Funnel filter forward pass.

        Args:
            trajectory: (B, T, 6) [x,y,z,vx,vy,vz]
            intent_logits: (B, num_classes)
            intent_history: (T, num_classes) intent history
            protected_zone_pos: (B, 3) no-fly zone center (optional, else estimated from route end)

        Returns:
            trigger_decision: (B,) bool
            trigger_score: (B,) float [0,1]
            stage_results: dict per-stage details
        """
        B, T, C = trajectory.shape
        device = trajectory.device
        intent_probs = F.softmax(intent_logits, dim=-1)

        stage_results = {}

        # ============================================================
        # Stage 1: spatiotemporal boundary filter — CTE vs R_protect
        # ============================================================
        cte, dist_to_zone = self._compute_cte(trajectory)
        stage_results['cte'] = cte
        stage_results['dist_to_zone'] = dist_to_zone

        # Inside protected zone?
        inside_protected = cte < self.cte_threshold  # (B,)
        stage_results['inside_protected'] = inside_protected

        # ============================================================
        # Stage 2: continue only for targets entering the zone
        # ============================================================
        cte_pass = cte < self.r_protect  # CTE < R_protect?
        stage_results['cte_pass'] = cte_pass

        # ============================================================
        # Stage 3: motion intent extraction
        # ============================================================
        # 3a. heading intent angle
        heading_angle, heading_aligned = self._compute_heading_intent(trajectory, intent_probs)
        stage_results['heading_angle'] = heading_angle
        stage_results['heading_aligned'] = heading_aligned

        # 3b. stability analysis
        long_stab, lat_stab = self._compute_stability(trajectory)
        stage_results['longitudinal_stability'] = long_stab
        stage_results['lateral_stability'] = lat_stab

        # Learnable stability score
        stab_input = torch.cat([
            cte.unsqueeze(-1),
            heading_angle.unsqueeze(-1) / 180.0,
            long_stab.unsqueeze(-1),
            lat_stab.unsqueeze(-1),
            (trajectory[:, :, 3:6].norm(dim=-1).std(dim=-1)).unsqueeze(-1),  # speed fluctuation
            (trajectory[:, :, 3:6].norm(dim=-1).mean(dim=-1)).unsqueeze(-1), # mean speed
        ], dim=-1)  # (B, 6)
        stab_score = self.stability_encoder(stab_input).squeeze(-1)  # (B,)
        stage_results['stability_score'] = stab_score

        # ============================================================
        # Stage 4: buffer management — Buffer Size >= W?
        # ============================================================
        # Simplified: use trajectory length T as buffer sufficiency proxy
        buffer_sufficient = torch.ones(B, dtype=torch.bool, device=device)  # assume always OK
        if T < self.buffer_size:
            buffer_sufficient = torch.zeros(B, dtype=torch.bool, device=device)
        stage_results['buffer_sufficient'] = buffer_sufficient

        # ============================================================
        # Stage 5: threat timing — D_t < D_warn OR TTA_t < T_warn?
        # ============================================================
        speed = torch.norm(trajectory[:, -1, 3:6], dim=-1) + 1e-6  # (B,) current speed

        tta = self._compute_tta(dist_to_zone, speed)  # time to arrival
        stage_results['tta'] = tta

        distance_threat = dist_to_zone < self.d_warn
        time_threat = tta < self.t_warn
        stage_results['distance_threat'] = distance_threat
        stage_results['time_threat'] = time_threat

        # Learnable threat score
        threat_input = torch.cat([
            (dist_to_zone / self.d_warn).clamp(0, 3).unsqueeze(-1),
            (tta / self.t_warn).clamp(0, 3).unsqueeze(-1),
            speed.unsqueeze(-1) / 20.0,
            cte.unsqueeze(-1) / self.cte_threshold,
            heading_angle.unsqueeze(-1) / 180.0,
        ], dim=-1)  # (B, 5)
        threat_score = self.threat_encoder(threat_input).squeeze(-1)  # (B,)
        stage_results['threat_score'] = threat_score

        # ============================================================
        # Stage 6: attitude confirmation — aligned + stable -> predict
        # ============================================================
        # Combined decision: all conditions must hold
        all_conditions = (
            cte_pass &                          # CTE < R_protect
            inside_protected &                  # inside protected zone
            heading_aligned &                   # heading aligned
            buffer_sufficient &                 # buffer sufficient
            (distance_threat | time_threat)     # distance or time threat
        )

        # Learnable heading score (fuses intent info)
        heading_input = torch.cat([
            heading_angle.unsqueeze(-1) / 180.0,
            long_stab.unsqueeze(-1),
            lat_stab.unsqueeze(-1),
            intent_probs,  # (B, num_classes)
        ], dim=-1)  # (B, 3 + num_classes)
        heading_conf = self.heading_encoder(heading_input).squeeze(-1)  # (B,)
        stage_results['heading_confidence'] = heading_conf

        # Final score: hard decision + learnable score
        condition_score = all_conditions.float()  # hard condition satisfaction [0, 1]
        learned_score = (
            0.25 * stab_score +
            0.35 * threat_score +
            0.40 * heading_conf
        )
        # Hard conditions gate the score: unmet -> large reduction
        trigger_score = condition_score * learned_score + (1 - condition_score) * learned_score * 0.3
        trigger_decision = trigger_score > 0.5

        stage_results['all_conditions_met'] = all_conditions
        stage_results['condition_score'] = condition_score
        stage_results['learned_score'] = learned_score

        return {
            'trigger_decision': trigger_decision,
            'trigger_score': trigger_score,
            'maneuver_score': stab_score,
            'stage_results': stage_results,
        }
