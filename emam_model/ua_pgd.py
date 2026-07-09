"""Uncertainty-Aware Physics-Guided Decoder with inertia gate, anchor pull, and kinematic model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================================
# Submodule 1: Orthogonal Step Encoder
# ============================================================================

class OrthogonalStepEncoder(nn.Module):
    """Orthogonal step encoding: generates a distinct representation per prediction step.

    Uses multi-frequency sine-cosine pairs with exponentially growing frequencies so
    that encodings of different steps are approximately orthogonal.
    """
    def __init__(self, pred_len: int, d_model: int):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # Fixed (non-learnable) orthogonal basis to avoid overfitting
        steps = torch.arange(1, pred_len + 1).float()           # [1, ..., pred_len]
        # Frequencies grow exponentially from 1 to ~148; high frequencies distinguish far steps
        freqs = torch.exp(torch.linspace(0, math.log(150), d_model // 2))

        pe = torch.zeros(pred_len, d_model)
        pe[:, 0::2] = torch.sin(steps.unsqueeze(1) * freqs)    # even dims: sin
        pe[:, 1::2] = torch.cos(steps.unsqueeze(1) * freqs)   # odd dims: cos
        self.register_buffer('_pe', pe)                         # (pred_len, d_model)

        # Learnable scaling factor so the model can adapt the encoding amplitude
        self.step_scale = nn.Parameter(torch.ones(1))

    def forward(self) -> torch.Tensor:
        """
        Returns:
            step_encoding: (pred_len, d_model)  orthogonal step encoding
        """
        return self.step_scale * self._pe


# ============================================================================
# Submodule 2: Physics Inertia Gate (core mechanism)
# ============================================================================

class PhysicsInertiaGate(nn.Module):
    """
    Physical Inertia Gating.

    Core idea: step output = physics extrapolation x inertia gate + neural prediction x (1 - inertia gate),
    with an added anchor pull-back effect.

    Four gates:
        gate_inertia:    ratio of historical inertia to keep (0~1)
                         - rises during maneuvers: follow abrupt actions
                         - falls during cruise/hover: rely on model prediction
        gate_anchor:     anchor pull-back strength (0~1)
                         - stronger for farther steps, prevents long-horizon divergence
        gate_confidence: model confidence (0~1), mapped from uncertainty
                         - reduces neural weight under high uncertainty
        gate_mode:       flight-mode modulation (0~1), mapped from intent weights
                         - forces stronger anchor pull-back when hovering

    Inputs: last-step encoded feature (B, d_model)
            intent weights (B, num_intent_classes)
            step encoding (pred_len, d_model)
    Output: gate parameter sequence (B, pred_len, num_gates)
    """
    def __init__(self, d_model: int, num_intent_classes: int, pred_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_intent_classes = num_intent_classes
        self.num_gates = 4  # inertia, anchor, confidence, mode

        # Predict base gates from encoded feature (gate_inertia, gate_anchor)
        hidden = d_model // 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),        # gate_inertia, gate_anchor
            nn.Sigmoid()                  # constrain to (0,1)
        )

        # Predict mode gate from intent weights (per class)
        # gate_mode: (B, num_intent_classes) one mode value per intent
        self.intent_to_mode = nn.Sequential(
            nn.Linear(num_intent_classes, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, num_intent_classes),
            nn.Sigmoid()
        )

        # Orthogonal step-encoding projection (for step-dependent anchor weights)
        self.step_proj = nn.Linear(d_model, 1, bias=False)

        # Gate temperature parameter (avoids overly binary gates)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Init: default to physics inertia dominance, moderate anchor pull-back
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.xavier_uniform_(self.gate_mlp[2].weight)
        # Init gate_inertia high (~0.6), gate_anchor moderate (~0.3)
        with torch.no_grad():
            self.gate_mlp[2].bias.copy_(torch.tensor([0.6, 0.3]))

    def forward(
        self,
        last_encoded: torch.Tensor,         # (B, d_model)
        intent_weights: torch.Tensor,         # (B, num_intent_classes)
        step_encoding: torch.Tensor,          # (pred_len, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate_inertia:      (B, pred_len)    inertia retention ratio
            gate_anchor:       (B, pred_len)    anchor pull-back strength
            gate_confidence:   (B, pred_len)    confidence (no uncertainty input yet, set to 1)
            gate_mode:         (B, num_intent_classes)  per-class mode strength
            gate_mode_effective: (B, pred_len)  weighted per-step mode strength
        """
        B = last_encoded.shape[0]
        P = step_encoding.shape[0]          # pred_len

        # --- 1. Base gates (predicted from encoded feature) ---
        base_gates = self.gate_mlp(last_encoded)          # (B, 2)
        gate_inertia_base = base_gates[:, 0]               # (B,)
        gate_anchor_base = base_gates[:, 1]                # (B,)

        # --- 2. Step-dependent anchor weights (farther step, stronger anchor) ---
        # step_encoding: (pred_len, d_model)
        step_weights_raw = self.step_proj(step_encoding).squeeze(-1)   # (pred_len,)
        step_weights = torch.sigmoid(step_weights_raw)                # (pred_len,)
        # Step weights grow gradually from ~0.2 to ~0.8
        step_weights = 0.2 + 0.6 * step_weights

        # gate_anchor grows with step
        gate_anchor = gate_anchor_base.unsqueeze(1) * step_weights.unsqueeze(0)  # (B, pred_len)
        gate_anchor = gate_anchor.clamp(0.0, 1.0)

        # --- 3. Inertia gate: decreases with step (far-horizon prediction relies more on model) ---
        # gate_inertia is inversely proportional to step
        inertia_step_decay = torch.linspace(1.0, 0.4, P, device=last_encoded.device)  # early 1.0 -> far 0.4
        gate_inertia = gate_inertia_base.unsqueeze(1) * inertia_step_decay.unsqueeze(0)  # (B, pred_len)
        gate_inertia = gate_inertia.clamp(0.0, 1.0)

        # --- 4. Mode gate (predicted per intent class) ---
        # intent_to_mode outputs (B, num_intent_classes), same shape as intent_weights
        # gate_mode in [0,1]; high value means anchor pull-back is enhanced for that intent
        gate_mode = self.intent_to_mode(intent_weights)   # (B, num_intent_classes)

        # Map per-class gate_mode to a weighted average over prediction steps
        # gate_mode_effective[b, p] = sum_c(gate_mode[b,c] * intent_weights[b,c])
        # Physical meaning: overall anchor pull-back strength under the current intent distribution
        gate_mode_per_step = gate_mode * intent_weights          # (B, num_intent_classes)
        gate_mode_effective = gate_mode_per_step.sum(dim=1, keepdim=True)  # (B, 1)
        gate_mode_effective = gate_mode_effective.expand(-1, P)  # (B, P)

        # gate_confidence: no uncertainty input yet, default all ones (can later accept an external uncertainty feature)
        gate_confidence = torch.ones(B, P, device=last_encoded.device)

        return gate_inertia, gate_anchor, gate_confidence, gate_mode, gate_mode_effective


# ============================================================================
# Submodule 3: Kinematic Physics Model
# ============================================================================

class KinematicPhysicsModel(nn.Module):
    """
    Kinematic Physics Model with decoupled degrees of freedom.

    Models position/velocity/acceleration independently for physical consistency:

    Position extrapolation: p_{t+dt} = p_t + v_t * dt + 0.5 * a_t * dt^2
    Velocity extrapolation: v_{t+dt} = v_t + a_t * dt
    Acceleration bound: |a| <= max_acc (typical drone value: 15 m/s^2)

    Serves only as a strong inductive bias, blended in when the inertia gate is open; not a hard constraint.
    """
    def __init__(self, trajectory_dim: int = 6, max_accel: float = 15.0):
        super().__init__()
        self.traj_dim = trajectory_dim          # 6: [x,y,z,vx,vy,vz]
        self.max_accel = max_accel
        self.pos_dim = 3                         # position dims: [x,y,z]

        # Acceleration estimation MLP: predicts an acceleration correction from state
        # Input: position(3) + velocity(3)
        # Output: acceleration correction(3)
        self.acc_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 3)
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extrapolate one displacement step (relative displacement) via kinematic equations.

        Args:
            trajectory: (B, T, 6)  [x, y, z, vx, vy, vz]
        Returns:
            physics_delta: (B, 3)  relative displacement increment (same units as neural_delta)
        """
        dt = 0.1

        # Take the last and second-to-last time steps
        last = trajectory[:, -1, :]   # (B, 6)
        prev = trajectory[:, -2, :]  # (B, 6)

        # Reference position (absolute)
        base_pos = last[:, :3]       # (B, 3)
        # Velocity (finite-difference estimate)
        vel = (last[:, :3] - prev[:, :3]) / dt
        vel = torch.clamp(vel, -50, 50)

        # Acceleration network estimate
        state_input = torch.cat([base_pos, vel], dim=-1)
        acc_pred = self.acc_net(state_input)       # (B, 3)
        acc_pred = torch.clamp(acc_pred, -self.max_accel, self.max_accel)

        # Relative displacement: p_{t+1} = p_t + v_t*dt + 0.5*a_t*dt^2
        physics_delta = vel * dt + 0.5 * acc_pred * (dt ** 2)  # (B, 3)

        return physics_delta

    def multi_step(
        self,
        trajectory: torch.Tensor,
        pred_len: int
    ) -> torch.Tensor:
        """
        Multi-step physics extrapolation (no neural network, pure kinematic equations).

        Outputs relative displacement: the sequence of displacement increments extrapolated
        from the last historical position.
        Output shape: (B, pred_len, 3), aligned with neural_delta (same units).

        Args:
            trajectory: (B, T, 6)  historical trajectory [x,y,z,vx,vy,vz]
            pred_len: number of prediction steps
        Returns:
            physics_trajectory: (B, pred_len, 3)  relative displacement sequence
        """
        B = trajectory.shape[0]
        dt = 0.1

        # Reference position: position at the last historical step (absolute coordinates)
        base_pos = trajectory[:, -1, :3].clone()  # (B, 3)

        # Initial velocity: finite difference from the last two frames
        vel = (trajectory[:, -1, :3] - trajectory[:, -2, :3]) / dt
        vel = torch.clamp(vel, -50, 50)

        # Position accumulation (start from reference position, accumulate displacement)
        pos_delta = torch.zeros(B, 3, device=trajectory.device)
        physics_preds = []

        # Acceleration state (for exponential smoothing)
        acc_state = torch.zeros_like(vel)

        for step in range(pred_len):
            # Acceleration network estimate
            state_input = torch.cat([base_pos + pos_delta, vel], dim=-1)  # use absolute position
            acc_pred = self.acc_net(state_input)  # (B, 3)
            acc_pred = torch.clamp(acc_pred, -self.max_accel, self.max_accel)

            # Exponentially smooth acceleration (avoid abrupt changes)
            acc_state = 0.7 * acc_state + 0.3 * acc_pred

            # Update velocity and displacement
            vel = vel + acc_state * dt
            pos_delta = pos_delta + vel * dt

            # Record relative displacement (excluding reference position)
            physics_preds.append(pos_delta.clone())

        return torch.stack(physics_preds, dim=1)  # (B, pred_len, 3)


# ============================================================================
# Submodule 4: Neural Decoder
# ============================================================================

class NeuralDecoder(nn.Module):
    """
    Neural decoder: decodes encoded features into displacement increments.

    Modulates features with the step encoding so different steps produce different predictions.
    """
    def __init__(self, d_model: int, trajectory_dim: int = 6):
        super().__init__()
        self.d_model = d_model

        # Feature -> hidden state
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        # Output heads: displacement + uncertainty (log(variance))
        self.delta_head = nn.Linear(d_model, 3)
        self.var_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 3),
            nn.Softplus(),  # output log(variance) in (0, +inf), ensures positive variance
        )

    def forward(
        self,
        encoded: torch.Tensor,         # (B, d_model)  encoded feature (last step or pooled)
        step_encoding: torch.Tensor,    # (pred_len, d_model)  step encoding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            neural_delta:  (B, pred_len, 3)  neural displacement increment
            logvar:        (B, pred_len, 3)  log variance (uncertainty)
        """
        B = encoded.shape[0]
        P = step_encoding.shape[0]

        # Feature + step encoding
        feat = self.proj(encoded)                           # (B, d_model)
        feat = feat.unsqueeze(1) + step_encoding.unsqueeze(0)  # (B, pred_len, d_model)

        # Flatten batch and step dims, decode uniformly
        feat_flat = feat.reshape(B * P, self.d_model)       # (B*P, d_model)

        delta_flat = self.delta_head(feat_flat)             # (B*P, 3)
        var_flat = self.var_head(feat_flat)                 # (B*P, 3)

        delta = delta_flat.reshape(B, P, 3)                 # (B, pred_len, 3)
        logvar = var_flat.reshape(B, P, 3)                  # (B, pred_len, 3)

        return delta, logvar


# ============================================================================
# Submodule 5: Multi-Hypothesis Neural Decoder
# ============================================================================

class MultiHeadNeuralDecoder(nn.Module):
    """Multi-hypothesis decoder: K independent prediction heads plus a confidence scoring head.

    Trained with Winner-Takes-All loss; at inference outputs K trajectories with confidence scores.
    """
    def __init__(self, d_model: int, trajectory_dim: int = 6, K: int = 5):
        super().__init__()
        self.d_model = d_model
        self.K = K

        # Shared feature projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        # K independent displacement prediction heads
        self.delta_heads = nn.ModuleList([
            nn.Linear(d_model, 3) for _ in range(K)
        ])

        # K independent uncertainty heads
        self.var_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 32),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 3),
                nn.Softplus(),
            ) for _ in range(K)
        ])

        # Confidence scoring head: predicts which hypothesis is most likely correct
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, K),
        )

    def forward(
        self,
        encoded: torch.Tensor,         # (B, d_model)
        step_encoding: torch.Tensor,    # (pred_len, d_model)
        return_all: bool = False,       # if True, return full dict; else return (delta, logvar) tuple
    ) -> Dict[str, torch.Tensor]:
        """
        Interface compatible with NeuralDecoder:
        - return_all=False: returns (best_delta, best_logvar) tuple
        - return_all=True:  returns a full dict

        Returns (when return_all=True):
            deltas:      (K, B, pred_len, 3)  K trajectory displacements
            logvars:     (K, B, pred_len, 3)  per-trajectory uncertainty
            confidences: (B, K)               confidence logits
            best_delta:  (B, pred_len, 3)     highest-confidence trajectory
            best_logvar: (B, pred_len, 3)     corresponding uncertainty
        """
        B = encoded.shape[0]
        P = step_encoding.shape[0]

        # Shared feature projection
        feat = self.proj(encoded)                           # (B, d_model)
        feat = feat.unsqueeze(1) + step_encoding.unsqueeze(0)  # (B, pred_len, d_model)
        feat_flat = feat.reshape(B * P, self.d_model)       # (B*P, d_model)

        # K independent predictions
        all_deltas = []
        all_logvars = []
        for k in range(self.K):
            d_flat = self.delta_heads[k](feat_flat)         # (B*P, 3)
            v_flat = self.var_heads[k](feat_flat)           # (B*P, 3)
            all_deltas.append(d_flat.reshape(B, P, 3))
            all_logvars.append(v_flat.reshape(B, P, 3))

        deltas = torch.stack(all_deltas, dim=0)              # (K, B, P, 3)
        logvars = torch.stack(all_logvars, dim=0)            # (K, B, P, 3)

        # Confidence scoring (based on pooled features)
        pooled_feat = feat.mean(dim=1)                       # (B, d_model)
        confidences = self.confidence_head(pooled_feat)      # (B, K)

        # Best hypothesis (highest confidence)
        best_idx = confidences.argmax(dim=1)                 # (B,)
        best_delta = deltas[best_idx, torch.arange(B, device=encoded.device)]  # (B, P, 3)
        best_logvar = logvars[best_idx, torch.arange(B, device=encoded.device)]

        if return_all:
            return {
                'deltas': deltas,
                'logvars': logvars,
                'confidences': confidences,
                'best_delta': best_delta,
                'best_logvar': best_logvar,
            }
        else:
            # Backward-compatible: return (delta, logvar) tuple
            return best_delta, best_logvar

    @staticmethod
    def compute_wta_loss(
        deltas: torch.Tensor,           # (K, B, P, 3)
        logvars: torch.Tensor,          # (K, B, P, 3)
        confidences: torch.Tensor,      # (B, K)
        targets: torch.Tensor,          # (B, P, 3)
    ) -> Dict[str, torch.Tensor]:
        """
        Winner-Takes-All loss.

        1. Compute the L2 error of each trajectory
        2. Select the one with the smallest error as the winner
        3. MSE loss is backpropagated only through the winner
        4. Confidence loss uses cross-entropy (teaches the model which head is best)
        """
        K, B, P, _ = deltas.shape
        device = deltas.device

        # MSE per trajectory (B,)
        errors = []
        for k in range(K):
            err = F.mse_loss(deltas[k], targets, reduction='none').mean(dim=(1, 2))  # (B,)
            errors.append(err)
        error_matrix = torch.stack(errors, dim=1)  # (B, K)

        # Winner = smallest error
        winners = error_matrix.argmin(dim=1)  # (B,)

        # WTA displacement loss: only backprop through winner
        wta_disp_loss = torch.tensor(0.0, device=device)
        for k in range(K):
            mask = (winners == k)
            if mask.any():
                wta_disp_loss = wta_disp_loss + F.mse_loss(
                    deltas[k][mask], targets[mask]
                ) * mask.float().mean()

        # Confidence loss: cross-entropy, teaches the model to predict the winner
        conf_loss = F.cross_entropy(confidences, winners)

        return {
            'wta_disp_loss': wta_disp_loss,
            'conf_loss': conf_loss,
            'total_wta_loss': wta_disp_loss + 0.1 * conf_loss,
            'winners': winners,
            'min_error': error_matrix.min(dim=1).values.mean(),
        }

    @staticmethod
    def compute_minade_fde(
        deltas: torch.Tensor,           # (K, B, P, 3)
        targets: torch.Tensor,          # (B, P, 3)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute minADE_K and minFDE_K:
        for each sample, pick the trajectory with the smallest error among K and compute the metrics.

        Standard multi-hypothesis evaluation metrics.
        """
        K, B, P, _ = deltas.shape

        # ADE and FDE per trajectory
        all_ade = []  # K tensors of shape (B,)
        all_fde = []
        for k in range(K):
            step_errs = torch.norm(deltas[k] - targets, dim=-1)  # (B, P)
            all_ade.append(step_errs.mean(dim=1))                 # (B,)
            all_fde.append(step_errs[:, -1])                      # (B,)

        ade_matrix = torch.stack(all_ade, dim=1)  # (B, K)
        fde_matrix = torch.stack(all_fde, dim=1)

        min_ade = ade_matrix.min(dim=1).values     # (B,)
        min_fde = fde_matrix.min(dim=1).values

        return {
            'min_ade': min_ade,      # (B,)
            'min_fde': min_fde,      # (B,)
            'ade_all': ade_matrix,   # (B, K)
            'fde_all': fde_matrix,   # (B, K)
        }


# ============================================================================
# Main module: UA-PGD
# ============================================================================

class UncertaintyAwarePGD(nn.Module):
    """
    Uncertainty-Aware Physics-Guided Decoder.

    Overall flow::

        encoded_feat ──────────────────────────────────┐
                                                          │
        global_anchor ──→ fuse with encoded ──┐          │
                                           ↓             │
        history ──→ kinematic physics model ─→ physics_delta ─→ ┴→ inertia-gate blend → neural residual → final displacement
                                           ↑             │
        step encoding ──────────────────────────────────┘

    Blend formula::

        pred[t] = gate_inertia[t] * physics_delta[t]
                + (1 - gate_inertia[t]) * (neural_delta[t] + global_anchor_contribution)
                + gate_mode[t] * anchor_pull[t]

    where anchor_pull[t] = gate_anchor[t] * (global_anchor - current_pos)

    Gate behavior:
        - gate_inertia:  high during maneuvers, low during cruise, lower for far steps
        - gate_anchor:   stronger for farther steps, forced up when hovering
        - gate_mode:     mapped from intent weights, hover intent -> anchor dominates
        - gate_confidence: mapped from uncertainty, high uncertainty -> stronger anchor pull-back
    """

    def __init__(
        self,
        d_model: int = 256,
        pred_len: int = 20,
        trajectory_dim: int = 6,
        num_intent_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.traj_dim = trajectory_dim
        self.num_intent_classes = num_intent_classes

        # Submodules
        self.step_encoder = OrthogonalStepEncoder(pred_len, d_model)
        self.physics_model = KinematicPhysicsModel(trajectory_dim)
        self.neural_decoder = NeuralDecoder(d_model, trajectory_dim)

        self.physics_gate = PhysicsInertiaGate(
            d_model=d_model,
            num_intent_classes=num_intent_classes,
            pred_len=pred_len,
        )

        # Global anchor fusion: project the anchor vector to position space via an MLP
        self.anchor_to_pos = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )

        # Feature compression layer (project encoded feature into decoding space)
        self.feat_compress = nn.Linear(d_model, d_model)

        # Dropout (prevent overfitting)
        self.dropout = nn.Dropout(p=dropout)

        # Physics consistency loss weight (used by external compute_loss)
        self.physics_loss_weight = 0.05

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.feat_compress.weight)
        nn.init.zeros_(self.feat_compress.bias)
        for m in self.anchor_to_pos.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        encoded_feat: torch.Tensor,       # (B, T, d_model)
        global_anchor: torch.Tensor,      # (B, 1, d_model)
        historical_trajectory: torch.Tensor,  # (B, T, 6)
        return_uncertainty: bool = True,
        intent_weights: Optional[torch.Tensor] = None,  # (B, num_intent_classes)
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass.

        Args:
            encoded_feat:         (B, T, d_model)   EMam-SE encoded feature
            global_anchor:        (B, 1, d_model)   global target anchor
            historical_trajectory:(B, T, 6)          [x,y,z,vx,vy,vz]
            return_uncertainty:  whether to return uncertainty
            intent_weights:       (B, num_intent_classes)  intent weights (uniform if None)

        Returns:
            predictions:  (B, pred_len, 3)  future 3D displacement
            logvar:      (B, pred_len, 3)  log variance
            (optional) gates: (B, pred_len, 4)  four gate values (for intent read-out)
        """
        B, T, D = encoded_feat.shape
        P = self.pred_len
        device = encoded_feat.device

        # -------------------------------------------------------------------------
        # 1. Orthogonal step encoding
        # -------------------------------------------------------------------------
        step_encoding = self.step_encoder()                        # (P, d_model)
        step_encoding = step_encoding.to(device)

        # -------------------------------------------------------------------------
        # 2. Feature preparation
        # -------------------------------------------------------------------------
        # Take last-step encoded feature
        last_encoded = encoded_feat[:, -1, :]                       # (B, d_model)
        last_encoded = self.feat_compress(last_encoded)           # (B, d_model)
        last_encoded = self.dropout(last_encoded)

        # -------------------------------------------------------------------------
        # 3. Project global anchor to position target
        # -------------------------------------------------------------------------
        # anchor from (B, 1, d_model) -> (B, 3) position
        anchor_pos = self.anchor_to_pos(global_anchor.squeeze(1))  # (B, 3)

        # -------------------------------------------------------------------------
        # 4. Physics inertia gate
        # -------------------------------------------------------------------------
        # Intent weights: use uniform distribution if not provided
        if intent_weights is None:
            intent_weights = torch.ones(B, self.num_intent_classes, device=device)
            intent_weights = intent_weights / self.num_intent_classes

        gate_inertia, gate_anchor, gate_confidence, gate_mode, gate_mode_effective = self.physics_gate(
            last_encoded=last_encoded,
            intent_weights=intent_weights,
            step_encoding=step_encoding,
        )
        # Gate shapes:
        #   gate_inertia:      (B, P)
        #   gate_anchor:       (B, P)
        #   gate_confidence:   (B, P)
        #   gate_mode:         (B, num_intent_classes)  per-intent mode strength
        #   gate_mode_effective: (B, P)  weighted per-step mode strength

        # -------------------------------------------------------------------------
        # 5. Multi-step physics extrapolation
        # -------------------------------------------------------------------------
        # physics_trajectory: (B, P, 3)  multi-step physics displacement sequence (each step extrapolated from initial position)
        physics_trajectory = self.physics_model.multi_step(
            historical_trajectory, pred_len=P
        )  # (B, P, 3)

        # -------------------------------------------------------------------------
        # 6. Neural decoding
        # -------------------------------------------------------------------------
        neural_delta, logvar = self.neural_decoder(
            encoded=last_encoded,           # (B, d_model)
            step_encoding=step_encoding,   # (P, d_model)
        )  # neural_delta: (B, P, 3), logvar: (B, P, 3)

        # -------------------------------------------------------------------------
        # 7. Physics inertia gate blending (core mechanism)
        # -------------------------------------------------------------------------
        # Position reference: last-step position
        last_pos = historical_trajectory[:, -1, :3]                  # (B, 3)
        last_pos_expanded = last_pos.unsqueeze(1).expand(-1, P, -1)  # (B, P, 3)

        # Anchor pull-back vector: (anchor_pos - last_pos) x gate_anchor
        anchor_pull = (anchor_pos.unsqueeze(1) - last_pos_expanded)  # (B, P, 3)
        anchor_pull = anchor_pull * gate_anchor.unsqueeze(-1)         # (B, P, 3)

        # Neural prediction + anchor pull-back (accounting for confidence)
        confidence_factor = gate_confidence.unsqueeze(-1)  # (B, P, 1)
        neural_guided = neural_delta + anchor_pull        # (B, P, 3)
        neural_guided = neural_guided * confidence_factor  # lower neural weight under low confidence

        # Mode modulation: strengthen anchor dominance when hovering (higher mode -> physics inertia more suppressed by anchor pull-back)
        gate_mode_exp = gate_mode_effective.unsqueeze(-1)  # (B, P, 1)
        inertia_effective = gate_inertia.unsqueeze(-1) * (1.0 - 0.3 * gate_mode_exp)  # (B, P, 1)

        # Blend
        blended = (
            inertia_effective * physics_trajectory
            + (1.0 - inertia_effective) * neural_guided
        )

        # -------------------------------------------------------------------------
        # 8. Kinematic constraint (optional post-processing, does not modify gradients)
        # -------------------------------------------------------------------------
        # * _kinematic_postprocess is disabled - thresholds have wrong units in normalized space
        # and break normal predictions (train RMSE 0.5m -> eval RMSE 853m). Re-enable after fixing.
        # if not self.training:
        #     blended = self._kinematic_postprocess(blended, historical_trajectory)

        # -------------------------------------------------------------------------
        # 9. Uncertainty handling
        # -------------------------------------------------------------------------
        # Softplus output in (0, +inf), but keep clamp as a numerical safety bound
        # clamp range [-10, 10] corresponds to variance [4.5e-5, 2.2e4], covering a reasonable interval
        logvar_clamped = logvar.clamp(-10.0, 10.0)

        return {
            'predictions': blended,              # (B, pred_len, 3)
            'logvar': logvar_clamped,            # (B, pred_len, 3)
            'gate_inertia': gate_inertia,        # (B, pred_len)    inertia retention ratio
            'gate_anchor': gate_anchor,          # (B, pred_len)    anchor pull-back strength
            'gate_mode': gate_mode,              # (B, num_intent)  per-class mode strength
            'gate_confidence': gate_confidence,  # (B, pred_len)    confidence
            'gate_mode_effective': gate_mode_effective,  # (B, pred_len) per-step mode strength
            'physics_trajectory': physics_trajectory,  # (B, pred_len, 3) physics extrapolation displacement
            'neural_delta': neural_delta,        # (B, pred_len, 3) neural displacement increment
        }

    def forward_multi_head(
        self,
        encoded_feat: torch.Tensor,       # (B, T, d_model)
        global_anchor: torch.Tensor,      # (B, 1, d_model)
        historical_trajectory: torch.Tensor,  # (B, T, 6)
        intent_weights: Optional[torch.Tensor] = None,  # (B, num_intent_classes)
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-hypothesis forward pass: uses MultiHeadNeuralDecoder to generate K trajectories.

        Shared components (physics extrapolation, gates, anchor) are computed once, then blended with each of the K neural predictions.
        """
        B, T, D = encoded_feat.shape
        P = self.pred_len
        device = encoded_feat.device

        # ---- Shared components (same as forward) ----
        step_encoding = self.step_encoder().to(device)              # (P, d_model)

        last_encoded = encoded_feat[:, -1, :]                       # (B, d_model)
        last_encoded = self.feat_compress(last_encoded)             # (B, d_model)
        last_encoded = self.dropout(last_encoded)

        anchor_pos = self.anchor_to_pos(global_anchor.squeeze(1))  # (B, 3)

        if intent_weights is None:
            intent_weights = torch.ones(B, self.num_intent_classes, device=device)
            intent_weights = intent_weights / self.num_intent_classes

        gate_inertia, gate_anchor, gate_confidence, gate_mode, gate_mode_effective = self.physics_gate(
            last_encoded=last_encoded,
            intent_weights=intent_weights,
            step_encoding=step_encoding,
        )

        physics_trajectory = self.physics_model.multi_step(
            historical_trajectory, pred_len=P
        )  # (B, P, 3)

        # Anchor pull-back vector
        last_pos = historical_trajectory[:, -1, :3]                  # (B, 3)
        last_pos_expanded = last_pos.unsqueeze(1).expand(-1, P, -1)  # (B, P, 3)
        anchor_pull = (anchor_pos.unsqueeze(1) - last_pos_expanded)  # (B, P, 3)
        anchor_pull = anchor_pull * gate_anchor.unsqueeze(-1)         # (B, P, 3)

        confidence_factor = gate_confidence.unsqueeze(-1)             # (B, P, 1)
        gate_mode_exp = gate_mode_effective.unsqueeze(-1)             # (B, P, 1)
        inertia_effective = gate_inertia.unsqueeze(-1) * (1.0 - 0.3 * gate_mode_exp)

        # ---- Multi-hypothesis decoding ----
        if not isinstance(self.neural_decoder, MultiHeadNeuralDecoder):
            raise RuntimeError(
                'forward_multi_head requires MultiHeadNeuralDecoder. '
                'Call replace_with_multi_head() first.'
            )

        mh_out = self.neural_decoder(
            encoded=last_encoded,
            step_encoding=step_encoding,
            return_all=True,
        )
        deltas = mh_out['deltas']      # (K, B, P, 3)
        logvars = mh_out['logvars']    # (K, B, P, 3)
        confidences = mh_out['confidences']  # (B, K)

        K = deltas.shape[0]

        # Blend each hypothesis separately
        all_blended = []
        for k in range(K):
            neural_delta_k = deltas[k]                                 # (B, P, 3)
            neural_guided = (neural_delta_k + anchor_pull) * confidence_factor

            blended_k = (
                inertia_effective * physics_trajectory
                + (1.0 - inertia_effective) * neural_guided
            )
            all_blended.append(blended_k)

        all_blended = torch.stack(all_blended, dim=0)  # (K, B, P, 3)
        all_logvars = logvars.clamp(-10.0, 10.0)       # (K, B, P, 3)

        # De-normalization: consistent with Step 5 in model.py forward()
        _scale_pos = 100.0
        all_blended = all_blended * _scale_pos
        physics_trajectory = physics_trajectory * _scale_pos

        # Best hypothesis
        best_idx = confidences.argmax(dim=1)            # (B,)
        best_pred = all_blended[best_idx, torch.arange(B, device=device)]  # (B, P, 3)
        best_logvar = all_logvars[best_idx, torch.arange(B, device=device)]

        return {
            'predictions': best_pred,          # (B, P, 3) highest-confidence trajectory (meters)
            'all_predictions': all_blended,    # (K, B, P, 3) K trajectories (meters)
            'all_logvars': all_logvars,        # (K, B, P, 3)
            'confidences': confidences,        # (B, K)
            'logvar': best_logvar,             # (B, P, 3)
            'gate_inertia': gate_inertia,
            'gate_anchor': gate_anchor,
            'physics_trajectory': physics_trajectory,
            'neural_delta': deltas[0] * _scale_pos,  # first head for compatibility
        }

    def replace_with_multi_head(self, K: int, noise_std: float = 0.02):
        """Replace NeuralDecoder with MultiHeadNeuralDecoder and initialize weights."""
        old_decoder = self.neural_decoder
        new_decoder = MultiHeadNeuralDecoder(
            d_model=self.d_model,
            trajectory_dim=self.traj_dim,
            K=K,
        )

        # Copy projection layer
        for p_old, p_new in zip(old_decoder.proj.parameters(), new_decoder.proj.parameters()):
            if p_old.shape == p_new.shape:
                p_new.data.copy_(p_old.data)

        # Copy and perturb the K heads
        orig_w = old_decoder.delta_head.weight.data.clone()
        orig_b = old_decoder.delta_head.bias.data.clone()
        w_std = orig_w.std()
        b_std = orig_b.std()

        for k in range(K):
            nw = torch.randn_like(orig_w) * noise_std * w_std
            nb = torch.randn_like(orig_b) * noise_std * b_std
            new_decoder.delta_heads[k].weight.data.copy_(orig_w + nw)
            new_decoder.delta_heads[k].bias.data.copy_(orig_b + nb)

            for p_old, p_new in zip(old_decoder.var_head.parameters(),
                                      new_decoder.var_heads[k].parameters()):
                if p_old.shape == p_new.shape:
                    noise = torch.randn_like(p_old) * noise_std * p_old.std()
                    p_new.data.copy_(p_old.data + noise)

        self.neural_decoder = new_decoder
        return new_decoder

    def _kinematic_postprocess(
        self,
        predictions: torch.Tensor,
        history: torch.Tensor,
        max_accel: float = 15.0,
        max_jerk: float = 30.0,
    ) -> torch.Tensor:
        """
        Inference-time kinematic post-processing (enforces physical consistency).

        Applies kinematic feasibility checks to the predicted trajectory:
        1. Acceleration upper bound
        2. Velocity upper bound
        3. Position jump check

        Called only at inference; does not affect training gradients.
        """
        dt = 0.1
        preds = predictions.clone()

        # Velocity constraint: max speed 50 m/s
        max_vel = 50.0

        # Compute predicted velocity and acceleration
        pos_history = history[:, -1, :3]                      # (B, 3)
        pos_series = torch.cat([pos_history.unsqueeze(1), preds], dim=1)  # (B, P+1, 3)

        for step in range(preds.shape[1]):
            pos_curr = pos_series[:, step + 1, :]
            pos_prev = pos_series[:, step, :]

            # Velocity
            vel = (pos_curr - pos_prev) / dt
            vel_norm = torch.norm(vel, dim=-1, keepdim=True)  # (B, 1)
            vel = vel / (vel_norm.clamp(min=1e-6)) * vel_norm.clamp(max=max_vel)
            vel_clamped = vel

            # Acceleration
            if step > 0:
                pos_pp = pos_series[:, step - 1, :]
                vel_prev = (pos_prev - pos_pp) / dt
                acc = (vel_clamped - vel_prev) / dt
                acc_norm = torch.norm(acc, dim=-1, keepdim=True)

                # Acceleration constraint
                acc = acc / (acc_norm.clamp(min=1e-6)) * acc_norm.clamp(max=max_accel)

                # Correct position: recompute using the constrained acceleration
                pos_corrected = pos_prev + vel_prev * dt + 0.5 * acc * (dt ** 2)
                pos_series[:, step + 1, :] = pos_corrected

        return pos_series[:, 1:, :]  # (B, P, 3)

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        physics_trajectory: torch.Tensor,
        targets: torch.Tensor,
        gate_inertia: torch.Tensor,
    ) -> torch.Tensor:
        """
        Physics consistency loss: encourages predictions to stay close to physics extrapolation
        in high-inertia regions (maneuvers) and close to the neural prediction in low-inertia regions (cruise).

        loss = |pred - physics| * gate_inertia + |pred - neural| * (1 - gate_inertia)

        Since neural_delta is unknown here, this is simplified to:
        loss = |pred - physics| * gate_inertia.mean()

        i.e., during maneuvers (high gate_inertia), force predictions to stay close to the physics model.
        """
        residual = (predictions - physics_trajectory).abs()   # (B, P, 3)
        physics_loss = (residual * gate_inertia.unsqueeze(-1)).mean()
        return physics_loss * self.physics_loss_weight
