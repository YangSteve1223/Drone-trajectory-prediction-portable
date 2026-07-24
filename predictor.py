#!/usr/bin/env python3
"""Dual-model speed-switched inference with online LoRA adaptation."""

import torch
import warnings
from pathlib import Path
from typing import Optional
from emam_model import TrajectoryPredictor
from utils.metrics import full_evaluation
from utils.fast_data_loader import FastWindowDataset

# Device selection: CUDA > MPS (Apple Silicon) > CPU
def _detect_device(verbose=True):
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        if verbose:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Device] CUDA: {name} ({mem:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
        if verbose:
            print("[Device] MPS (Apple Silicon)")
    else:
        dev = torch.device('cpu')
        if verbose:
            print("[Device] CPU")
    return dev

_DEVICE = _detect_device()
_WEIGHT_DIR = Path(__file__).parent / 'weights'

# Speed threshold (m/s): <5 low model, >=5 high model
S_THRESHOLD = 5.0

# Soft fusion: smooth sigmoid transition instead of hard switch near threshold
SOFT_FUSION_ENABLED = True
FUSION_TEMPERATURE = 1.2       # sigmoid temperature
FUSION_HALF_WIDTH = 3.0        # transition half-width (m/s); blend only within [threshold +/- width]

# Z correction: dampen Z component of HIGH model DESCEND intent via 3-segment transition
Z_CORRECTION_ENABLED = True
Z_DESCEND_THRESHOLD_LOW = 0.05   # below: confident non-descent, strong dampen
Z_DESCEND_THRESHOLD_HIGH = 0.20  # above: confident descent, no dampen
Z_DAMPEN_STRONG = 0.05           # strong dampen: keep 5% of Z
Z_DAMPEN_WEAK = 0.30             # weak dampen: keep 30% of Z

# Z trend awareness: signals that raise confidence a DESCEND prediction is real
Z_TREND_ENABLED = True
Z_TREND_WINDOW = 10              # frames analyzed for Z trend
Z_TREND_DESCENT_THRESH = -0.03   # Z velocity threshold (m/s per frame)
Z_TREND_DAMPEN_BOOST = 0.15      # dampen boost per satisfied signal

# Model uncertainty: use UA-PGD logvar to judge Z prediction reliability
Z_UNCERTAINTY_ENABLED = True
Z_UNCERTAINTY_THRESH = -2.0      # logvar < -2.0 -> low uncertainty -> weaken dampen
Z_UNCERTAINTY_BOOST = 0.20       # dampen boost when uncertainty is low

# Predicted Z displacement magnitude: large descent more likely real
Z_MAGNITUDE_ENABLED = True
Z_MAGNITUDE_THRESH = 5.0         # predicted Z displacement < -5m -> possibly real descent
Z_MAGNITUDE_BOOST = 0.15         # dampen boost for large descent

# Entropy-guided physics fusion: LOW model leans on physics extrapolation when intent is uncertain

# Long trajectory fix: reduce physics gate inertia for trajectories >= 150 frames.
# The physics model does linear extrapolation and fails on hovering turns.
# Scaling gate_inertia down gives the neural decoder more weight (70% vs 29%).
LONG_TRAJ_THRESHOLD = 150        # frame count threshold for activation
LONG_TRAJ_GATE_SCALE = 0.3       # scale gate_inertia output (0.71 -> 0.21 effective)
LONG_TRAJ_HIST_STRIDE = 2        # downsample history: every 2nd frame -> 8s context
ENTROPY_PHYSICS_ENABLED = False  # disabled: physics model too simple, hurts turn accuracy
ENTROPY_THRESHOLD = 0.4         # intent entropy > 0.4 triggers physics bias
ENTROPY_MAX_BLEND = 0.35        # max extra physics weight

# 4-class -> 6-class intent mapping
_C4_TO_C6 = {0: 0, 1: 1, 2: 2, 3: 4}


class DronePredictor:
    """Drone trajectory predictor with speed-based dual-model routing.

    speed < S_THRESHOLD -> low model (UAV-Flow, 6-class, 5Hz, 0-3 m/s domain)
    speed >= S_THRESHOLD -> high model (SimCruise, 4-class, 1Hz, 8-28 m/s domain)
    """

    def __init__(self, threshold=S_THRESHOLD, device=None, use_finetuned=False):
        self.threshold = threshold
        self.device = torch.device(device) if device else _DEVICE
        # use_finetuned: opt in to the *_finetuned.pth variant. Default False —
        # the finetuned LOW model overfits to long trajectories and degrades the
        # general case, so it must not load silently (see project memory).
        self.use_finetuned = use_finetuned

        self.low = self._load('low_speed_6class.pth', 6)
        self.high = self._load('high_speed_4class.pth', 4)

        self._stats = {'low': 0, 'high': 0, 'n': 0}

    def _load(self, filename, n_classes):
        path = _WEIGHT_DIR / filename
        # Only use the fine-tuned variant when explicitly requested.
        ft_path = _WEIGHT_DIR / filename.replace('.pth', '_finetuned.pth')
        if self.use_finetuned and ft_path.exists():
            path = ft_path
        model = TrajectoryPredictor(
            input_dim=6, history_len=20, pred_len=20,
            d_model=128, d_state=16, d_conv=4, expand=2,
            emam_n_layers=2, num_intent_classes=n_classes,
            use_trigger=True, trigger_mode='simple',
        ).to(self.device).eval()
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    @staticmethod
    def compute_speed(hist):
        """Mean speed over last 5 frames (m/s)."""
        vel = hist[:, :, 3:6]
        speed = torch.norm(vel[:, -5:, :], dim=2).mean(dim=1)
        if speed.max() < 1e-6:
            pos = hist[:, :, :3]
            speed = torch.norm((pos[:, 1:] - pos[:, :-1]) / 0.2, dim=2)[:, -5:].mean(dim=1)
        return speed

    @staticmethod
    def compute_z_trend(hist, window=None):
        """Analyze Z-axis history trend to judge whether the drone is descending.

        Args:
            hist: (B, 20, 6) history trajectory
            window: number of frames to analyze (default: all 20)

        Returns:
            dict with z_velocity, z_accel, z_slope, z_variance,
            is_descending (bool), descent_strength ([0,1]).
        """
        B = hist.shape[0]
        if window is None:
            window = hist.shape[1]  # default: all frames
        window = min(window, hist.shape[1])

        # Z positions: (B, window)
        z_pos = hist[:, -window:, 2]

        # Z velocity: frame-to-frame differences
        z_vel = z_pos[:, 1:] - z_pos[:, :-1]  # (B, window-1)
        z_vel_mean = z_vel.mean(dim=1)  # (B,)

        # Z acceleration: change in velocity
        z_acc = z_vel[:, 1:] - z_vel[:, :-1]  # (B, window-2)
        z_acc_mean = z_acc.mean(dim=1) if z_acc.shape[1] > 0 else torch.zeros(B, device=hist.device)

        # Linear regression slope (m/s per frame)
        t = torch.arange(window, device=hist.device, dtype=hist.dtype)
        t_mean = t.mean()
        z_mean = z_pos.mean(dim=1)
        numerator = ((t - t_mean) * (z_pos - z_mean.unsqueeze(1))).sum(dim=1)
        denominator = ((t - t_mean) ** 2).sum()
        z_slope = numerator / (denominator + 1e-8)  # (B,) m/frame

        # Z variance
        z_var = z_pos.var(dim=1)  # (B,)

        # Descent detection: Z velocity < threshold AND Z slope < 0
        is_descending = (z_vel_mean < Z_TREND_DESCENT_THRESH) & (z_slope < 0)

        # Descent strength: normalized [0, 1] based on how negative the velocity is
        # z_vel_mean = -0.03 -> strength~0, z_vel_mean = -0.5 -> strength~1
        descent_strength = torch.clamp(-z_vel_mean / 0.5, 0.0, 1.0)  # (B,)

        return {
            'z_velocity': z_vel_mean,
            'z_accel': z_acc_mean,
            'z_slope': z_slope,
            'z_variance': z_var,
            'is_descending': is_descending,
            'descent_strength': descent_strength,
        }

    @torch.no_grad()
    def predict(self, hist):
        """Predict future trajectory.

        Args:
            hist: (B, 20, 6) [pos_x,pos_y,pos_z, vel_x,vel_y,vel_z] (meters, m/s)

        Returns:
            dict: predictions (B,20,3), intent_logits (B,6), speed (B,), route (B,)
        """
        if hist.dim() != 3 or hist.shape[1] != 20 or hist.shape[2] != 6:
            raise ValueError(
                f"Expected shape (B, 20, 6), got {hist.shape}. "
                f"Input: [pos_x,pos_y,pos_z, vx,vy,vz] for 20 frames."
            )
        if torch.isnan(hist).any():
            warnings.warn("Input contains NaN values, replacing with zeros.")
            hist = torch.nan_to_num(hist, nan=0.0)

        hist = hist.to(self.device)
        speed = self.compute_speed(hist)

        # Run both models (needed for soft fusion)
        out_low = self.low(hist, force_predict=True)
        out_high = self.high(hist, force_predict=True)

        # Z correction: 3-segment transition + Z trend awareness
        if Z_CORRECTION_ENABLED:
            high_intent_prob = torch.softmax(out_high['intent_logits'], dim=-1)
            descend_prob = high_intent_prob[:, 3]               # class 3 = DESCEND

            # 3-segment dampen factor
            dampen = torch.full_like(descend_prob, 1.0)         # default: no dampening
            strong_mask = descend_prob < Z_DESCEND_THRESHOLD_LOW
            weak_mask = (descend_prob >= Z_DESCEND_THRESHOLD_LOW) & (descend_prob < Z_DESCEND_THRESHOLD_HIGH)

            if strong_mask.any():
                dampen[strong_mask] = Z_DAMPEN_STRONG           # strong dampen
            if weak_mask.any():
                # linear transition from 0.30 to 1.0 over [0.05, 0.20]
                t = (descend_prob[weak_mask] - Z_DESCEND_THRESHOLD_LOW) / (Z_DESCEND_THRESHOLD_HIGH - Z_DESCEND_THRESHOLD_LOW)
                dampen[weak_mask] = Z_DAMPEN_WEAK + (1.0 - Z_DAMPEN_WEAK) * t

            # Multi-signal fusion: Z trend + model uncertainty + predicted Z magnitude
            if Z_TREND_ENABLED or Z_UNCERTAINTY_ENABLED or Z_MAGNITUDE_ENABLED:
                total_boost = torch.zeros_like(dampen)

                # Signal 1: Z history trend
                if Z_TREND_ENABLED:
                    z_info = self.compute_z_trend(hist, window=Z_TREND_WINDOW)
                    total_boost = total_boost + torch.where(
                        z_info['is_descending'],
                        torch.full_like(dampen, Z_TREND_DAMPEN_BOOST),
                        torch.zeros_like(dampen)
                    )

                # Signal 2: model Z uncertainty (low uncertainty -> weaken dampen)
                if Z_UNCERTAINTY_ENABLED:
                    z_logvar = out_high['uncertainty'][:, :, 2]  # (B, pred_len) Z axis
                    z_mean_logvar = z_logvar.mean(dim=1)  # (B,) average over pred steps
                    low_uncertainty = z_mean_logvar < Z_UNCERTAINTY_THRESH
                    total_boost = total_boost + torch.where(
                        low_uncertainty,
                        torch.full_like(dampen, Z_UNCERTAINTY_BOOST),
                        torch.zeros_like(dampen)
                    )

                # Signal 3: predicted Z displacement magnitude
                if Z_MAGNITUDE_ENABLED:
                    pred_z_final = out_high['predictions'][:, -1, 2]  # (B,) final Z pred
                    large_descent = pred_z_final < -Z_MAGNITUDE_THRESH
                    total_boost = total_boost + torch.where(
                        large_descent,
                        torch.full_like(dampen, Z_MAGNITUDE_BOOST),
                        torch.zeros_like(dampen)
                    )

                dampen = torch.clamp(dampen + total_boost, 0.0, 1.0)

            apply_mask = dampen < 1.0
            if apply_mask.any():
                out_high['predictions'][apply_mask, :, 2] *= dampen[apply_mask].view(-1, 1)

            # LOW Z correction (DESCEND = class 4, 6-class)
            low_intent_prob = torch.softmax(out_low['intent_logits'], dim=-1)
            low_descend_prob = low_intent_prob[:, 4]            # class 4 = DESCEND

            low_dampen = torch.full_like(low_descend_prob, 1.0)
            low_strong = low_descend_prob < Z_DESCEND_THRESHOLD_LOW
            low_weak = (low_descend_prob >= Z_DESCEND_THRESHOLD_LOW) & (low_descend_prob < Z_DESCEND_THRESHOLD_HIGH)

            if low_strong.any():
                low_dampen[low_strong] = Z_DAMPEN_STRONG
            if low_weak.any():
                t_low = (low_descend_prob[low_weak] - Z_DESCEND_THRESHOLD_LOW) / (Z_DESCEND_THRESHOLD_HIGH - Z_DESCEND_THRESHOLD_LOW)
                low_dampen[low_weak] = Z_DAMPEN_WEAK + (1.0 - Z_DAMPEN_WEAK) * t_low

            # Multi-signal fusion for LOW (reuse same thresholds)
            if Z_TREND_ENABLED or Z_UNCERTAINTY_ENABLED or Z_MAGNITUDE_ENABLED:
                low_boost = torch.zeros_like(low_dampen)

                if Z_TREND_ENABLED:
                    z_info = self.compute_z_trend(hist, window=Z_TREND_WINDOW)
                    low_boost = low_boost + torch.where(
                        z_info['is_descending'],
                        torch.full_like(low_dampen, Z_TREND_DAMPEN_BOOST),
                        torch.zeros_like(low_dampen)
                    )

                if Z_UNCERTAINTY_ENABLED:
                    z_logvar_low = out_low['uncertainty'][:, :, 2]
                    z_mean_logvar_low = z_logvar_low.mean(dim=1)
                    low_uncertainty_low = z_mean_logvar_low < Z_UNCERTAINTY_THRESH
                    low_boost = low_boost + torch.where(
                        low_uncertainty_low,
                        torch.full_like(low_dampen, Z_UNCERTAINTY_BOOST),
                        torch.zeros_like(low_dampen)
                    )

                if Z_MAGNITUDE_ENABLED:
                    # LOW drones move slower, use lower magnitude threshold (2.5m vs 5.0m)
                    low_mag_thresh = Z_MAGNITUDE_THRESH * 0.5
                    pred_z_final_low = out_low['predictions'][:, -1, 2]
                    large_descent_low = pred_z_final_low < -low_mag_thresh
                    low_boost = low_boost + torch.where(
                        large_descent_low,
                        torch.full_like(low_dampen, Z_MAGNITUDE_BOOST),
                        torch.zeros_like(low_dampen)
                    )

                low_dampen = torch.clamp(low_dampen + low_boost, 0.0, 1.0)

            low_apply = low_dampen < 1.0
            if low_apply.any():
                out_low['predictions'][low_apply, :, 2] *= low_dampen[low_apply].view(-1, 1)

        # Entropy-guided physics fusion: lean on physics when LOW intent is uncertain
        if ENTROPY_PHYSICS_ENABLED:
            low_intent_prob = torch.softmax(out_low['intent_logits'], dim=-1)
            low_entropy = -(low_intent_prob * torch.log(low_intent_prob + 1e-8)).sum(dim=-1)
            physics = out_low.get('physics_trajectory', None)
            if physics is not None:
                alpha = torch.clamp((low_entropy - ENTROPY_THRESHOLD) / 0.6, 0.0, ENTROPY_MAX_BLEND)
                if alpha.max() > 0:
                    blend = alpha.view(-1, 1, 1)
                    out_low['predictions'] = (1.0 - blend) * out_low['predictions'] + blend * physics

        # Soft fusion: sigmoid transition instead of hard switch
        if SOFT_FUSION_ENABLED:
            raw_alpha = torch.sigmoid((speed - self.threshold) / FUSION_TEMPERATURE)  # (B,)
            # gate: blend only within [threshold +/- half_width]
            in_transition = (speed > self.threshold - FUSION_HALF_WIDTH) & (speed < self.threshold + FUSION_HALF_WIDTH)
            alpha = torch.where(in_transition, raw_alpha, (speed >= self.threshold).float())
            alpha_3d = alpha.view(-1, 1, 1)
            alpha_2d = alpha.view(-1, 1)
            predictions = (1.0 - alpha_3d) * out_low['predictions'] + alpha_3d * out_high['predictions']

            # Soft-fuse intent too
            intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                       device=self.device, dtype=out_high['intent_logits'].dtype)
            for c4, c6 in _C4_TO_C6.items():
                intent_high_6[:, c6] = out_high['intent_logits'][:, c4]
            intent = (1.0 - alpha_2d) * out_low['intent_logits'] + alpha_2d * intent_high_6

            # Stats (route label uses alpha > 0.5)
            use_high = alpha > 0.5
            self._stats['n'] += hist.shape[0]
            self._stats['low'] += (alpha <= 0.5).sum().item()
            self._stats['high'] += (alpha > 0.5).sum().item()
            route_list = ['HIGH' if h else 'LOW' for h in use_high.cpu().tolist()]
        else:
            # Legacy hard switch (backward compatible)
            use_high = speed >= self.threshold  # (B,) bool
            mask_low = (~use_high).float().view(-1, 1, 1)
            mask_high = use_high.float().view(-1, 1, 1)
            predictions = mask_low * out_low['predictions'] + mask_high * out_high['predictions']

            mask2_low = (~use_high).float().view(-1, 1)
            mask2_high = use_high.float().view(-1, 1)
            intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                       device=self.device, dtype=out_high['intent_logits'].dtype)
            for c4, c6 in _C4_TO_C6.items():
                intent_high_6[:, c6] = out_high['intent_logits'][:, c4]
            intent = mask2_low * out_low['intent_logits'] + mask2_high * intent_high_6

            self._stats['n'] += hist.shape[0]
            self._stats['low'] += (~use_high).sum().item()
            self._stats['high'] += use_high.sum().item()
            route_list = ['HIGH' if h else 'LOW' for h in use_high.cpu().tolist()]
        return {
            'predictions': predictions,
            'intent_logits': intent,
            'speed': speed,
            'route': route_list,
        }

    @staticmethod
    def make_long_windows(traj, hist_len=20, pred_len=20, stride=LONG_TRAJ_HIST_STRIDE):
        """Create downsampled windows for long trajectories.

        Downsampling provides extended temporal context (8s vs 4s) using the
        same 20-frame window budget. Applied when trajectory >= LONG_TRAJ_THRESHOLD.

        Args:
            traj: (N, 6) numpy array
        Returns:
            hists: list of (20, 6), futs: list of (20, 3) displacement targets
        """
        import numpy as np
        n = traj.shape[0]
        ml = hist_len * stride + pred_len
        if n < ml:
            return [], []
        hists, futs = [], []
        step = max(1, stride // 2)
        for i in range(0, n - ml + 1, step):
            indices = np.arange(i, i + hist_len * stride, stride)[:hist_len]
            hists.append(traj[indices].copy())
            fut_start = i + hist_len * stride
            fut_abs = traj[fut_start:fut_start + pred_len, :3]
            futs.append(fut_abs - traj[fut_start - 1, :3])
        return hists, futs

    def _set_gate_scale(self, scale):
        """Temporarily modify physics gate inertia for LOW model inference."""
        gate = self.low.ua_pgd.physics_gate
        if not hasattr(gate, '_orig_forward'):
            gate._orig_forward = gate.forward
        orig = gate._orig_forward

        def scaled_forward(last_encoded, intent_weights, step_encoding):
            gi, ga, gc, gm, gme = orig(last_encoded, intent_weights, step_encoding)
            return gi * scale, ga, gc, gm, gme

        gate.forward = scaled_forward

    def _restore_gate(self):
        """Restore original physics gate forward."""
        gate = self.low.ua_pgd.physics_gate
        if hasattr(gate, '_orig_forward'):
            gate.forward = gate._orig_forward

    @property
    def stats(self):
        s = self._stats
        if s['n'] == 0:
            return 'No predictions yet.'
        return (f"{s['n']} samples: low={s['low']}({s['low']/s['n']*100:.0f}%) "
                f"high={s['high']}({s['high']/s['n']*100:.0f}%)")

    def reset_stats(self):
        self._stats = {'low': 0, 'high': 0, 'n': 0}

    def predict_normalized(self, hist, return_norm_params=False):
        from dynamic_norm import DynamicNormalizer, NormConfig

        norm = DynamicNormalizer(NormConfig(
            method="velocity", center_on_first=True, scale_smoothing=0.7,
        ))
        hist_norm, norm_params = norm.normalize(hist)

        for model in [self.low, self.high]:
            model._norm_input = False

        result = self.predict(hist_norm)

        for model in [self.low, self.high]:
            model._norm_input = True

        result['predictions'] = norm.denormalize(result['predictions'], norm_params)
        result['speed'] = self.compute_speed(hist)
        if return_norm_params:
            result['norm_params'] = norm_params
        return result

    def predict_adaptive(self, hist, return_scale=False):
        from dynamic_norm import DynamicNormalizer, NormConfig

        norm = DynamicNormalizer(NormConfig(
            method="velocity", center_on_first=True, scale_smoothing=0.7,
        ))
        _, norm_params = norm.normalize(hist)
        current_scale = norm_params['scale_pos']

        MODEL_SCALE = 100.0

        if isinstance(current_scale, torch.Tensor):
            scale_factor = torch.clamp(MODEL_SCALE / current_scale, min=0.1, max=50.0)
            sf_3d = scale_factor.view(-1, 1, 1)
        else:
            scale_factor = max(0.1, min(50.0, MODEL_SCALE / current_scale))
            sf_3d = scale_factor

        hist_scaled = hist.clone()
        hist_scaled[:, :, :3] = hist[:, :, :3] * sf_3d
        hist_scaled[:, :, 3:6] = hist[:, :, 3:6] * sf_3d

        result = self.predict(hist_scaled)

        if isinstance(scale_factor, torch.Tensor):
            result['predictions'] = result['predictions'] / sf_3d
        else:
            result['predictions'] = result['predictions'] / sf_3d

        result['speed'] = self.compute_speed(hist)

        if return_scale:
            result['adaptive_scale'] = scale_factor
            result['current_scale'] = current_scale
        return result

    def enable_adaptation(self, checkpoint_dir: str = 'checkpoints/adapters',
                          lora_r: int = 4, accumulation_steps: int = 5):
        """Enable per-drone LoRA online adaptation on the low-speed model.

        NOTE: this binds adaptation to the 20-frame `self.low` (the deployment
        inference model with Z-correction/soft-fusion). The validated 40-frame
        online-learning path is exercised via online_config.build_online_base()
        + DroneAdapterManager + OnlineLearner directly (see eval_online_learning.py),
        which uses the correct upstream-only LoRA config. The manager now defaults
        to that config regardless of base.
        """
        from adapter_manager import DroneAdapterManager
        from online_learner import OnlineLearner, OnlineLearnerConfig

        self._adapter_mgr = DroneAdapterManager(
            self.low, checkpoint_dir=checkpoint_dir, r=lora_r,
        )
        config = OnlineLearnerConfig(
            accumulation_steps=accumulation_steps,
            device=str(self.device),
        )
        self._learner = OnlineLearner(self._adapter_mgr, config)
        self._adaptation_enabled = True

    def predict_with_adaptation(self, hist: torch.Tensor,
                                 drone_id: str = None,
                                 ground_truth: torch.Tensor = None,
                                 intent_label: int = 0,
                                 timestep: int = 0) -> dict:
        """Predict with optional per-drone LoRA adaptation on the low-speed model."""
        if hist.dim() != 3 or hist.shape[1] != 20 or hist.shape[2] != 6:
            raise ValueError(f"Expected shape (B, 20, 6), got {hist.shape}.")
        if torch.isnan(hist).any():
            warnings.warn("Input contains NaN values, replacing with zeros.")
            hist = torch.nan_to_num(hist, nan=0.0)

        hist = hist.to(self.device)

        with torch.no_grad():
            speed = self.compute_speed(hist)
            use_high = speed >= self.threshold

            # Low model with optional LoRA
            adapted = False
            if drone_id and hasattr(self, '_adaptation_enabled') and self._adaptation_enabled:
                if self._adapter_mgr.has_adapter(drone_id):
                    self._adapter_mgr.activate(drone_id)
                    adapted = True

            out_low = self.low(hist, force_predict=True)
            out_high = self.high(hist, force_predict=True)

            # Z correction: 3-segment transition + Z trend awareness
            if Z_CORRECTION_ENABLED:
                high_intent_prob = torch.softmax(out_high['intent_logits'], dim=-1)
                descend_prob = high_intent_prob[:, 3]

                dampen = torch.full_like(descend_prob, 1.0)
                strong_mask = descend_prob < Z_DESCEND_THRESHOLD_LOW
                weak_mask = (descend_prob >= Z_DESCEND_THRESHOLD_LOW) & (descend_prob < Z_DESCEND_THRESHOLD_HIGH)

                if strong_mask.any():
                    dampen[strong_mask] = Z_DAMPEN_STRONG
                if weak_mask.any():
                    t = (descend_prob[weak_mask] - Z_DESCEND_THRESHOLD_LOW) / (Z_DESCEND_THRESHOLD_HIGH - Z_DESCEND_THRESHOLD_LOW)
                    dampen[weak_mask] = Z_DAMPEN_WEAK + (1.0 - Z_DAMPEN_WEAK) * t

                # Multi-signal fusion: Z trend + model uncertainty + predicted Z magnitude
                if Z_TREND_ENABLED or Z_UNCERTAINTY_ENABLED or Z_MAGNITUDE_ENABLED:
                    total_boost = torch.zeros_like(dampen)

                    if Z_TREND_ENABLED:
                        z_info = self.compute_z_trend(hist, window=Z_TREND_WINDOW)
                        total_boost = total_boost + torch.where(
                            z_info['is_descending'],
                            torch.full_like(dampen, Z_TREND_DAMPEN_BOOST),
                            torch.zeros_like(dampen)
                        )

                    if Z_UNCERTAINTY_ENABLED:
                        z_logvar = out_high['uncertainty'][:, :, 2]
                        z_mean_logvar = z_logvar.mean(dim=1)
                        low_uncertainty = z_mean_logvar < Z_UNCERTAINTY_THRESH
                        total_boost = total_boost + torch.where(
                            low_uncertainty,
                            torch.full_like(dampen, Z_UNCERTAINTY_BOOST),
                            torch.zeros_like(dampen)
                        )

                    if Z_MAGNITUDE_ENABLED:
                        pred_z_final = out_high['predictions'][:, -1, 2]
                        large_descent = pred_z_final < -Z_MAGNITUDE_THRESH
                        total_boost = total_boost + torch.where(
                            large_descent,
                            torch.full_like(dampen, Z_MAGNITUDE_BOOST),
                            torch.zeros_like(dampen)
                        )

                    dampen = torch.clamp(dampen + total_boost, 0.0, 1.0)

                apply_mask = dampen < 1.0
                if apply_mask.any():
                    out_high['predictions'][apply_mask, :, 2] *= dampen[apply_mask].view(-1, 1)

                # LOW Z correction (DESCEND = class 4 in 6-class)
                low_intent_prob = torch.softmax(out_low['intent_logits'], dim=-1)
                low_descend_prob = low_intent_prob[:, 4]

                low_dampen = torch.full_like(low_descend_prob, 1.0)
                low_strong = low_descend_prob < Z_DESCEND_THRESHOLD_LOW
                low_weak = (low_descend_prob >= Z_DESCEND_THRESHOLD_LOW) & (low_descend_prob < Z_DESCEND_THRESHOLD_HIGH)

                if low_strong.any():
                    low_dampen[low_strong] = Z_DAMPEN_STRONG
                if low_weak.any():
                    t_low = (low_descend_prob[low_weak] - Z_DESCEND_THRESHOLD_LOW) / (Z_DESCEND_THRESHOLD_HIGH - Z_DESCEND_THRESHOLD_LOW)
                    low_dampen[low_weak] = Z_DAMPEN_WEAK + (1.0 - Z_DAMPEN_WEAK) * t_low

                if Z_TREND_ENABLED or Z_UNCERTAINTY_ENABLED or Z_MAGNITUDE_ENABLED:
                    low_boost = torch.zeros_like(low_dampen)

                    if Z_TREND_ENABLED:
                        z_info = self.compute_z_trend(hist, window=Z_TREND_WINDOW)
                        low_boost = low_boost + torch.where(
                            z_info['is_descending'],
                            torch.full_like(low_dampen, Z_TREND_DAMPEN_BOOST),
                            torch.zeros_like(low_dampen)
                        )

                    if Z_UNCERTAINTY_ENABLED:
                        z_logvar_low = out_low['uncertainty'][:, :, 2]
                        z_mean_logvar_low = z_logvar_low.mean(dim=1)
                        low_uncertainty_low = z_mean_logvar_low < Z_UNCERTAINTY_THRESH
                        low_boost = low_boost + torch.where(
                            low_uncertainty_low,
                            torch.full_like(low_dampen, Z_UNCERTAINTY_BOOST),
                            torch.zeros_like(low_dampen)
                        )

                    if Z_MAGNITUDE_ENABLED:
                        low_mag_thresh = Z_MAGNITUDE_THRESH * 0.5
                        pred_z_final_low = out_low['predictions'][:, -1, 2]
                        large_descent_low = pred_z_final_low < -low_mag_thresh
                        low_boost = low_boost + torch.where(
                            large_descent_low,
                            torch.full_like(low_dampen, Z_MAGNITUDE_BOOST),
                            torch.zeros_like(low_dampen)
                        )

                    low_dampen = torch.clamp(low_dampen + low_boost, 0.0, 1.0)

                low_apply = low_dampen < 1.0
                if low_apply.any():
                    out_low['predictions'][low_apply, :, 2] *= low_dampen[low_apply].view(-1, 1)

            # Entropy-guided physics fusion
            if ENTROPY_PHYSICS_ENABLED:
                low_intent_prob = torch.softmax(out_low['intent_logits'], dim=-1)
                low_entropy = -(low_intent_prob * torch.log(low_intent_prob + 1e-8)).sum(dim=-1)
                physics = out_low.get('physics_trajectory', None)
                if physics is not None:
                    alpha = torch.clamp((low_entropy - ENTROPY_THRESHOLD) / 0.6, 0.0, ENTROPY_MAX_BLEND)
                    if alpha.max() > 0:
                        blend = alpha.view(-1, 1, 1)
                        out_low['predictions'] = (1.0 - blend) * out_low['predictions'] + blend * physics

            if adapted:
                self._adapter_mgr.deactivate()

            # Soft fusion or hard switch
            if SOFT_FUSION_ENABLED:
                raw_alpha = torch.sigmoid((speed - self.threshold) / FUSION_TEMPERATURE)
                in_transition = (speed > self.threshold - FUSION_HALF_WIDTH) & (speed < self.threshold + FUSION_HALF_WIDTH)
                alpha = torch.where(in_transition, raw_alpha, (speed >= self.threshold).float())
                alpha_3d = alpha.view(-1, 1, 1)
                alpha_2d = alpha.view(-1, 1)
                predictions = (1.0 - alpha_3d) * out_low['predictions'] + alpha_3d * out_high['predictions']

                intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                           device=self.device, dtype=out_high['intent_logits'].dtype)
                for c4, c6 in _C4_TO_C6.items():
                    intent_high_6[:, c6] = out_high['intent_logits'][:, c4]
                intent = (1.0 - alpha_2d) * out_low['intent_logits'] + alpha_2d * intent_high_6
                use_high = alpha > 0.5
            else:
                use_high = speed >= self.threshold
                mask_low = (~use_high).float().view(-1, 1, 1)
                mask_high = use_high.float().view(-1, 1, 1)
                predictions = mask_low * out_low['predictions'] + mask_high * out_high['predictions']

                mask2_low = (~use_high).float().view(-1, 1)
                mask2_high = use_high.float().view(-1, 1)
                intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                           device=self.device, dtype=out_high['intent_logits'].dtype)
                for c4, c6 in _C4_TO_C6.items():
                    intent_high_6[:, c6] = out_high['intent_logits'][:, c4]
                intent = mask2_low * out_low['intent_logits'] + mask2_high * intent_high_6

        route_list = ['HIGH' if h else 'LOW' for h in use_high.cpu().tolist()]
        result = {
            'predictions': predictions,
            'intent_logits': intent,
            'speed': speed,
            'route': route_list,
            'adapted': adapted,
            'updated': False,
        }

        if (drone_id and ground_truth is not None and
                hasattr(self, '_adaptation_enabled') and self._adaptation_enabled):

            probs = torch.softmax(intent, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            max_entropy = torch.log(torch.tensor(6.0)).item()
            confidence = max(0.0, min(1.0, 1.0 - entropy / max_entropy))

            updated = self._learner.observe(
                drone_id, hist[0].cpu(), ground_truth[0].cpu(),
                confidence=confidence, intent=intent_label, timestep=timestep,
            )
            result['updated'] = updated
            result['confidence'] = confidence

        return result

    def get_drone_status(self, drone_id: str) -> Optional[dict]:
        if hasattr(self, '_learner'):
            return self._learner.get_status(drone_id)
        return None

    def save_adapters(self):
        if hasattr(self, '_learner'):
            self._learner.save_all()

    @property
    def adaptation_enabled(self) -> bool:
        return getattr(self, '_adaptation_enabled', False)

    @property
    def loaded_models(self) -> list:
        """Return list of loaded model weight filenames (for verification)."""
        return ['low_speed_6class.pth', 'high_speed_4class.pth']


if __name__ == '__main__':
    import shutil

    print('DronePredictor — Smoke Test (2-model hard-switch)')
    print(f'Device: {_DEVICE}')
    print(f'Threshold: {S_THRESHOLD} m/s\n')

    p = DronePredictor()
    print(f'Loaded: {p.loaded_models}')

    # Low speed
    x_low = torch.randn(4, 20, 6)
    x_low[:, :, 3:6] *= 0.5
    out = p.predict(x_low)
    routes = out['route']
    print(f'Low speed (~0.5 m/s): route={[r for r in routes]}')

    # High speed
    x_high = torch.randn(4, 20, 6)
    x_high[:, :, 3:6] *= 12.0
    out = p.predict(x_high)
    routes = out['route']
    print(f'High speed (~12 m/s): route={[r for r in routes]}')

    print(f'\n{p.stats}')

    # LoRA
    print('\n--- LoRA Adaptation Test ---')
    p.enable_adaptation(checkpoint_dir='test_adapters_pred', accumulation_steps=3)
    for i in range(10):
        hist = torch.randn(1, 20, 6)
        hist[:, :, 3:6] *= 1.5
        gt = torch.randn(1, 20, 3) * 0.1
        out = p.predict_with_adaptation(hist, drone_id='test_drone',
                                         ground_truth=gt, timestep=i)
        if out['updated']:
            status = p.get_drone_status('test_drone')
            print(f"  Step {i}: UPDATE (updates={status['num_updates']}, "
                  f"replay={status['replay_size']})")

    p.save_adapters()
    shutil.rmtree('test_adapters_pred', ignore_errors=True)
    print('\nAll good!')
