#!/usr/bin/env python3
"""三模型速度自适应软切换推理 + LoRA在线持续学习。"""

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

# 速度阈值 (真实物理 m/s)
S_LOW = 2.0
S_HIGH = 8.0

# 4-class → 6-class 意图映射
_C4_TO_C6 = {0: 0, 1: 1, 2: 2, 3: 4}


class DronePredictor:
    """无人机轨迹预测器 — 速度自适应三模型软切换。"""

    def __init__(self, s_low=S_LOW, s_high=S_HIGH, device=None):
        self.s_low = s_low
        self.s_high = s_high
        self.device = torch.device(device) if device else _DEVICE

        # 加载三模型
        self.low = self._load('low_speed_6class.pth', 6)
        self.mixed = self._load('mixed_6class.pth', 6)
        self.high = self._load('high_speed_4class.pth', 4)

        self._stats = {'low': 0, 'mixed': 0, 'high': 0, 'n': 0}

    def _load(self, filename, n_classes):
        path = _WEIGHT_DIR / filename
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
        """最近5帧平均速率 (m/s)"""
        vel = hist[:, :, 3:6]
        speed = torch.norm(vel[:, -5:, :], dim=2).mean(dim=1)
        if speed.max() < 1e-6:
            pos = hist[:, :, :3]
            speed = torch.norm((pos[:, 1:] - pos[:, :-1]) / 0.2, dim=2)[:, -5:].mean(dim=1)
        return speed

    def blend_weights(self, speed):
        """三角隶属三模型权重"""
        s, lo, hi = speed, self.s_low, self.s_high
        alpha = torch.clamp((s - lo) / (hi - lo), 0.0, 1.0)

        w_low = torch.clamp(1.0 - 2.0 * alpha, min=0.0)
        w_high = torch.clamp(2.0 * alpha - 1.0, min=0.0)
        w_mixed = 1.0 - 2.0 * torch.abs(alpha - 0.5)

        w_mixed = torch.where((s > lo) & (s < hi), w_mixed, torch.zeros_like(s))
        w_low = torch.where(s <= lo, torch.ones_like(s), w_low)
        w_high = torch.where(s >= hi, torch.ones_like(s), w_high)

        total = w_low + w_mixed + w_high + 1e-8
        w_low, w_mixed, w_high = w_low / total, w_mixed / total, w_high / total

        self._stats['n'] += len(s)
        self._stats['low'] += (w_low > 0.5).sum().item()
        self._stats['mixed'] += (w_mixed > 0.3).sum().item()
        self._stats['high'] += (w_high > 0.5).sum().item()

        return w_low, w_mixed, w_high

    @torch.no_grad()
    def predict(self, hist):
        """预测未来轨迹。

        Args:
            hist: (B, 20, 6) [pos_x,pos_y,pos_z, vel_x,vel_y,vel_z] (米, 米/秒)

        Returns:
            dict: predictions (B,20,3), intent_logits (B,6), speed (B,)
        """
        # Input validation
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
        w_low, w_mixed, w_high = self.blend_weights(speed)

        out_low = self.low(hist, force_predict=True)
        out_mixed = self.mixed(hist, force_predict=True)
        out_high = self.high(hist, force_predict=True)

        w3 = lambda w: w.view(-1, 1, 1)
        predictions = (w3(w_low) * out_low['predictions'] +
                       w3(w_mixed) * out_mixed['predictions'] +
                       w3(w_high) * out_high['predictions'])

        # 意图: low+mixed 原生6-class, high 需 4→6 映射
        w2 = lambda w: w.view(-1, 1)
        intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                   device=self.device, dtype=out_high['intent_logits'].dtype)
        for c4, c6 in _C4_TO_C6.items():
            intent_high_6[:, c6] = out_high['intent_logits'][:, c4]

        intent = (w2(w_low) * out_low['intent_logits'] +
                  w2(w_mixed) * out_mixed['intent_logits'] +
                  w2(w_high) * intent_high_6)

        return {'predictions': predictions, 'intent_logits': intent, 'speed': speed}

    @property
    def stats(self):
        s = self._stats
        if s['n'] == 0:
            return 'No predictions yet.'
        return (f"{s['n']} samples: low={s['low']}({s['low']/s['n']*100:.0f}%) "
                f"mixed={s['mixed']}({s['mixed']/s['n']*100:.0f}%) "
                f"high={s['high']}({s['high']/s['n']*100:.0f}%)")

    def reset_stats(self):
        self._stats = {'low': 0, 'mixed': 0, 'high': 0, 'n': 0}

    def predict_normalized(self, hist, return_norm_params=False):
        """Per-window dynamic normalization prediction. For FUTURE retrained models only.
        For existing weights, use predict_adaptive() instead."""
        from dynamic_norm import DynamicNormalizer, NormConfig

        norm = DynamicNormalizer(NormConfig(
            method="velocity", center_on_first=True, scale_smoothing=0.7,
        ))
        hist_norm, norm_params = norm.normalize(hist)

        for model in [self.low, self.mixed, self.high]:
            model._norm_input = False

        result = self.predict(hist_norm)

        for model in [self.low, self.mixed, self.high]:
            model._norm_input = True

        result['predictions'] = norm.denormalize(result['predictions'], norm_params)
        result['speed'] = self.compute_speed(hist)
        if return_norm_params:
            result['norm_params'] = norm_params
        return result

    def predict_adaptive(self, hist, return_scale=False):
        """Scale-adaptive prediction — works with EXISTING trained weights.
        Scales input to model's training distribution instead of bypassing internal /100 normalization."""
        from dynamic_norm import DynamicNormalizer, NormConfig

        # Compute current window scale (same as predict_normalized)
        norm = DynamicNormalizer(NormConfig(
            method="velocity", center_on_first=True, scale_smoothing=0.7,
        ))
        _, norm_params = norm.normalize(hist)
        current_scale = norm_params['scale_pos']

        # Model was trained with _scale_pos=100 — that's its comfortable range
        MODEL_SCALE = 100.0

        # Scale factor: bring input to model's training distribution
        # scale_factor > 1: small-domain data gets amplified (UAV-Flow)
        # scale_factor ≈ 1: large-domain data unchanged (NPZDATA)
        if isinstance(current_scale, torch.Tensor):
            scale_factor = torch.clamp(MODEL_SCALE / current_scale, min=0.1, max=50.0)
            sf_3d = scale_factor.view(-1, 1, 1)
        else:
            scale_factor = max(0.1, min(50.0, MODEL_SCALE / current_scale))
            sf_3d = scale_factor

        # Scale input to model's comfort zone
        hist_scaled = hist.clone()
        hist_scaled[:, :, :3] = hist[:, :, :3] * sf_3d
        hist_scaled[:, :, 3:6] = hist[:, :, 3:6] * sf_3d

        # Standard prediction (model's internal /100 works correctly now)
        result = self.predict(hist_scaled)

        # Undo scaling on output
        if isinstance(scale_factor, torch.Tensor):
            result['predictions'] = result['predictions'] / sf_3d
        else:
            result['predictions'] = result['predictions'] / sf_3d

        # Speed is computed from original (unscaled) input
        result['speed'] = self.compute_speed(hist)

        if return_scale:
            result['adaptive_scale'] = scale_factor
            result['current_scale'] = current_scale
        return result

    def enable_adaptation(self, checkpoint_dir: str = 'checkpoints/adapters',
                          lora_r: int = 4, accumulation_steps: int = 5):
        """Enable per-drone LoRA online adaptation on the mixed model.
        Use predict_with_adaptation() afterwards for inference with per-drone fine-tuning."""
        from adapter_manager import DroneAdapterManager
        from online_learner import OnlineLearner, OnlineLearnerConfig

        self._adapter_mgr = DroneAdapterManager(
            self.mixed, checkpoint_dir=checkpoint_dir, r=lora_r,
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
        """Predict with optional per-drone LoRA adaptation.

        Args:
            hist: (B, 20, 6) past trajectory in real meters, m/s.
            drone_id: Unique drone identifier. None = no adaptation (same as predict()).
            ground_truth: (B, 20, 3) future displacement. If provided, accumulated for online learning.
            intent_label: Intent label for this observation (optional).
            timestep: Observation timestep (optional).

        Returns:
            Same as predict(), plus adapted (bool), updated (bool).
        """
        # Input validation
        if hist.dim() != 3 or hist.shape[1] != 20 or hist.shape[2] != 6:
            raise ValueError(
                f"Expected shape (B, 20, 6), got {hist.shape}."
            )
        if torch.isnan(hist).any():
            warnings.warn("Input contains NaN values, replacing with zeros.")
            hist = torch.nan_to_num(hist, nan=0.0)

        hist = hist.to(self.device)

        # Inference under no_grad (base models + speed + weights)
        with torch.no_grad():
            speed = self.compute_speed(hist)
            w_low, w_mixed, w_high = self.blend_weights(speed)

            out_low = self.low(hist, force_predict=True)
            out_high = self.high(hist, force_predict=True)

            # Mixed model: with or without LoRA
            adapted = False
            if drone_id and hasattr(self, '_adaptation_enabled') and self._adaptation_enabled:
                if self._adapter_mgr.has_adapter(drone_id):
                    self._adapter_mgr.activate(drone_id)
                    adapted = True

            out_mixed = self.mixed(hist, force_predict=True)

            if adapted:
                self._adapter_mgr.deactivate()

            # Blend predictions (still under no_grad)
            w3 = lambda w: w.view(-1, 1, 1)
            predictions = (w3(w_low) * out_low['predictions'] +
                           w3(w_mixed) * out_mixed['predictions'] +
                           w3(w_high) * out_high['predictions'])

            w2 = lambda w: w.view(-1, 1)
            intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                       device=self.device, dtype=out_high['intent_logits'].dtype)
            for c4, c6 in _C4_TO_C6.items():
                intent_high_6[:, c6] = out_high['intent_logits'][:, c4]
            intent = (w2(w_low) * out_low['intent_logits'] +
                      w2(w_mixed) * out_mixed['intent_logits'] +
                      w2(w_high) * intent_high_6)

        result = {
            'predictions': predictions,
            'intent_logits': intent,
            'speed': speed,
            'adapted': adapted,
            'updated': False,
        }

        # Online learning hook
        if (drone_id and ground_truth is not None and
                hasattr(self, '_adaptation_enabled') and self._adaptation_enabled):

            # Compute confidence (simple entropy-based)
            probs = torch.softmax(intent, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            max_entropy = torch.log(torch.tensor(6.0)).item()  # 6 classes
            confidence = max(0.0, min(1.0, 1.0 - entropy / max_entropy))

            # Observe and potentially update
            updated = self._learner.observe(
                drone_id, hist[0].cpu(), ground_truth[0].cpu(),
                confidence=confidence, intent=intent_label, timestep=timestep,
            )
            result['updated'] = updated
            result['confidence'] = confidence

        return result

    def get_drone_status(self, drone_id: str) -> Optional[dict]:
        """Get online learning status for a drone."""
        if hasattr(self, '_learner'):
            return self._learner.get_status(drone_id)
        return None

    def save_adapters(self):
        """Persist all drone adapters to disk."""
        if hasattr(self, '_learner'):
            self._learner.save_all()

    @property
    def adaptation_enabled(self) -> bool:
        return getattr(self, '_adaptation_enabled', False)


if __name__ == '__main__':
    import shutil

    print('DronePredictor — Smoke Test')
    print(f'Device: {_DEVICE}\n')

    p = DronePredictor()

    # 低速场景 (~1 m/s)
    x_low = torch.randn(4, 20, 6)
    x_low[:, :, 3:6] *= 0.5
    out = p.predict(x_low)
    print(f"Low speed (~0.5 m/s): speed={[f'{s:.1f}' for s in out['speed']]}")
    print(f"  pred shape: {out['predictions'].shape}")

    # 高速场景 (~15 m/s)
    x_high = torch.randn(4, 20, 6)
    x_high[:, :, 3:6] *= 12.0
    out = p.predict(x_high)
    print(f"High speed (~15 m/s): speed={[f'{s:.1f}' for s in out['speed']]}")
    print(f"  intent shape: {out['intent_logits'].shape}")

    print(f"\n{p.stats}")

    # ---- LoRA adaptation test ----
    print('\n--- Online Adaptation Test ---')
    p.enable_adaptation(checkpoint_dir='test_adapters_pred', accumulation_steps=3)

    for i in range(10):
        hist = torch.randn(1, 20, 6)
        hist[:, :, 3:6] *= 1.5  # ~1.5 m/s (low-speed domain)
        gt = torch.randn(1, 20, 3) * 0.1  # small displacements

        out = p.predict_with_adaptation(hist, drone_id='test_drone',
                                         ground_truth=gt, timestep=i)
        if out['updated']:
            status = p.get_drone_status('test_drone')
            print(f"  Step {i}: UPDATE (updates={status['num_updates']}, "
                  f"replay={status['replay_size']})")

    status = p.get_drone_status('test_drone')
    print(f"\nFinal: {status}")
    print(f"Adaptation enabled: {p.adaptation_enabled}")

    p.save_adapters()
    shutil.rmtree('test_adapters_pred', ignore_errors=True)
    print('\nAll good!')
