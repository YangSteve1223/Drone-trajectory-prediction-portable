"""
Per-window dynamic normalization for trajectory data. Adapts scale to local velocity statistics.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NormConfig:
    """Normalization configuration for a specific dataset or deployment."""

    # Scale computation method
    method: str = "velocity"

    # Fixed scales (used when method="fixed" or as fallback)
    fixed_pos_scale: float = 100.0
    fixed_vel_scale: float = 10.0

    # Safety bounds for dynamic scale
    min_pos_scale: float = 1.0
    max_pos_scale: float = 1000.0
    min_vel_scale: float = 0.1
    max_vel_scale: float = 100.0

    # Time parameters
    dt: float = 0.2
    pred_len: int = 20

    # Smoothing: EMA factor for scale (0=no smoothing, 0.9=heavy smoothing)
    scale_smoothing: float = 0.7

    # Zero-centering
    center_on_first: bool = True
    center_on_mean: bool = False


class DynamicNormalizer:
    """
    Per-window dynamic normalization.
    Normalization: pos_norm = (pos - origin) / scale_pos, vel_norm = vel / scale_vel.
    """

    def __init__(self, config: NormConfig = None):
        self.config = config or NormConfig()
        self._last_scale_pos: float = self.config.fixed_pos_scale
        self._last_origin: Optional[torch.Tensor] = None

    def compute_params(self, history: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Compute origin, scale_pos, scale_vel from a history window (B, 20, 6)."""
        cfg = self.config
        B = history.shape[0]
        device = history.device

        pos = history[:, :, :3]  # (B, 20, 3)
        vel = history[:, :, 3:6]  # (B, 20, 3)

        # Origin
        if cfg.center_on_first:
            origin = pos[:, 0, :]
        elif cfg.center_on_mean:
            origin = pos.mean(dim=1)
        else:
            origin = torch.zeros(B, 3, device=device)

        # Scale
        if cfg.method == "velocity":
            recent_vel = vel[:, -10:, :]
            mean_speed = torch.norm(recent_vel, dim=2).mean(dim=1)  # (B,)
            scale_pos = mean_speed * cfg.pred_len * cfg.dt

        elif cfg.method == "displacement":
            disp = pos[:, -1, :] - pos[:, 0, :]
            scale_pos = torch.norm(disp, dim=1)

        elif cfg.method == "fixed":
            scale_pos = torch.full((B,), cfg.fixed_pos_scale, device=device)

        else:
            raise ValueError(f"Unknown method: {cfg.method}")

        # Clamp
        scale_pos = torch.clamp(scale_pos,
                               min=cfg.min_pos_scale,
                               max=cfg.max_pos_scale)

        # Smooth with EMA (per-batch average for stability)
        if isinstance(scale_pos, torch.Tensor):
            avg_scale = scale_pos.mean().item()
        else:
            avg_scale = float(scale_pos)
        smoothed = cfg.scale_smoothing * self._last_scale_pos + (1 - cfg.scale_smoothing) * avg_scale
        self._last_scale_pos = smoothed
        scale_pos = torch.full((B,), smoothed, device=device)

        # Velocity scale derived from smoothed position scale
        scale_vel = scale_pos / cfg.pred_len
        scale_vel = torch.clamp(scale_vel,
                               min=cfg.min_vel_scale,
                               max=cfg.max_vel_scale)

        return origin, scale_pos, scale_vel

    def normalize(self, history: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Normalize history window. Returns normalized tensor and params dict."""
        origin, s_pos, s_vel = self.compute_params(history)
        B = history.shape[0]
        device = history.device

        if isinstance(s_pos, (int, float)):
            s_pos = torch.full((B,), s_pos, device=device)
        if isinstance(s_vel, (int, float)):
            s_vel = torch.full((B,), s_vel, device=device)
        if s_pos.dim() == 0:
            s_pos = s_pos.unsqueeze(0).expand(B)
        if s_vel.dim() == 0:
            s_vel = s_vel.unsqueeze(0).expand(B)

        s_pos_3d = s_pos.view(B, 1, 1)
        s_vel_3d = s_vel.view(B, 1, 1)
        origin_3d = origin.view(B, 1, 3)

        hist_norm = history.clone()
        # pos_norm = (pos - origin) / scale_pos
        hist_norm[:, :, :3] = (history[:, :, :3] - origin_3d) / s_pos_3d
        # vel_norm = vel / scale_vel
        hist_norm[:, :, 3:6] = history[:, :, 3:6] / s_vel_3d

        params = {
            'origin': origin,
            'scale_pos': s_pos,
            'scale_vel': s_vel,
        }

        return hist_norm, params

    def denormalize(self, predictions: torch.Tensor, params: dict) -> torch.Tensor:
        """Convert normalized displacement predictions back to real meters."""
        return predictions * self._get_scale_tensor(params['scale_pos'], predictions)

    def _get_scale_tensor(self, scale, ref: torch.Tensor):
        """Ensure scale is a tensor broadcastable to (B, 1, 1)."""
        B = ref.shape[0]
        if isinstance(scale, (int, float)):
            scale = torch.full((B,), scale, device=ref.device, dtype=ref.dtype)
        elif scale.device != ref.device:
            scale = scale.to(ref.device)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0).expand(B)
        return scale.view(B, 1, 1)

    def denormalize_absolute(self, predictions: torch.Tensor, params: dict) -> torch.Tensor:
        """Convert normalized predictions to absolute real-world positions."""
        B = predictions.shape[0]
        origin = params['origin'].to(predictions.device)
        if origin.dim() == 2:
            origin = origin.view(B, 1, 3)

        real_disp = predictions * self._get_scale_tensor(params['scale_pos'], predictions)
        absolute = torch.cumsum(real_disp, dim=1)
        absolute = absolute + origin
        return absolute

    def normalize_target(self, future_gt: torch.Tensor, params: dict) -> torch.Tensor:
        """Normalize ground truth for loss computation."""
        return future_gt / self._get_scale_tensor(params['scale_pos'], future_gt)

    @property
    def last_scale(self) -> float:
        return self._last_scale_pos

    @property
    def last_origin(self) -> Optional[torch.Tensor]:
        return self._last_origin


def scale_invariant_mse(pred: torch.Tensor, target: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant MSE loss: SI-MSE = MSE(pred/s, target/s) where s = mean(|target|).
    Independent of absolute trajectory scale; focuses on relative motion pattern.
    """
    # Per-sample scale: mean absolute displacement
    scale = target.abs().mean(dim=[1, 2], keepdim=True) + eps  # (B, 1, 1)

    pred_scaled = pred / scale
    target_scaled = target / scale

    return nn.functional.mse_loss(pred_scaled, target_scaled)


def scale_invariant_l1(pred: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant L1 loss."""
    scale = target.abs().mean(dim=[1, 2], keepdim=True) + eps
    return nn.functional.l1_loss(pred / scale, target / scale)


if __name__ == '__main__':
    print('=== Dynamic Normalizer Smoke Test ===\n')

    # Low-speed scenario (UAV-Flow, ~1 m/s)
    print('--- Low-speed scenario (UAV-Flow, ~1 m/s) ---')
    cfg = NormConfig(method="velocity", center_on_first=True)
    norm = DynamicNormalizer(cfg)

    hist_low = torch.randn(2, 20, 6)
    hist_low[:, :, :3] = torch.cumsum(torch.randn(2, 20, 3) * 0.2, dim=1)
    hist_low[:, :, 3:6] = torch.randn(2, 20, 3) * 0.3 + 1.0

    norm_hist, params = norm.normalize(hist_low)
    sp = params['scale_pos']; sv = params['scale_vel']
    sp_val = sp[0].item() if isinstance(sp, torch.Tensor) and sp.numel() > 0 else float(sp)
    sv_val = sv[0].item() if isinstance(sv, torch.Tensor) and sv.numel() > 0 else float(sv)
    print(f'  scale_pos={sp_val:.1f}, scale_vel={sv_val:.2f}')
    print(f'  norm_pos range: [{norm_hist[:,:,:3].min():.2f}, {norm_hist[:,:,:3].max():.2f}]')
    print(f'  norm_vel range: [{norm_hist[:,:,3:].min():.2f}, {norm_hist[:,:,3:].max():.2f}]')

    pred_norm = torch.randn(2, 20, 3) * 0.1
    pred_real = norm.denormalize(pred_norm, params)
    print(f'  pred_norm range: [{pred_norm.min():.2f}, {pred_norm.max():.2f}]')
    print(f'  pred_real range: [{pred_real.min():.2f}, {pred_real.max():.2f}]')

    # High-speed scenario (NPZDATA, ~20 m/s)
    print('\n--- High-speed scenario (NPZDATA, ~20 m/s) ---')
    hist_high = torch.randn(2, 20, 6)
    hist_high[:, :, :3] = torch.cumsum(torch.randn(2, 20, 3) * 4.0, dim=1)
    hist_high[:, :, 3:6] = torch.randn(2, 20, 3) * 5.0 + 20.0

    norm_hist2, params2 = norm.normalize(hist_high)
    sp2 = params2['scale_pos']
    sp2_val = sp2[0].item() if isinstance(sp2, torch.Tensor) and sp2.numel() > 0 else float(sp2)
    print(f'  scale_pos={sp2_val:.0f}')
    print(f'  norm_pos range: [{norm_hist2[:,:,:3].min():.2f}, {norm_hist2[:,:,:3].max():.2f}]')

    # Hovering edge case (near-zero velocity)
    print('\n--- Hovering scenario (~0.01 m/s) ---')
    hist_hover = torch.randn(2, 20, 6) * 0.001
    hist_hover[:, :, 3:6] *= 0.01
    _, params3 = norm.normalize(hist_hover)
    sp3 = params3['scale_pos']
    sp3_val = sp3[0].item() if isinstance(sp3, torch.Tensor) and sp3.numel() > 0 else float(sp3)
    print(f'  scale_pos={sp3_val:.2f} (should be >= min_pos_scale={cfg.min_pos_scale})')

    # Scale-Invariant Loss
    print('\n--- Scale-Invariant Loss ---')
    pred = torch.randn(2, 20, 3) * 10
    target = torch.randn(2, 20, 3) * 10
    si_loss = scale_invariant_mse(pred, target)
    mse_loss = nn.functional.mse_loss(pred, target)
    print(f'  SI-MSE: {si_loss:.4f} (scale-independent)')
    print(f'  Raw MSE: {mse_loss:.4f} (depends on absolute scale)')

    # SI-MSE with same pattern at different scales should be similar
    pred_a = torch.randn(1, 20, 3)
    target_a = torch.randn(1, 20, 3)
    pred_b = pred_a * 50
    target_b = target_a * 50
    si_a = scale_invariant_mse(pred_a, target_a)
    si_b = scale_invariant_mse(pred_b, target_b)
    print(f'  SI-MSE same pattern @ 1x: {si_a:.4f}')
    print(f'  SI-MSE same pattern @ 50x: {si_b:.4f} (should be approx same)')

    print('\nAll tests passed!')
