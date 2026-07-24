"""Evaluation metrics: displacement error, distance accuracy, direction accuracy, kinematic smoothness."""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_displacement_error(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute displacement error.
    Args:
        predictions: (B, T, 3) predicted displacement
        targets: (B, T, 3) ground-truth displacement
    Returns:
        dict: mse, rmse, mae
    """
    diff = predictions - targets  # (B,T,3)
    mse = (diff ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = diff.abs().mean()
    return {'mse': mse, 'rmse': rmse, 'mae': mae}


def compute_distance_accuracy(
    predictions: torch.Tensor,  # (B, T, 3) predicted displacement
    targets: torch.Tensor,     # (B, T, 3) ground-truth displacement
    max_distance: float = 1.5,
    step_multiplier: float = 0.15,
    step_offset: float = 0.25
) -> Dict[str, torch.Tensor]:
    """
    Distance accuracy (PPT definition).
    Per-step threshold: threshold = min(0.25 + step * 0.15, 1.5)
    Fraction of samples with |pred - target| < threshold.
    Args:
        predictions: (B, T, 3)
        targets: (B, T, 3)
        max_distance: max threshold 1.5m
        step_multiplier: step multiplier 0.15
        step_offset: base offset 0.25
    Returns:
        per_step_accuracy: (T,) accuracy per step
        overall_accuracy: scalar
    """
    B, T, _ = predictions.shape
    device = predictions.device

    # Per-step error
    errors = torch.norm(predictions - targets, dim=-1)  # (B,T)

    # Per-step threshold
    steps = torch.arange(1, T + 1, device=device).float()  # (T,)
    thresholds = (step_offset + steps * step_multiplier).clamp(max=max_distance)  # (T,)
    thresholds = thresholds.unsqueeze(0)  # (1,T)

    # Per-step accuracy = fraction with error < threshold
    per_step_ok = (errors < thresholds).float()  # (B,T)
    per_step_accuracy = per_step_ok.mean(dim=0)  # (T,)
    overall_accuracy = per_step_ok.mean()

    return {
        'per_step_accuracy': per_step_accuracy,
        'overall_accuracy': overall_accuracy,
        'thresholds': thresholds.squeeze(0)
    }


def compute_direction_accuracy(
    predictions: torch.Tensor,  # (B, T, 3)
    targets: torch.Tensor,      # (B, T, 3)
    angle_threshold: float = 15.0
) -> Dict[str, torch.Tensor]:
    """
    Direction accuracy (PPT definition).
    Fraction of samples with angle < 15 degrees.
    """
    B, T, _ = predictions.shape
    device = predictions.device

    # Normalize vectors
    pred_norm = F.normalize(predictions, dim=-1)  # (B,T,3)
    tgt_norm = F.normalize(targets, dim=-1)       # (B,T,3)

    # Cosine similarity
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1).clamp(-1, 1)  # (B,T)
    angles = cos_sim.acos() * 180.0 / 3.14159  # to degrees

    angle_ok = angles < angle_threshold  # (B,T)
    per_step_accuracy = angle_ok.float().mean(dim=0)  # (T,)
    overall_accuracy = angle_ok.float().mean()

    return {
        'per_step_accuracy': per_step_accuracy,
        'overall_accuracy': overall_accuracy,
        'angles': angles
    }


def compute_kinematic_smoothness(
    positions: torch.Tensor,  # (B, T, 3) absolute position trajectory
    dt: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Kinematic smoothness (computed on predicted trajectory; reconstruct absolute positions first).
    Args:
        positions: (B, T, 3) absolute positions
        dt: time step (seconds)
    Returns:
        mean_jerk: mean jerk (3rd derivative)
        mean_accel: mean acceleration (2nd derivative)
        max_accel: max acceleration
    """
    if positions.shape[1] < 4:
        return {'mean_jerk': torch.tensor(0.0), 'mean_accel': torch.tensor(0.0), 'max_accel': torch.tensor(0.0)}

    device = positions.device

    # Numerical differentiation
    vel = torch.diff(positions, dim=1) / dt               # (B, T-1, 3)
    accel = torch.diff(vel, dim=1) / dt                   # (B, T-2, 3)
    jerk = torch.diff(accel, dim=1) / dt                   # (B, T-3, 3)

    # Magnitudes
    accel_mag = accel.norm(dim=-1)   # (B, T-2)
    jerk_mag = jerk.norm(dim=-1)     # (B, T-3)

    mean_accel = accel_mag.mean()
    mean_jerk = jerk_mag.mean()
    max_accel = accel_mag.max()

    return {
        'mean_jerk': mean_jerk,
        'mean_accel': mean_accel,
        'max_accel': max_accel
    }


def full_evaluation(
    predictions: torch.Tensor,   # (B, pred_len, 3) predicted displacement
    targets: torch.Tensor,        # (B, pred_len, 3) ground-truth displacement
    last_position: torch.Tensor = None,  # (B, 3) last historical position
) -> Dict[str, float]:
    """Full evaluation report."""
    if last_position is None:
        last_position = torch.zeros(predictions.shape[0], 3, device=predictions.device)

    # Reconstruct absolute positions
    pred_abs = torch.cumsum(predictions, dim=1) + last_position.unsqueeze(1)
    tgt_abs = torch.cumsum(targets, dim=1) + last_position.unsqueeze(1)

    # Distance error
    err = compute_displacement_error(predictions, targets)
    dist_acc = compute_distance_accuracy(predictions, targets)
    dir_acc = compute_direction_accuracy(predictions, targets)
    kin = compute_kinematic_smoothness(pred_abs)

    # Maneuver-intensity split; approximate intensity via variance of prediction error
    step_errors = torch.norm(predictions - targets, dim=-1)  # (B, pred_len)
    high_maneuver_mask = step_errors.std(dim=-1) > step_errors.std() * 0.5

    results = {
        'RMSE': err['rmse'].item(),
        'MAE': err['mae'].item(),
        'Distance_Accuracy': dist_acc['overall_accuracy'].item(),
        'Direction_Accuracy': dir_acc['overall_accuracy'].item(),
        'Mean_Jerk': kin['mean_jerk'].item(),
        'Mean_Accel': kin['mean_accel'].item(),
        'Max_Accel': kin['max_accel'].item(),
    }

    # Per-step detailed report
    for step in [0, 4, 9, 14, 19]:  # 1s, 5s, 10s, 15s, 20s (assuming 0.1s/step)
        if step < predictions.shape[1]:
            results[f'Step_{step+1}_DistAcc'] = dist_acc['per_step_accuracy'][step].item()
            results[f'Step_{step+1}_DirAcc'] = dir_acc['per_step_accuracy'][step].item()

    return results


def maneuver_classification(trajectory: torch.Tensor, dt: float = 0.1
                             ) -> Tuple[torch.Tensor, str]:
    """
    Classify scene by maneuver intensity.
    Split into smooth/normal/high by 25%/75% quantiles.

    Args:
        trajectory: (T, 6) [x,y,z,vx,vy,vz]
    Returns:
        level: 0=smooth, 1=normal, 2=high
        label: str
    """
    if trajectory.shape[0] < 5:
        return 0, 'unknown'

    positions = trajectory[:, :3]
    velocities = torch.diff(positions, dim=0) / dt
    accelerations = torch.diff(velocities, dim=0) / dt

    accel_mag = accelerations.norm(dim=-1).mean().item()
    vel_std = velocities.norm(dim=-1).std().item()

    # Composite score (approximate range)
    score = accel_mag * 0.6 + vel_std * 0.4

    # Quantile thresholds (tunable)
    thresholds = {'smooth': 2.0, 'normal': 5.0}
    if score < thresholds['smooth']:
        return 0, 'smooth'
    elif score < thresholds['normal']:
        return 1, 'normal'
    else:
        return 2, 'high_maneuver'
