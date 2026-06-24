"""
评估指标
完全按PPT定义实现:
1. 距离误差 (RMSE/MSE)
2. 距离准率: 0.25 + pred_step * 0.15, 最大1.5m
3. 方向准率: 夹角 < 15° 的样本占比
4. 动力学平滑度: Mean Jerk / Mean Accel / Max Accel
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_displacement_error(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    计算位移误差
    Args:
        predictions: (B, T, 3) 预测位移
        targets: (B, T, 3) 真值位移
    Returns:
        dict: mse, rmse, mae
    """
    diff = predictions - targets  # (B,T,3)
    mse = (diff ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = diff.abs().mean()
    return {'mse': mse, 'rmse': rmse, 'mae': mae}


def compute_distance_accuracy(
    predictions: torch.Tensor,  # (B, T, 3) 预测位移
    targets: torch.Tensor,     # (B, T, 3) 真值位移
    max_distance: float = 1.5,
    step_multiplier: float = 0.15,
    step_offset: float = 0.25
) -> Dict[str, torch.Tensor]:
    """
    距离准率 (PPT定义)
    准率阈值随步数增长: threshold = min(0.25 + step * 0.15, 1.5)
    统计 |pred - target| < threshold 的样本占比
    Args:
        predictions: (B, T, 3)
        targets: (B, T, 3)
        max_distance: 最大阈值 1.5m
        step_multiplier: 步长乘数 0.15
        step_offset: 基准偏移 0.25
    Returns:
        per_step_accuracy: (T,) 每步准率
        overall_accuracy: scalar
    """
    B, T, _ = predictions.shape
    device = predictions.device

    # 每步的误差
    errors = torch.norm(predictions - targets, dim=-1)  # (B,T)

    # 计算每步阈值
    steps = torch.arange(1, T + 1, device=device).float()  # (T,)
    thresholds = (step_offset + steps * step_multiplier).clamp(max=max_distance)  # (T,)
    thresholds = thresholds.unsqueeze(0)  # (1,T)

    # 每步准率 = 误差 < 阈值的样本占比
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
    方向准率 (PPT定义)
    夹角 < 15° 的样本占比
    """
    B, T, _ = predictions.shape
    device = predictions.device

    # 向量归一化
    pred_norm = F.normalize(predictions, dim=-1)  # (B,T,3)
    tgt_norm = F.normalize(targets, dim=-1)       # (B,T,3)

    # 余弦相似度
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1).clamp(-1, 1)  # (B,T)
    angles = cos_sim.acos() * 180.0 / 3.14159  # 转换为度

    angle_ok = angles < angle_threshold  # (B,T)
    per_step_accuracy = angle_ok.float().mean(dim=0)  # (T,)
    overall_accuracy = angle_ok.float().mean()

    return {
        'per_step_accuracy': per_step_accuracy,
        'overall_accuracy': overall_accuracy,
        'angles': angles
    }


def compute_kinematic_smoothness(
    positions: torch.Tensor,  # (B, T, 3) 绝对位置轨迹
    dt: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    动力学平滑度评估
    在预测轨迹上计算 (需要先重建绝对位置)
    Args:
        positions: (B, T, 3) 绝对位置
        dt: 时间步长 (秒)
    Returns:
        mean_jerk: 平均抖动 (三阶导)
        mean_accel: 平均加速度 (二阶导)
        max_accel: 最大加速度
    """
    if positions.shape[1] < 4:
        return {'mean_jerk': torch.tensor(0.0), 'mean_accel': torch.tensor(0.0), 'max_accel': torch.tensor(0.0)}

    device = positions.device

    # 数值微分
    vel = torch.diff(positions, dim=1) / dt               # (B, T-1, 3)
    accel = torch.diff(vel, dim=1) / dt                   # (B, T-2, 3)
    jerk = torch.diff(accel, dim=1) / dt                   # (B, T-3, 3)

    # 标量
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
    predictions: torch.Tensor,   # (B, pred_len, 3) 预测位移
    targets: torch.Tensor,        # (B, pred_len, 3) 真值位移
    last_position: torch.Tensor = None,  # (B, 3) 历史最后位置
) -> Dict[str, float]:
    """
    完整评估报告
    """
    if last_position is None:
        last_position = torch.zeros(predictions.shape[0], 3, device=predictions.device)

    # 重建绝对位置
    pred_abs = torch.cumsum(predictions, dim=1) + last_position.unsqueeze(1)
    tgt_abs = torch.cumsum(targets, dim=1) + last_position.unsqueeze(1)

    # 距离误差
    err = compute_displacement_error(predictions, targets)
    dist_acc = compute_distance_accuracy(predictions, targets)
    dir_acc = compute_direction_accuracy(predictions, targets)
    kin = compute_kinematic_smoothness(pred_abs)

    # 分机动强度报告
    # 简化: 用预测误差的方差估算机动强度
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

    # 分步详细报告
    for step in [0, 4, 9, 14, 19]:  # 1s, 5s, 10s, 15s, 20s (假设0.1s/步)
        if step < predictions.shape[1]:
            results[f'Step_{step+1}_DistAcc'] = dist_acc['per_step_accuracy'][step].item()
            results[f'Step_{step+1}_DirAcc'] = dir_acc['per_step_accuracy'][step].item()

    return results


def maneuver_classification(trajectory: torch.Tensor, dt: float = 0.1
                             ) -> Tuple[torch.Tensor, str]:
    """
    基于机动强度划分场景类别
    按25%/75%分位数划分为平稳/普通/急剧

    Args:
        trajectory: (T, 6) [x,y,z,vx,vy,vz]
    Returns:
        level: 0=平稳, 1=普通, 2=急剧
        label: str
    """
    if trajectory.shape[0] < 5:
        return 0, 'unknown'

    positions = trajectory[:, :3]
    velocities = torch.diff(positions, dim=0) / dt
    accelerations = torch.diff(velocities, dim=0) / dt

    accel_mag = accelerations.norm(dim=-1).mean().item()
    vel_std = velocities.norm(dim=-1).std().item()

    # 综合得分 (大致范围)
    score = accel_mag * 0.6 + vel_std * 0.4

    # 分位阈值 (可调整)
    thresholds = {'smooth': 2.0, 'normal': 5.0}
    if score < thresholds['smooth']:
        return 0, 'smooth'
    elif score < thresholds['normal']:
        return 1, 'normal'
    else:
        return 2, 'high_maneuver'
