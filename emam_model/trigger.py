"""Event-Driven Trigger for Adaptive Trajectory Prediction.

Supports two modes:
  Mode 1 (EventDrivenTrigger): Simple 3-factor learned scoring (原有)
  Mode 2 (FunnelTrigger): PPT Slide 5 漏斗式多级过滤 (新增)

PPT Funnel Pipeline:
  目标状态序列 → CTE < R_protect? → 航线交叉轨迹误差CTE
  → 朝向意图角/纵向稳定性/横向稳定性 → Buffer Size >= W?
  → D_t < D_warn OR TTA_t < T_warn? → 正对禁飞区稳定飞行 → 开启预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# =====================================================================
# Mode 0: Always-on trigger (fallback)
# =====================================================================

class SimpleTrigger(nn.Module):
    """始终触发的简单触发器 (fallback)"""
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
# Mode 1: Simple learned 3-factor trigger (原实现, 保留兼容)
# =====================================================================

class EventDrivenTrigger(nn.Module):
    """
    三因素加权触发器 (原实现, 作为 Mode 1 保留).

    trigger_score = 0.3*threat + 0.3*intent + 0.4*spatial
    """
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
# Mode 2: Full Funnel Trigger (PPT Slide 5 完整实现)
# =====================================================================

class FunnelTrigger(nn.Module):
    """
    PPT Slide 5 漏斗式多级过滤触发器.

    六阶段流水线:
      Stage 1 — 时空边界过滤: 目标是否进入保护空域 (CTE vs R_protect)
      Stage 2 — 航线交叉轨迹误差 (CTE) 计算
      Stage 3 — 运动意图提取: 朝向意图角 / 纵向稳定性 / 横向稳定性
      Stage 4 — 缓冲区管理: Buffer Size >= W? (滑动窗口累积)
      Stage 5 — 威胁时效判决: D_t < D_warn OR TTA_t < T_warn?
      Stage 6 — 飞行姿态确认: 正对禁飞区 + 稳定飞行 → 开启预测

    Args:
        feature_dim: 输入特征维度
        num_intent_classes: 意图类别数
        buffer_size: 滑动窗口大小 W (帧数)
        r_protect: 保护空域半径 (m)
        d_warn: 预警距离阈值 (m)
        t_warn: 预警时间阈值 (s)
        cte_threshold: 航线交叉误差阈值 (m)
        heading_tolerance: 朝向对准容差 (度)
        stability_threshold: 稳定性阈值 (速度方差)
    """
    def __init__(
        self,
        feature_dim: int = 6,
        num_intent_classes: int = 5,
        # Funnel thresholds
        buffer_size: int = 10,          # W: 最小观察窗口
        r_protect: float = 500.0,       # R_protect: 保护空域半径 (m)
        d_warn: float = 200.0,          # D_warn: 距离预警阈值 (m)
        t_warn: float = 30.0,           # T_warn: 时间预警阈值 (s)
        cte_threshold: float = 50.0,    # CTE 阈值 (m)
        heading_tolerance: float = 30.0, # 朝向容差 (度)
        stability_threshold: float = 2.0,# 速度方差稳定性阈值
        dt: float = 0.1,                # 时间步长 (s)
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

        # 可学习的评分器 (用于微调阈值判定)
        self.stability_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.heading_encoder = nn.Sequential(
            nn.Linear(3 + num_intent_classes, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.threat_encoder = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # 滑动窗口缓冲区 (per-sample buffer state, 不注册为 buffer 以避免梯度问题)
        self.register_buffer('_buf_ptr', torch.zeros(1, dtype=torch.long))
        self._buffer_store = {}  # {batch_idx: trajectory_buffer}

    def _compute_cte(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Stage 1-2: 计算航线交叉轨迹误差 (CTE).

        CTE = 当前实际位置到参考航线的垂直距离.
        参考航线 = 历史轨迹的首尾连线 (简化).

        Returns:
            cte: (B,) 交叉轨迹误差 (m)
            distance_to_protected: (B,) 距保护空域距离 (m)
        """
        B, T, C = trajectory.shape
        pos = trajectory[:, :, :3]   # (B, T, 3)

        # 参考航线: 历史轨迹 start → end 的直线
        start_pos = pos[:, 0, :]      # (B, 3)
        end_pos = pos[:, -1, :]       # (B, 3)
        route_vec = end_pos - start_pos  # (B, 3)
        route_len = torch.norm(route_vec, dim=-1).clamp(min=1e-6)  # (B,)

        # 最后时刻位置到航线的投影
        last_pos = pos[:, -1, :]      # (B, 3)
        to_end = last_pos - start_pos  # (B, 3)

        # 投影参数 t = dot(to_end, route_vec) / |route_vec|^2
        t = torch.sum(to_end * route_vec, dim=-1) / (route_len ** 2)  # (B,)
        t = t.clamp(0.0, 1.0)  # 限制在线段内

        # 投影点
        proj = start_pos + t.unsqueeze(-1) * route_vec  # (B, 3)
        cte = torch.norm(last_pos - proj, dim=-1)  # (B,) 交叉轨迹误差

        # 距保护空域距离 (简化: 航线终点作为保护区中心)
        distance_to_protected = route_len * (1.0 - t)  # (B,) 沿航线剩余距离

        return cte, distance_to_protected

    def _compute_heading_intent(
        self, trajectory: torch.Tensor, intent_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: 计算朝向意图角 + 纵向/横向稳定性.

        Returns:
            heading_to_zone: (B,) 当前朝向与保护区方向的夹角 (度)
            heading_aligned: (B,) 是否对准保护区 (bool)
        """
        B, T, C = trajectory.shape
        pos = trajectory[:, :, :3]
        vel = trajectory[:, :, 3:6]

        # 当前速度方向 (取最后3帧平均, 降噪)
        recent_vel = vel[:, -3:, :].mean(dim=1)  # (B, 3)
        vel_norm = recent_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel_dir = recent_vel / vel_norm  # (B, 3)  避免 F.normalize 零向量 → NaN

        # 保护区方向 (航线前进方向)
        route_vec = pos[:, -1, :3] - pos[:, 0, :3]
        route_norm = route_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        route_dir = route_vec / route_norm  # (B, 3)

        # 夹角
        cos_angle = (vel_dir * route_dir).sum(dim=-1).clamp(-1, 1)  # (B,)
        heading_to_zone = torch.acos(cos_angle) * 180.0 / math.pi  # (B,) 度
        heading_aligned = heading_to_zone < self.heading_tolerance

        return heading_to_zone, heading_aligned

    def _compute_stability(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: 计算纵向/横向稳定性.

        纵向稳定性 = 速度大小的变异系数 (低 = 稳定)
        横向稳定性 = 侧向加速度的方差 (低 = 稳定)

        Returns:
            long_stability: (B,) 纵向稳定性得分 [0,1] (高=稳定)
            lat_stability: (B,) 横向稳定性得分 [0,1] (高=稳定)
        """
        B, T, C = trajectory.shape
        vel = trajectory[:, :, 3:6]  # (B, T, 3)

        # 速度大小
        speed = torch.norm(vel, dim=-1)  # (B, T)
        speed_mean = speed.mean(dim=-1)  # (B,)
        speed_std = speed.std(dim=-1)    # (B,)

        # 纵向稳定性: 速度变异系数
        cv = speed_std / (speed_mean + 1e-6)  # 变异系数
        long_stability = torch.exp(-cv * 5.0)  # [0,1], 低CV=高稳定性

        # 横向稳定性: 垂直于速度方向的加速度
        speed_3d = vel  # (B, T, 3)
        # 方向向量 (避免 F.normalize 零向量 → NaN)
        vel_curr = speed_3d[:, 1:, :]   # (B, T-1, 3)
        vel_prev = speed_3d[:, :-1, :]  # (B, T-1, 3)
        vel_dir = vel_curr / vel_curr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel_dir_prev = vel_prev / vel_prev.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # 方向变化率 (曲率近似)
        lat_acc = torch.norm(vel_dir - vel_dir_prev, dim=-1)  # (B, T-1)
        lat_std = lat_acc.std(dim=-1)  # (B,)
        lat_stability = torch.exp(-lat_std * 10.0)  # [0,1]

        return long_stability, lat_stability

    def _compute_tta(self, distance: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """
        Stage 5: 计算到达保护区的预估时间 (TTA).

        Args:
            distance: (B,) 距保护区的距离 (m)
            speed: (B,) 当前速度 (m/s)
        Returns:
            tta: (B,) 预计到达时间 (s)
        """
        tta = distance / (speed + 1e-6)
        return tta

    def forward(
        self,
        trajectory: torch.Tensor,
        intent_logits: torch.Tensor,
        intent_history: Optional[torch.Tensor] = None,
        protected_zone_pos: Optional[torch.Tensor] = None,  # (B, 3) 保护区中心 (可选)
    ) -> Dict[str, torch.Tensor]:
        """
        漏斗式多级过滤前向传播.

        Args:
            trajectory: (B, T, 6) [x,y,z,vx,vy,vz]
            intent_logits: (B, num_classes)
            intent_history: (T, num_classes) 历史意图
            protected_zone_pos: (B, 3) 禁飞区中心坐标 (可选, 无则用轨迹终点估算)

        Returns:
            trigger_decision: (B,) bool
            trigger_score: (B,) float [0,1]
            stage_results: dict 各阶段详细结果
        """
        B, T, C = trajectory.shape
        device = trajectory.device
        intent_probs = F.softmax(intent_logits, dim=-1)

        stage_results = {}

        # ============================================================
        # Stage 1: 时空边界过滤 — CTE vs R_protect
        # ============================================================
        cte, dist_to_zone = self._compute_cte(trajectory)
        stage_results['cte'] = cte
        stage_results['dist_to_zone'] = dist_to_zone

        # 是否在保护空域内
        inside_protected = cte < self.cte_threshold  # (B,)
        stage_results['inside_protected'] = inside_protected

        # ============================================================
        # Stage 2: 仅对进入保护空域的目标继续 (CTE 超标直接过滤)
        # ============================================================
        cte_pass = cte < self.r_protect  # CTE < R_protect?
        stage_results['cte_pass'] = cte_pass

        # ============================================================
        # Stage 3: 运动意图提取
        # ============================================================
        # 3a. 朝向意图角
        heading_angle, heading_aligned = self._compute_heading_intent(trajectory, intent_probs)
        stage_results['heading_angle'] = heading_angle
        stage_results['heading_aligned'] = heading_aligned

        # 3b. 稳定性分析
        long_stab, lat_stab = self._compute_stability(trajectory)
        stage_results['longitudinal_stability'] = long_stab
        stage_results['lateral_stability'] = lat_stab

        # 可学习的稳定性评分
        stab_input = torch.cat([
            cte.unsqueeze(-1),
            heading_angle.unsqueeze(-1) / 180.0,
            long_stab.unsqueeze(-1),
            lat_stab.unsqueeze(-1),
            (trajectory[:, :, 3:6].norm(dim=-1).std(dim=-1)).unsqueeze(-1),  # 速度波动
            (trajectory[:, :, 3:6].norm(dim=-1).mean(dim=-1)).unsqueeze(-1), # 平均速度
        ], dim=-1)  # (B, 6)
        stab_score = self.stability_encoder(stab_input).squeeze(-1)  # (B,)
        stage_results['stability_score'] = stab_score

        # ============================================================
        # Stage 4: 缓冲区管理 — Buffer Size >= W?
        # ============================================================
        # 简化: 用速度稳定性作为 buffer 充足的代理
        # (完整实现需要维护 per-target buffer, 这里用轨迹长度 T 作为近似)
        buffer_sufficient = torch.ones(B, dtype=torch.bool, device=device)  # 假设 always OK
        if T < self.buffer_size:
            buffer_sufficient = torch.zeros(B, dtype=torch.bool, device=device)
        stage_results['buffer_sufficient'] = buffer_sufficient

        # ============================================================
        # Stage 5: 威胁时效判决 — D_t < D_warn OR TTA_t < T_warn?
        # ============================================================
        speed = torch.norm(trajectory[:, -1, 3:6], dim=-1) + 1e-6  # (B,) 当前速度

        tta = self._compute_tta(dist_to_zone, speed)  # 到达时间
        stage_results['tta'] = tta

        distance_threat = dist_to_zone < self.d_warn
        time_threat = tta < self.t_warn
        stage_results['distance_threat'] = distance_threat
        stage_results['time_threat'] = time_threat

        # 可学习的威胁评分
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
        # Stage 6: 飞行姿态确认 — 正对保护区 + 稳定飞行 → 预测
        # ============================================================
        # 综合判决: 所有条件都满足才触发
        all_conditions = (
            cte_pass &                          # CTE < R_protect
            inside_protected &                  # 在保护空域内
            heading_aligned &                   # 朝向对准
            buffer_sufficient &                 # buffer 充足
            (distance_threat | time_threat)     # 距离或时间威胁
        )

        # 可学习的朝向评分 (融合意图信息)
        heading_input = torch.cat([
            heading_angle.unsqueeze(-1) / 180.0,
            long_stab.unsqueeze(-1),
            lat_stab.unsqueeze(-1),
            intent_probs,  # (B, num_classes)
        ], dim=-1)  # (B, 3 + num_classes)
        heading_conf = self.heading_encoder(heading_input).squeeze(-1)  # (B,)
        stage_results['heading_confidence'] = heading_conf

        # 最终评分: 硬判决 + 可学习评分 的组合
        condition_score = all_conditions.float()  # 硬条件满足度 [0, 1]
        learned_score = (
            0.25 * stab_score +
            0.35 * threat_score +
            0.40 * heading_conf
        )
        # 硬条件作为门控: 不满足则大幅降低评分
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
