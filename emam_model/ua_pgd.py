"""
Uncertainty-Aware Physics-Guided Decoder (UA-PGD)
=================================================

核心四机制：
  1. 正交步长编码 (Orthogonal Step Encoding)
  2. 不确定性自适应全局锚定 (Uncertainty Adaptive Global Anchoring)
  3. 物理惯性门控 (Physical Inertia Gating)
  4. 运动学解耦反馈 (Kinematic Decoupling Feedback)

输入:
    encoded_feat: (B, T, d_model)       EMam-SE 编码特征
    global_anchor: (B, 1, d_model)    IA-DTP 生成的全局目标锚点
    historical_trajectory: (B, T, 6)    [x, y, z, vx, vy, vz]

输出:
    predictions: (B, pred_len, 3)      未来3D位移预测
    logvar: (B, pred_len, 3)           对数方差（不确定性）
    (可选) intermediate_gates: 门控状态序列（用于意图读出）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================================
# 子模块 1: 正交步长编码器
# ============================================================================

class OrthogonalStepEncoder(nn.Module):
    """
    正交步长编码 (Orthogonal Step Encoding)

    为每个预测步生成正交的步长表征，使得：
      - 相邻步之间的表征区分度高
      - 远期步的编码不受近期步编码的干扰
      - 网络能够区分"预测第1步"和"预测第10步"的状态

    实现：多频率正弦-余弦对偶编码，频率指数增长。
    理论上任意两个不同步的编码内积趋近于0（正交性）。
    """
    def __init__(self, pred_len: int, d_model: int):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # 生成固定的正交基（不可学习），避免过拟合
        steps = torch.arange(1, pred_len + 1).float()           # [1, ..., pred_len]
        # 频率从 1 指数增长到 ~148，保证高频分量用于区分长步
        freqs = torch.exp(torch.linspace(0, math.log(150), d_model // 2))

        pe = torch.zeros(pred_len, d_model)
        pe[:, 0::2] = torch.sin(steps.unsqueeze(1) * freqs)    # 偶数维：sin
        pe[:, 1::2] = torch.cos(steps.unsqueeze(1) * freqs)   # 奇数维：cos
        self.register_buffer('_pe', pe)                         # (pred_len, d_model)

        # 可学习的缩放因子，让模型自适应调整编码幅度
        self.step_scale = nn.Parameter(torch.ones(1))

    def forward(self) -> torch.Tensor:
        """
        Returns:
            step_encoding: (pred_len, d_model)  正交步长编码
        """
        return self.step_scale * self._pe


# ============================================================================
# 子模块 2: 物理惯性门控 (核心机制)
# ============================================================================

class PhysicsInertiaGate(nn.Module):
    """
    物理惯性门控 (Physical Inertia Gating)

    核心思想：预测步的输出 = 物理模型外推 × 惯性门控 + 神经网络预测 × (1 - 惯性门控)
    同时叠加全局锚点的拉回效应。

    四个门控:
        gate_inertia:    历史惯性保留比例 (0~1)
                         - 机动时上升：跟随突发动作
                         - 巡航/悬停时下降：依赖模型预测
        gate_anchor:     锚点拉回强度 (0~1)
                         - 步越远越强，防止长期预测发散
        gate_confidence: 模型置信度 (0~1)，从不确定性映射
                         - 高不确定性时，降低神经网络权重
        gate_mode:       飞行模式调制 (0~1)，从意图权重映射
                         - 悬停时强制锚点拉回增强

    输入: 最后时刻编码特征 (B, d_model)
          意图权重 (B, num_intent_classes)
          步长编码 (pred_len, d_model)
    输出: 门控参数序列 (B, pred_len, num_gates)
    """
    def __init__(self, d_model: int, num_intent_classes: int, pred_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_intent_classes = num_intent_classes
        self.num_gates = 4  # inertia, anchor, confidence, mode

        # 从编码特征预测基础门控 (gate_inertia, gate_anchor)
        hidden = d_model // 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),        # gate_inertia, gate_anchor
            nn.Sigmoid()                  # 约束到 (0,1)
        )

        # 从意图权重预测模式门控（逐类别）
        # gate_mode: (B, num_intent_classes) 每种意图对应一个模式值
        self.intent_to_mode = nn.Sequential(
            nn.Linear(num_intent_classes, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, num_intent_classes),
            nn.Sigmoid()
        )

        # 正交步长编码投影（用于计算步相关锚点权重）
        self.step_proj = nn.Linear(d_model, 1, bias=False)

        # 门控温度参数（避免门控过于 binary）
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 初始化：默认以物理惯性为主，锚点拉回适中
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.xavier_uniform_(self.gate_mlp[2].weight)
        # 初始化 gate_inertia 偏高 (~0.6)，gate_anchor 适中 (~0.3)
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
            gate_inertia:      (B, pred_len)    惯性保留比例
            gate_anchor:       (B, pred_len)    锚点拉回强度
            gate_confidence:   (B, pred_len)    置信度（暂无不确定性输入，设为1）
            gate_mode:         (B, num_intent_classes)  逐类别模式强度
            gate_mode_effective: (B, pred_len)  加权映射后的步级模式强度
        """
        B = last_encoded.shape[0]
        P = step_encoding.shape[0]          # pred_len

        # --- 1. 基础门控 (从编码特征预测) ---
        base_gates = self.gate_mlp(last_encoded)          # (B, 2)
        gate_inertia_base = base_gates[:, 0]               # (B,)
        gate_anchor_base = base_gates[:, 1]                # (B,)

        # --- 2. 步长依赖的锚点权重 (步越远锚点越强) ---
        # step_encoding: (pred_len, d_model)
        step_weights_raw = self.step_proj(step_encoding).squeeze(-1)   # (pred_len,)
        step_weights = torch.sigmoid(step_weights_raw)                # (pred_len,)
        # 步权重从 ~0.2 逐渐增加到 ~0.8
        step_weights = 0.2 + 0.6 * step_weights

        # gate_anchor 随步长增长
        gate_anchor = gate_anchor_base.unsqueeze(1) * step_weights.unsqueeze(0)  # (B, pred_len)
        gate_anchor = gate_anchor.clamp(0.0, 1.0)

        # --- 3. 惯性门控：步长增加时下降（远期预测更依赖模型） ---
        # gate_inertia 反比于步长
        inertia_step_decay = torch.linspace(1.0, 0.4, P, device=last_encoded.device)  # 早期1.0 → 远期0.4
        gate_inertia = gate_inertia_base.unsqueeze(1) * inertia_step_decay.unsqueeze(0)  # (B, pred_len)
        gate_inertia = gate_inertia.clamp(0.0, 1.0)

        # --- 4. 模式门控 (逐意图类别预测) ---
        # intent_to_mode 输出 (B, num_intent_classes)，与 intent_weights 形状相同
        # gate_mode 范围 [0,1]，高值表示该意图对应的锚点拉回增强
        gate_mode = self.intent_to_mode(intent_weights)   # (B, num_intent_classes)

        # 将逐意图类别的 gate_mode 映射为预测步长的加权平均
        # gate_mode_effective[b, p] = sum_c(gate_mode[b,c] * intent_weights[b,c])
        # 物理含义：在当前意图分布下，锚点拉回的整体强度
        gate_mode_per_step = gate_mode * intent_weights          # (B, num_intent_classes)
        gate_mode_effective = gate_mode_per_step.sum(dim=1, keepdim=True)  # (B, 1)
        gate_mode_effective = gate_mode_effective.expand(-1, P)  # (B, P)

        # gate_confidence：暂无不确定性输入，默认全 1（后续可从外部输入的不确定性特征接入）
        gate_confidence = torch.ones(B, P, device=last_encoded.device)

        return gate_inertia, gate_anchor, gate_confidence, gate_mode, gate_mode_effective


# ============================================================================
# 子模块 3: 运动学物理模型
# ============================================================================

class KinematicPhysicsModel(nn.Module):
    """
    运动学解耦物理模型 (Kinematic Physics Model)

    对 位置/速度/加速度 三自由度独立建模，保证物理一致性：

    位置外推: p_{t+dt} = p_t + v_t * dt + 0.5 * a_t * dt^2
    速度外推: v_{t+dt} = v_t + a_t * dt
    加速度上界: |a| <= max_acc (无人机典型值: 15 m/s²)

    仅作为强归纳偏置，在惯性门控开启时混合使用，不强制约束。
    """
    def __init__(self, trajectory_dim: int = 6, max_accel: float = 15.0):
        super().__init__()
        self.traj_dim = trajectory_dim          # 6: [x,y,z,vx,vy,vz]
        self.max_accel = max_accel
        self.pos_dim = 3                         # 位置维度: [x,y,z]

        # 加速度估算 MLP：从状态预测加速度修正量
        # 输入: 位置(3) + 速度(3)
        # 输出: 加速度修正(3)
        self.acc_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 3)
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        基于运动学方程外推一步位移（相对位移）。

        Args:
            trajectory: (B, T, 6)  [x, y, z, vx, vy, vz]
        Returns:
            physics_delta: (B, 3)  相对位移增量（与neural_delta量纲一致）
        """
        dt = 0.1

        # 取最后时刻和倒数第二时刻
        last = trajectory[:, -1, :]   # (B, 6)
        prev = trajectory[:, -2, :]  # (B, 6)

        # 基准位置（绝对）
        base_pos = last[:, :3]       # (B, 3)
        # 速度（差分估算）
        vel = (last[:, :3] - prev[:, :3]) / dt
        vel = torch.clamp(vel, -50, 50)

        # 加速度网络估算
        state_input = torch.cat([base_pos, vel], dim=-1)
        acc_pred = self.acc_net(state_input)       # (B, 3)
        acc_pred = torch.clamp(acc_pred, -self.max_accel, self.max_accel)

        # 相对位移：p_{t+1} = p_t + v_t*dt + 0.5*a_t*dt^2
        physics_delta = vel * dt + 0.5 * acc_pred * (dt ** 2)  # (B, 3)

        return physics_delta

    def multi_step(
        self,
        trajectory: torch.Tensor,
        pred_len: int
    ) -> torch.Tensor:
        """
        多步物理外推（不使用神经网络，纯运动学方程）。

        输出相对位移：从历史最后位置开始的外推位移增量序列。
        输出形状：(B, pred_len, 3)，与 neural_delta 对齐（量纲一致）。

        Args:
            trajectory: (B, T, 6)  历史轨迹 [x,y,z,vx,vy,vz]
            pred_len: 预测步数
        Returns:
            physics_trajectory: (B, pred_len, 3)  相对位移序列
        """
        B = trajectory.shape[0]
        dt = 0.1

        # 基准位置：历史最后时刻的位置（绝对坐标）
        base_pos = trajectory[:, -1, :3].clone()  # (B, 3)

        # 初始速度：从最近两帧差分估算
        vel = (trajectory[:, -1, :3] - trajectory[:, -2, :3]) / dt
        vel = torch.clamp(vel, -50, 50)

        # 位置累积（从基准位置出发，累计位移）
        pos_delta = torch.zeros(B, 3, device=trajectory.device)
        physics_preds = []

        # 加速度状态（用于指数平滑）
        acc_state = torch.zeros_like(vel)

        for step in range(pred_len):
            # 加速度网络估算
            state_input = torch.cat([base_pos + pos_delta, vel], dim=-1)  # 用绝对位置
            acc_pred = self.acc_net(state_input)  # (B, 3)
            acc_pred = torch.clamp(acc_pred, -self.max_accel, self.max_accel)

            # 指数平滑加速度（避免突变）
            acc_state = 0.7 * acc_state + 0.3 * acc_pred

            # 更新速度和位移
            vel = vel + acc_state * dt
            pos_delta = pos_delta + vel * dt

            # 记录相对位移（不含基准位置）
            physics_preds.append(pos_delta.clone())

        return torch.stack(physics_preds, dim=1)  # (B, pred_len, 3)


# ============================================================================
# 子模块 4: 神经网络解码层
# ============================================================================

class NeuralDecoder(nn.Module):
    """
    神经网络解码器：将编码特征解码为位移增量

    使用步长编码调制特征，确保不同步输出不同预测。
    """
    def __init__(self, d_model: int, trajectory_dim: int = 6):
        super().__init__()
        self.d_model = d_model

        # 特征 → 隐状态
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        # 输出头：位移 + 不确定性（log(variance)）
        self.delta_head = nn.Linear(d_model, 3)
        self.var_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 3),
            nn.Softplus(),  # 输出 log(variance) ∈ (0, +∞)，确保方差为正
        )

    def forward(
        self,
        encoded: torch.Tensor,         # (B, d_model)  编码特征（取最后时刻或池化）
        step_encoding: torch.Tensor,    # (pred_len, d_model)  步长编码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            neural_delta:  (B, pred_len, 3)  神经网络位移增量
            logvar:        (B, pred_len, 3)  对数方差（不确定性）
        """
        B = encoded.shape[0]
        P = step_encoding.shape[0]

        # 特征 + 步长编码
        feat = self.proj(encoded)                           # (B, d_model)
        feat = feat.unsqueeze(1) + step_encoding.unsqueeze(0)  # (B, pred_len, d_model)

        # 展平批量和步长维度，统一解码
        feat_flat = feat.reshape(B * P, self.d_model)       # (B*P, d_model)

        delta_flat = self.delta_head(feat_flat)             # (B*P, 3)
        var_flat = self.var_head(feat_flat)                 # (B*P, 3)

        delta = delta_flat.reshape(B, P, 3)                 # (B, pred_len, 3)
        logvar = var_flat.reshape(B, P, 3)                  # (B, pred_len, 3)

        return delta, logvar


# ============================================================================
# 主模块: UA-PGD
# ============================================================================

class UncertaintyAwarePGD(nn.Module):
    """
    不确定性感知物理引导解码器 (Uncertainty-Aware Physics-Guided Decoder)

    整体流程::

        encoded_feat ──────────────────────────────────┐
                                                          │
        global_anchor ──→ 与 encoded 融合 ──┐           │
                                           ↓           │
        历史轨迹 ──→ 运动学物理模型 ─→ physics_delta ─→ ┴→ 物理惯性门控混合 → 神经网络残差修正 → 最终位移
                                           ↑           │
        步长编码 ───────────────────────────────────────┘

    混合公式::

        pred[t] = gate_inertia[t] * physics_delta[t]
                + (1 - gate_inertia[t]) * (neural_delta[t] + global_anchor_contribution)
                + gate_mode[t] * anchor_pull[t]

    其中 anchor_pull[t] = gate_anchor[t] * (global_anchor - current_pos)

    门控特性:
        - gate_inertia:  机动时高，巡航时低，远期步更低
        - gate_anchor:   步越远越强，悬停时强制增强
        - gate_mode:     从意图权重映射，悬停意图 → 锚点主导
        - gate_confidence: 从不确定性映射，高不确定性 → 锚点拉回增强
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

        # 子模块
        self.step_encoder = OrthogonalStepEncoder(pred_len, d_model)
        self.physics_model = KinematicPhysicsModel(trajectory_dim)
        self.neural_decoder = NeuralDecoder(d_model, trajectory_dim)

        self.physics_gate = PhysicsInertiaGate(
            d_model=d_model,
            num_intent_classes=num_intent_classes,
            pred_len=pred_len,
        )

        # 全局锚点融合：锚点向量经 MLP 投影到位置空间
        self.anchor_to_pos = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )

        # 特征压缩层（用于将编码特征投影到解码空间）
        self.feat_compress = nn.Linear(d_model, d_model)

        # Dropout（防止过拟合）
        self.dropout = nn.Dropout(p=dropout)

        # 物理一致性损失权重（供外部 compute_loss 使用）
        self.physics_loss_weight = 0.05

        # 初始化
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
        主前向传播。

        Args:
            encoded_feat:         (B, T, d_model)   EMam-SE 编码特征
            global_anchor:        (B, 1, d_model)   全局目标锚点
            historical_trajectory:(B, T, 6)          [x,y,z,vx,vy,vz]
            return_uncertainty:  是否返回不确定性
            intent_weights:       (B, num_intent_classes)  意图权重（若为None则用均匀分布）

        Returns:
            predictions:  (B, pred_len, 3)  未来3D位移
            logvar:      (B, pred_len, 3)  对数方差
            (可选) gates: (B, pred_len, 4)  四个门控值（用于意图读出）
        """
        B, T, D = encoded_feat.shape
        P = self.pred_len
        device = encoded_feat.device

        # -------------------------------------------------------------------------
        # 1. 正交步长编码
        # -------------------------------------------------------------------------
        step_encoding = self.step_encoder()                        # (P, d_model)
        step_encoding = step_encoding.to(device)

        # -------------------------------------------------------------------------
        # 2. 特征准备
        # -------------------------------------------------------------------------
        # 取最后时刻编码特征
        last_encoded = encoded_feat[:, -1, :]                       # (B, d_model)
        last_encoded = self.feat_compress(last_encoded)           # (B, d_model)
        last_encoded = self.dropout(last_encoded)

        # -------------------------------------------------------------------------
        # 3. 全局锚点投影为位置目标
        # -------------------------------------------------------------------------
        # anchor 从 (B, 1, d_model) → (B, 3) 位置
        anchor_pos = self.anchor_to_pos(global_anchor.squeeze(1))  # (B, 3)

        # -------------------------------------------------------------------------
        # 4. 物理惯性门控
        # -------------------------------------------------------------------------
        # 意图权重：若未提供，使用均匀分布
        if intent_weights is None:
            intent_weights = torch.ones(B, self.num_intent_classes, device=device)
            intent_weights = intent_weights / self.num_intent_classes

        gate_inertia, gate_anchor, gate_confidence, gate_mode, gate_mode_effective = self.physics_gate(
            last_encoded=last_encoded,
            intent_weights=intent_weights,
            step_encoding=step_encoding,
        )
        # 各门控形状:
        #   gate_inertia:      (B, P)
        #   gate_anchor:       (B, P)
        #   gate_confidence:   (B, P)
        #   gate_mode:         (B, num_intent_classes)  逐意图类别模式强度
        #   gate_mode_effective: (B, P)  加权映射后的步级模式强度

        # -------------------------------------------------------------------------
        # 5. 物理模型多步外推
        # -------------------------------------------------------------------------
        # physics_trajectory: (B, P, 3)  多步物理位移序列（每步是从初始位置的外推位移）
        physics_trajectory = self.physics_model.multi_step(
            historical_trajectory, pred_len=P
        )  # (B, P, 3)

        # -------------------------------------------------------------------------
        # 6. 神经网络解码
        # -------------------------------------------------------------------------
        neural_delta, logvar = self.neural_decoder(
            encoded=last_encoded,           # (B, d_model)
            step_encoding=step_encoding,   # (P, d_model)
        )  # neural_delta: (B, P, 3), logvar: (B, P, 3)

        # -------------------------------------------------------------------------
        # 7. 物理惯性门控混合（核心机制）
        # -------------------------------------------------------------------------
        # 位置基准：最后时刻位置
        last_pos = historical_trajectory[:, -1, :3]                  # (B, 3)
        last_pos_expanded = last_pos.unsqueeze(1).expand(-1, P, -1)  # (B, P, 3)

        # 锚点拉回向量：(anchor_pos - last_pos) × gate_anchor
        anchor_pull = (anchor_pos.unsqueeze(1) - last_pos_expanded)  # (B, P, 3)
        anchor_pull = anchor_pull * gate_anchor.unsqueeze(-1)         # (B, P, 3)

        # 神经网络预测 + 锚点拉回（考虑置信度）
        confidence_factor = gate_confidence.unsqueeze(-1)  # (B, P, 1)
        neural_guided = neural_delta + anchor_pull        # (B, P, 3)
        neural_guided = neural_guided * confidence_factor  # 低置信度时降低神经网络权重

        # 模式调制：悬停时增强锚点主导（mode越高，物理惯性越被锚点拉回压制）
        gate_mode_exp = gate_mode_effective.unsqueeze(-1)  # (B, P, 1)
        inertia_effective = gate_inertia.unsqueeze(-1) * (1.0 - 0.3 * gate_mode_exp)  # (B, P, 1)

        # 混合
        blended = (
            inertia_effective * physics_trajectory
            + (1.0 - inertia_effective) * neural_guided
        )

        # -------------------------------------------------------------------------
        # 8. 运动学约束（可选后处理，不修改梯度）
        # -------------------------------------------------------------------------
        # ★ _kinematic_postprocess 已禁用 — 阈值在归一化空间下单位错误，
        # 会破坏正常预测（train RMSE 0.5m → eval RMSE 853m）。待修复后重新启用。
        # if not self.training:
        #     blended = self._kinematic_postprocess(blended, historical_trajectory)

        # -------------------------------------------------------------------------
        # 9. 不确定性处理
        # -------------------------------------------------------------------------
        # Softplus 输出 ∈ (0, +∞)，但保留 clamp 作为数值安全边界
        # clamp 范围 [-10, 10] 对应方差 [4.5e-5, 2.2e4]，覆盖合理区间
        logvar_clamped = logvar.clamp(-10.0, 10.0)

        return {
            'predictions': blended,              # (B, pred_len, 3)
            'logvar': logvar_clamped,            # (B, pred_len, 3)
            'gate_inertia': gate_inertia,        # (B, pred_len)    惯性保留比例
            'gate_anchor': gate_anchor,          # (B, pred_len)    锚点拉回强度
            'gate_mode': gate_mode,              # (B, num_intent)  逐类别模式强度
            'gate_confidence': gate_confidence,  # (B, pred_len)    置信度
            'gate_mode_effective': gate_mode_effective,  # (B, pred_len) 步级模式强度
            'physics_trajectory': physics_trajectory,  # (B, pred_len, 3) 物理外推位移
            'neural_delta': neural_delta,        # (B, pred_len, 3) 神经位移增量
        }

    def _kinematic_postprocess(
        self,
        predictions: torch.Tensor,
        history: torch.Tensor,
        max_accel: float = 15.0,
        max_jerk: float = 30.0,
    ) -> torch.Tensor:
        """
        推理时的运动学后处理（强制物理一致性）。

        对预测轨迹做运动学可行性检验：
        1. 加速度上界约束
        2. 速度上界约束
        3. 位置突变检验

        此函数仅在推理时调用，不影响训练梯度。
        """
        dt = 0.1
        preds = predictions.clone()

        # 速度约束：最大速度 50 m/s
        max_vel = 50.0

        # 计算预测的速度和加速度
        pos_history = history[:, -1, :3]                      # (B, 3)
        pos_series = torch.cat([pos_history.unsqueeze(1), preds], dim=1)  # (B, P+1, 3)

        for step in range(preds.shape[1]):
            pos_curr = pos_series[:, step + 1, :]
            pos_prev = pos_series[:, step, :]

            # 速度
            vel = (pos_curr - pos_prev) / dt
            vel_norm = torch.norm(vel, dim=-1, keepdim=True)  # (B, 1)
            vel = vel / (vel_norm.clamp(min=1e-6)) * vel_norm.clamp(max=max_vel)
            vel_clamped = vel

            # 加速度
            if step > 0:
                pos_pp = pos_series[:, step - 1, :]
                vel_prev = (pos_prev - pos_pp) / dt
                acc = (vel_clamped - vel_prev) / dt
                acc_norm = torch.norm(acc, dim=-1, keepdim=True)

                # 加速度约束
                acc = acc / (acc_norm.clamp(min=1e-6)) * acc_norm.clamp(max=max_accel)

                # 修正位置：使用约束后的加速度重新计算
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
        物理一致性损失：鼓励在高惯性区域（机动）预测接近物理外推，
                       在低惯性区域（巡航）接近神经网络预测。

        损失 = |pred - physics| * gate_inertia + |pred - neural| * (1 - gate_inertia)

        但由于 neural_delta 未知，这里简化为：
        损失 = |pred - physics| * gate_inertia.mean()

        即：机动时（gate_inertia高），强制预测贴近物理模型。
        """
        residual = (predictions - physics_trajectory).abs()   # (B, P, 3)
        physics_loss = (residual * gate_inertia.unsqueeze(-1)).mean()
        return physics_loss * self.physics_loss_weight
