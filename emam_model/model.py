"""
完整轨迹预测模型: TrajectoryPredictor
EMam-SE + IA-DTP + UA-PGD 三模块串联
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .emam_se import EnhancedMambaSE
from .ia_dtp import IntentAwareDTP, IntentType, NUM_INTENT_CLASSES
from .ua_pgd import UncertaintyAwarePGD
from .trigger import EventDrivenTrigger, SimpleTrigger, FunnelTrigger


class TrajectoryPredictor(nn.Module):
    """
    完整轨迹预测模型

    流程:
    Input Trajectory → EMam-SE → IA-DTP → UA-PGD → 3D Displacement Prediction

    可选集成:
    - 事件驱动触发器 (Event-Driven Trigger)
    - 不确定性量化输出
    """
    def __init__(
        self,
        # 输入配置
        input_dim: int = 6,           # [x, y, z, vx, vy, vz]
        history_len: int = 20,        # 历史轨迹帧数
        pred_len: int = 20,           # 预测帧数
        # EMam-SE
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        emam_n_layers: int = 2,
        dropout: float = 0.1,
        # IA-DTP
        num_intent_classes: int = NUM_INTENT_CLASSES,
        intent_hidden: int = 128,
        # 触发器
        use_trigger: bool = True,
        trigger_mode: str = 'funnel',    # 'simple' | 'funnel' | 'learned'
        trigger_threshold: float = 0.5,
        # 损失权重
        loss_weights: Dict[str, float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.pred_len = pred_len
        self.d_model = d_model

        if loss_weights is None:
            loss_weights = {'displacement': 1.0, 'intent': 0.1, 'uncertainty': 0.05}
        self.loss_weights = loss_weights

        # === 核心三模块 ===
        self.emam_se = EnhancedMambaSE(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=emam_n_layers,
            dropout=dropout
        )

        self.ia_dtp = IntentAwareDTP(
            d_model=d_model,
            num_classes=num_intent_classes,
            hidden_dim=intent_hidden
        )

        self.ua_pgd = UncertaintyAwarePGD(
            d_model=d_model,
            pred_len=pred_len,
            num_intent_classes=num_intent_classes,
            trajectory_dim=input_dim,
            dropout=dropout
        )

        # === 触发器 ===
        if use_trigger:
            if trigger_mode == 'funnel':
                self.trigger = FunnelTrigger(
                    feature_dim=input_dim,
                    num_intent_classes=num_intent_classes,
                )
            elif trigger_mode == 'learned':
                self.trigger = EventDrivenTrigger(
                    feature_dim=input_dim,
                    num_intent_classes=num_intent_classes,
                    threat_weight=0.3,
                    intent_weight=0.3,
                    spatial_weight=0.4,
                    trigger_threshold=trigger_threshold,
                )
            else:  # 'simple'
                self.trigger = SimpleTrigger()
            self._trigger_mode = trigger_mode
        else:
            self.trigger = None
            self._trigger_mode = 'none'

        # === 历史意图缓冲区 ===
        self.register_buffer('intent_history', torch.zeros(history_len, num_intent_classes))

    def forward(
        self,
        history: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        return_all: bool = False,
        force_predict: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            history: (B, T, input_dim) 历史轨迹
            intent_labels: (B,) 可选, 意图类别标签
            return_all: 是否返回中间结果
            force_predict: 强制预测 (绕过触发器)
        Returns:
            dict:
                predictions: (B, pred_len, 3) 未来3D位移
                intent_logits: (B, num_classes) 意图logits
                intent_weights: (B, num_classes) 意图权重
                trigger_decision: (B,) bool, 是否触发
                uncertainty: (B, pred_len, 3) 不确定性
        """
        B, T, C = history.shape

        # === Step 0: 输入归一化 ===
        # 特征: [x, y, z, vx, vy, vz] — 位置 ~100m, 速度 ~10m/s
        # 缩放到 ~[0, 5], 所有内部计算在归一化空间进行, 输出再逆归一化
        _scale_pos = 100.0
        _scale_vel = 10.0
        _scale = history.new_tensor([_scale_pos, _scale_pos, _scale_pos,
                                     _scale_vel, _scale_vel, _scale_vel])
        h = history / _scale.unsqueeze(0).unsqueeze(0)  # 后续统一用 h

        # === Step 1: EMam-SE 编码 ===
        encoded = self.emam_se(h)  # (B, T, d_model)

        # === Step 2: IA-DTP 意图感知 ===
        dtp_out = self.ia_dtp(encoded, historical_trajectory=h)
        global_anchor = dtp_out['global_anchor']       # (B,1,d_model)
        intent_logits = dtp_out['intent_logits']        # (B,num_classes)
        intent_weights = dtp_out['intent_weights']      # (B,num_classes)
        enhanced_features = dtp_out['enhanced_features']  # (B,T,d_model)

        # === Step 3: 触发器决策 (用归一化轨迹) ===
        trigger_out = None
        if self.trigger is not None and not force_predict:
            trigger_out = self.trigger(
                trajectory=h,
                intent_logits=intent_logits,
                intent_history=self.intent_history.T
            )
            trigger_decision = trigger_out['trigger_decision']
        else:
            trigger_decision = torch.ones(B, dtype=torch.bool, device=h.device)

        # === Step 4: UA-PGD 解码 ===
        pgd_out = self.ua_pgd(
            encoded_feat=encoded,
            global_anchor=global_anchor,
            historical_trajectory=h,
            intent_weights=intent_weights,
            return_uncertainty=True
        )
        predictions = pgd_out['predictions']            # (B, pred_len, 3) — 归一化空间
        uncertainties = pgd_out['logvar']
        physics_trajectory = pgd_out.get('physics_trajectory',
                                         torch.zeros(B, self.pred_len, 3, device=h.device))
        gate_inertia = pgd_out.get('gate_inertia',
                                   torch.zeros(B, self.pred_len, device=h.device))

        # 未触发目标: 匀速基线 (归一化空间)
        if not force_predict and (self.trigger is not None):
            last_vel = h[:, -1, 3:6]                               # (B, 3) 归一化速度
            vel_recent = h[:, -3:, 3:6]                            # (B, 3, 3)
            w = torch.tensor([0.2, 0.3, 0.5], device=h.device)
            last_vel_smooth = (vel_recent * w.view(1, 3, 1)).sum(dim=1)
            step_indices = torch.arange(1, self.pred_len + 1, device=h.device).float()
            step_indices = step_indices.view(1, -1, 1) * 0.1
            baseline = last_vel_smooth.unsqueeze(1) * step_indices  # (B, pred_len, 3)
            mask = trigger_decision.float().unsqueeze(-1).unsqueeze(-1)
            predictions = predictions * mask + baseline * (1 - mask)

        # === Step 5: 逆归一化 — 位置位移缩回原始坐标 ===
        predictions = predictions * _scale_pos
        physics_trajectory = physics_trajectory * _scale_pos

        # 更新意图历史缓冲区 (防止 NaN 污染持久状态)
        if self.training:
            latest_intent = intent_weights.detach().mean(dim=0)  # (num_classes,)
            if torch.isfinite(latest_intent).all():
                self.intent_history = torch.cat([
                    self.intent_history[1:], latest_intent.unsqueeze(0)
                ], dim=0)

        result = {
            'predictions': predictions,          # (B, pred_len, 3)
            'intent_logits': intent_logits,       # (B, num_classes)
            'intent_weights': intent_weights,    # (B, num_classes)
            'trigger_decision': trigger_decision, # (B,)
            'uncertainty': uncertainties,        # (B, pred_len, 3)
            'physics_trajectory': physics_trajectory,  # (B, pred_len, 3)
            'gate_inertia': gate_inertia,              # (B, pred_len)
        }

        if return_all:
            result['encoded_features'] = encoded
            result['global_anchor'] = global_anchor
            result.update(dtp_out)
            if trigger_out is not None:
                result['trigger_score'] = trigger_out['trigger_score']
                result['maneuver_score'] = trigger_out['maneuver_score']

        return result

    def compute_loss(
        self,
        predictions: torch.Tensor,       # (B, pred_len, 3) 预测位移
        uncertainty: torch.Tensor,        # (B, pred_len, 3) logvar
        targets: torch.Tensor,            # (B, pred_len, 3) 真值位移
        intent_logits: torch.Tensor,      # (B, num_classes)
        intent_labels: torch.Tensor,       # (B,)
        intent_weights: torch.Tensor = None,
        physics_trajectory: torch.Tensor = None,  # (B, pred_len, 3) 物理外推轨迹
        gate_inertia: torch.Tensor = None,        # (B, pred_len)     惯性门控
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失 = 位移损失 + 意图损失 + 不确定性损失 + 物理一致性损失（可选）

        Args:
            physics_trajectory: 物理模型外推轨迹，若为None则跳过物理损失
            gate_inertia:      惯性门控值，若为None则跳过物理损失
        """
        device = predictions.device

        # 1. 位移预测损失 (MSE)
        loss_disp = F.mse_loss(predictions, targets)

        # 2. 意图分类损失 (CrossEntropy)
        loss_intent = F.cross_entropy(intent_logits, intent_labels)

        # 3. 不确定性损失 (负对数似然)
        # NLL = (pred - target)^2 / (2*var) + log(sqrt(2*pi*var))
        # 其中 var = exp(logvar)
        logvar_clamped = uncertainty.clamp(-10, 10)
        var = torch.exp(logvar_clamped)  # 防止数值爆炸
        nll = ((predictions - targets) ** 2 / (2 * var + 1e-8)
               + logvar_clamped * 0.5)
        loss_uncertainty = nll.mean()

        # 4. 物理一致性损失（可选）
        loss_physics = torch.tensor(0.0, device=device)
        if physics_trajectory is not None and gate_inertia is not None and self.ua_pgd is not None:
            loss_physics = self.ua_pgd.compute_physics_loss(
                predictions, physics_trajectory, targets, gate_inertia
            )

        # 5. 总损失
        total_loss = (
            self.loss_weights['displacement'] * loss_disp +
            self.loss_weights['intent'] * loss_intent +
            self.loss_weights['uncertainty'] * loss_uncertainty +
            loss_physics
        )

        return {
            'total_loss': total_loss,
            'loss_displacement': loss_disp,
            'loss_intent': loss_intent,
            'loss_uncertainty': loss_uncertainty,
            'loss_physics': loss_physics,
        }

    def predict(self, history: torch.Tensor) -> torch.Tensor:
        """推理接口: 直接返回预测位移"""
        with torch.no_grad():
            out = self.forward(history, force_predict=True)
            return out['predictions']

    def predict_with_uncertainty(
        self, history: torch.Tensor, n_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        不确定性感知推理 (Monte Carlo Dropout)
        多次采样取均值和方差
        """
        self.train()  # 开启dropout
        all_preds = []
        for _ in range(n_samples):
            out = self.forward(history, force_predict=True)
            all_preds.append(out['predictions'])

        self.eval()
        all_preds = torch.stack(all_preds, dim=0)  # (n_samples, B, pred_len, 3)
        mean_pred = all_preds.mean(dim=0)
        std_pred = all_preds.std(dim=0)

        return {
            'prediction': mean_pred,
            'uncertainty': std_pred,
            'all_samples': all_preds
        }
