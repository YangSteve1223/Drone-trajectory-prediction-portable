#!/usr/bin/env python3
"""双模型速度硬切换推理 + LoRA在线持续学习。"""

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

# 速度阈值 (m/s): <5 → 低速模型, ≥5 → 高速模型
S_THRESHOLD = 5.0

# 软融合: 替代硬切换, 在阈值附近平滑过渡
# α = sigmoid((speed - threshold) / temperature)
# temperature 越小过渡越陡 (0.1 ≈ 硬切换), 越大过渡越平滑
# 过渡区外使用硬分配, 避免一个模型在另一个的数据域上预测
SOFT_FUSION_ENABLED = True
FUSION_TEMPERATURE = 1.2       # sigmoid 温度参数
FUSION_HALF_WIDTH = 3.0        # 过渡区半宽 (m/s), 仅在 [threshold-width, threshold+width] 内混合

# Z轴纠正: HIGH模型处理DESCEND意图的Z分量
# 问题: 模型对DESCEND的预测概率偏低 (真下降仅5-12%, 假阳性也是6-12%)
# 原来一刀切20%阈值 → 真假都压制95% → 真下降的Z预测被错误清零
# 修复: 三段式连续过渡, 在不确定区(5-20%)部分允许Z分量
Z_CORRECTION_ENABLED = True
Z_DESCEND_THRESHOLD_LOW = 0.05   # 低于此: 确信非下降, 强压制
Z_DESCEND_THRESHOLD_HIGH = 0.20  # 高于此: 确信下降, 不压制
Z_DAMPEN_STRONG = 0.05           # 强压制: Z保留5% (非下降时几乎清零Z漂移)
Z_DAMPEN_WEAK = 0.30             # 弱压制: Z保留30% (不确定区, 允许部分Z)

# Z轴趋势感知: 用多种信号判断模型DESCEND预测是否可信
# 信号1: Z历史趋势 (仅当Z历史有变化时有效, SimCruise的DESCEND样本Z历史全为0)
# 信号2: 模型Z轴不确定性 — 低不确定性=模型自信=减弱压制
# 信号3: 预测Z位移量级 — 大位移下降=更像真下降
Z_TREND_ENABLED = True
Z_TREND_WINDOW = 10              # 分析最近N帧的Z趋势
Z_TREND_DESCENT_THRESH = -0.03   # Z速度阈值 (m/s per frame)
Z_TREND_DAMPEN_BOOST = 0.15      # 每个信号满足后 dampen因子增加值

# 模型不确定性感知: 利用UA-PGD输出的logvar判断Z预测是否可靠
Z_UNCERTAINTY_ENABLED = True
Z_UNCERTAINTY_THRESH = -2.0      # logvar < -2.0 → 低不确定性 → 模型自信 → 减弱压制
Z_UNCERTAINTY_BOOST = 0.20       # 低不确定性时 dampen因子增加值

# 预测Z位移量级: 大位移下降更可能是真下降
Z_MAGNITUDE_ENABLED = True
Z_MAGNITUDE_THRESH = 5.0         # 预测Z位移 < -5m → 可能真下降
Z_MAGNITUDE_BOOST = 0.15         # 大位移下降时 dampen因子增加值

# 熵引导物理融合: LOW模型在意图不确定时偏向物理外推
# 诊断: 最差样本(TURN→STRAIGHT错配)熵>0.7, 神经anchor预测直线但物理模型保留转弯动量
# 熵越高 → 越依赖物理外推 → 保留转弯, 抑制错误的STRAIGHT anchor
ENTROPY_PHYSICS_ENABLED = False  # 测试未通过: 物理模型过于简单(2帧速度估算), 叠加后反而降低转弯精度
ENTROPY_THRESHOLD = 0.4         # 意图熵>0.4时触发物理偏向
ENTROPY_MAX_BLEND = 0.35        # 最多额外叠加35%物理权重

# 4-class → 6-class 意图映射
_C4_TO_C6 = {0: 0, 1: 1, 2: 2, 3: 4}


class DronePredictor:
    """无人机轨迹预测器 — 双模型速度硬切换。

    speed < S_THRESHOLD → low model (UAV-Flow, 6-class, 5Hz, 0-3 m/s domain)
    speed >= S_THRESHOLD → high model (SimCruise, 4-class, 1Hz, 8-28 m/s domain)
    """

    def __init__(self, threshold=S_THRESHOLD, device=None):
        self.threshold = threshold
        self.device = torch.device(device) if device else _DEVICE

        self.low = self._load('low_speed_6class.pth', 6)
        self.high = self._load('high_speed_4class.pth', 4)

        self._stats = {'low': 0, 'high': 0, 'n': 0}

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

    @staticmethod
    def compute_z_trend(hist, window=None):
        """分析Z轴历史趋势, 判断无人机是否真的在下降。

        Args:
            hist: (B, 20, 6) 历史轨迹
            window: 分析窗口帧数 (默认用全部20帧)

        Returns:
            dict with:
                z_velocity: (B,) 最近window帧的Z轴平均速度 (m/s per frame), 负值=下降
                z_accel:    (B,) Z轴加速度 (速度的变化率)
                z_trend:    (B,) 线性回归斜率 (m/s²), 负值=加速下降
                z_variance: (B,) Z位置的方差, 高值=波动大
                is_descending: (B,) bool, 是否确认在下降
                descent_strength: (B,) [0,1] 下降确信度
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
        # z_vel_mean = -0.03 → strength≈0, z_vel_mean = -0.5 → strength≈1
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
        """预测未来轨迹。

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

        # 分别预测 (两个模型都运行, 用于软融合)
        out_low = self.low(hist, force_predict=True)
        out_high = self.high(hist, force_predict=True)

        # Z轴纠正: 三段式过渡 + Z历史趋势感知
        # descend_prob < 0.05 → 确信非下降, 强压制 (Z→5%)
        # 0.05 <= descend_prob < 0.20 → 不确定区, 弱压制 (Z→30%)
        # descend_prob >= 0.20 → 确信下降, 不压制 (Z→100%)
        # Z历史趋势: 如果Z确实在下降 → 减弱压制 (模型判断更可信)
        if Z_CORRECTION_ENABLED:
            high_intent_prob = torch.softmax(out_high['intent_logits'], dim=-1)
            descend_prob = high_intent_prob[:, 3]               # class 3 = DESCEND

            # 三段式dampen factor
            dampen = torch.full_like(descend_prob, 1.0)         # default: no dampening
            strong_mask = descend_prob < Z_DESCEND_THRESHOLD_LOW
            weak_mask = (descend_prob >= Z_DESCEND_THRESHOLD_LOW) & (descend_prob < Z_DESCEND_THRESHOLD_HIGH)

            if strong_mask.any():
                dampen[strong_mask] = Z_DAMPEN_STRONG           # 5% → 强压制
            if weak_mask.any():
                # 线性过渡: 在[0.05, 0.20]区间内dampen从0.30线性过渡到1.0
                t = (descend_prob[weak_mask] - Z_DESCEND_THRESHOLD_LOW) / (Z_DESCEND_THRESHOLD_HIGH - Z_DESCEND_THRESHOLD_LOW)
                dampen[weak_mask] = Z_DAMPEN_WEAK + (1.0 - Z_DAMPEN_WEAK) * t

            # 多信号融合: Z趋势 + 模型不确定性 + 预测Z量级
            # 每个信号独立判断"是否更像真下降", 满足则减弱压制
            if Z_TREND_ENABLED or Z_UNCERTAINTY_ENABLED or Z_MAGNITUDE_ENABLED:
                total_boost = torch.zeros_like(dampen)

                # 信号1: Z历史趋势
                if Z_TREND_ENABLED:
                    z_info = self.compute_z_trend(hist, window=Z_TREND_WINDOW)
                    total_boost = total_boost + torch.where(
                        z_info['is_descending'],
                        torch.full_like(dampen, Z_TREND_DAMPEN_BOOST),
                        torch.zeros_like(dampen)
                    )

                # 信号2: 模型Z轴不确定性 (低不确定性 → 模型自信 → 减弱压制)
                if Z_UNCERTAINTY_ENABLED:
                    z_logvar = out_high['uncertainty'][:, :, 2]  # (B, pred_len) Z axis
                    z_mean_logvar = z_logvar.mean(dim=1)  # (B,) average over pred steps
                    low_uncertainty = z_mean_logvar < Z_UNCERTAINTY_THRESH
                    total_boost = total_boost + torch.where(
                        low_uncertainty,
                        torch.full_like(dampen, Z_UNCERTAINTY_BOOST),
                        torch.zeros_like(dampen)
                    )

                # 信号3: 预测Z位移量级 (预测大负位移 → 更像真下降)
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

            # ── LOW Z轴纠正 (DESCEND=class 4, 6-class) ──
            # LOW 模型同样受益于Z纠正: 低速无人机在DESCEND时也有Z漂移问题
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

            # 多信号融合 for LOW (reuse same thresholds, LOW drones are slower so signals are more conservative)
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

        # 熵引导物理融合: LOW模型意图不确定时偏向物理外推
        # 高熵(=模型在STRAIGHT/TURN之间纠结)时, 物理外推保留转弯动量, 比混乱的anchor更可靠
        if ENTROPY_PHYSICS_ENABLED:
            low_intent_prob = torch.softmax(out_low['intent_logits'], dim=-1)
            low_entropy = -(low_intent_prob * torch.log(low_intent_prob + 1e-8)).sum(dim=-1)
            physics = out_low.get('physics_trajectory', None)
            if physics is not None:
                alpha = torch.clamp((low_entropy - ENTROPY_THRESHOLD) / 0.6, 0.0, ENTROPY_MAX_BLEND)
                if alpha.max() > 0:
                    blend = alpha.view(-1, 1, 1)
                    out_low['predictions'] = (1.0 - blend) * out_low['predictions'] + blend * physics

        # 软融合: sigmoid 平滑过渡替代硬切换
        # α = sigmoid((speed - threshold) / temperature)
        # 过渡区外: speed < threshold-width → α=0 (纯LOW), speed > threshold+width → α=1 (纯HIGH)
        if SOFT_FUSION_ENABLED:
            raw_alpha = torch.sigmoid((speed - self.threshold) / FUSION_TEMPERATURE)  # (B,)
            # 过渡区门控: 仅在 [threshold - half_width, threshold + half_width] 内混合
            in_transition = (speed > self.threshold - FUSION_HALF_WIDTH) & (speed < self.threshold + FUSION_HALF_WIDTH)
            alpha = torch.where(in_transition, raw_alpha, (speed >= self.threshold).float())
            alpha_3d = alpha.view(-1, 1, 1)
            alpha_2d = alpha.view(-1, 1)
            predictions = (1.0 - alpha_3d) * out_low['predictions'] + alpha_3d * out_high['predictions']

            # 意图也软融合
            intent_high_6 = torch.full((hist.shape[0], 6), float('-inf'),
                                       device=self.device, dtype=out_high['intent_logits'].dtype)
            for c4, c6 in _C4_TO_C6.items():
                intent_high_6[:, c6] = out_high['intent_logits'][:, c4]
            intent = (1.0 - alpha_2d) * out_low['intent_logits'] + alpha_2d * intent_high_6

            # 统计 (用 α>0.5 作为路由标签)
            use_high = alpha > 0.5
            self._stats['n'] += hist.shape[0]
            self._stats['low'] += (alpha <= 0.5).sum().item()
            self._stats['high'] += (alpha > 0.5).sum().item()
            route_list = ['HIGH' if h else 'LOW' for h in use_high.cpu().tolist()]
        else:
            # 原始硬切换 (向后兼容)
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
        """Enable per-drone LoRA online adaptation on the low-speed model."""
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

            # Z轴纠正: 三段式过渡 + Z历史趋势感知
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

                # 多信号融合: Z趋势 + 模型不确定性 + 预测Z量级
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

                # ── LOW Z轴纠正 (DESCEND=class 4 in 6-class) ──
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

            # 熵引导物理融合
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

            # 软融合 or 硬切换
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

    # 低速
    x_low = torch.randn(4, 20, 6)
    x_low[:, :, 3:6] *= 0.5
    out = p.predict(x_low)
    routes = out['route']
    print(f'Low speed (~0.5 m/s): route={[r for r in routes]}')

    # 高速
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
