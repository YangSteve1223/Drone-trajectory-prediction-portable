"""
Intent-Aware Dynamic Temporal Pyramid (IA-DTP) — Enhanced Version.

PPT Slide 7: "在统一框架下实现多尺度残差特征提取与隐式飞行意图感知的协同建模。
通过自适应地捕获高频机动下的微观状态残差，IA-DTP 能够生成融合了多尺度状态变化
与宏观意图先验的全局上下文向量。"

Architecture (Enhanced):
  Encoded Features (B, T, d_model)
      │
      ├─ MultiScaleTemporalPyramid ──────────────────────┐
      │   ├─ Scale 0: origin (T)                          │
      │   ├─ Scale 1: stride-2 down (T/2)                 │
      │   └─ Scale 2: stride-4 down (T/4)                 │
      │       ↓                                            │
      │   Per-Scale Residual Conv1D + HF Extraction        │
      │       ↓                                            │
      ├─ CrossScaleAttention ─────────────────────────────┤
      │   ├─ Coarse→Fine: coarse features guide fine       │
      │   └─ Fine→Coarse: fine features enrich coarse      │
      │       ↓                                            │
      ├─ AdaptiveFusion (learned per-scale weights) ──────┤
      │       ↓                                            │
      ├─ IntentHead ─→ intent_logits (B, 5)              │
      │       ↓                                            │
      ├─ AnchorGenerator ─→ global_anchor (B, 1, d_model) │
      │       ↓                                            │
      └─ FeatureEnhancement ─→ enhanced_features (B,T,d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Intent type enumeration
class IntentType:
    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    ASCEND = 3
    DESCEND = 4


NUM_INTENT_CLASSES = 5


# ============================================================================
# Sub-module 1: Multi-Scale Temporal Pyramid
# ============================================================================

class MultiScaleTemporalPyramid(nn.Module):
    """
    多尺度时序金字塔: 在不同时间分辨率下提取特征.

    Scales:
      - Scale 0: 原始分辨率 T → 捕捉微观机动细节
      - Scale 1: stride-2 降采样 T/2 → 捕捉中观机动模式
      - Scale 2: stride-4 降采样 T/4 → 捕捉宏观意图趋势

    Args:
        d_model: 特征维度
        num_scales: 金字塔层数 (default: 3)
        kernel_size: 每层卷积核大小
    """
    def __init__(self, d_model: int = 256, num_scales: int = 3, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales

        # 每个尺度的降采样卷积 (stride=2)
        self.down_convs = nn.ModuleList()
        for s in range(num_scales - 1):
            self.down_convs.append(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=1)
            )

        # 每个尺度的处理卷积 + 残差块
        self.scale_convs = nn.ModuleList()
        self.scale_residuals = nn.ModuleList()
        for s in range(num_scales):
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(8, d_model),
                    nn.SiLU(),
                    nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                )
            )
            # 高频残差提取 (对原始特征做差分, 捕获机动突变)
            self.scale_residuals.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2),
                    nn.SiLU(),
                    nn.Conv1d(d_model // 2, d_model, kernel_size=1),
                )
            )

        # 上采样投影 (恢复原始分辨率)
        self.up_projs = nn.ModuleList()
        for s in range(1, num_scales):
            self.up_projs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2 ** s, mode='linear', align_corners=False),
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.GroupNorm(8, d_model),
                    nn.SiLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (B, T, d_model) 编码特征
        Returns:
            fused: (B, T, d_model) 多尺度融合特征
            scale_features: list of (B, d_model) per-scale pooled features
        """
        B, T, D = x.shape
        x_t = x.transpose(1, 2)  # (B, d_model, T)

        # 构建金字塔
        pyramid = [x_t]  # Scale 0: 原始分辨率
        for down_conv in self.down_convs:
            pyramid.append(down_conv(pyramid[-1]))

        # 每层处理 + 高频残差提取
        processed = []
        scale_pooled = []
        for s, (conv, res_block) in enumerate(zip(self.scale_convs, self.scale_residuals)):
            feat = pyramid[s]  # (B, d_model, T_s)
            # 主通路
            main = conv(feat)
            # 高频残差: 对相邻帧差分, 放大机动突变信号
            feat_diff = feat[:, :, 1:] - feat[:, :, :-1]  # (B, d_model, T_s-1)
            feat_diff_padded = F.pad(feat_diff, (1, 0), mode='replicate')
            hf_residual = res_block(feat_diff_padded)
            # 残差融合
            out = main + hf_residual
            out = out + feat  # skip connection
            processed.append(out)
            # 池化到固定维度
            scale_pooled.append(out.mean(dim=-1))  # (B, d_model)

        # 上采样恢复到原始 T
        upsampled = [processed[0].transpose(1, 2)]  # Scale 0: (B, T, d_model)
        for s in range(1, self.num_scales):
            up_proj = self.up_projs[s - 1]
            feat_up = up_proj(processed[s])  # (B, d_model, T)
            # 截断或填充以匹配原始 T
            if feat_up.shape[-1] > T:
                feat_up = feat_up[:, :, :T]
            elif feat_up.shape[-1] < T:
                feat_up = F.pad(feat_up, (0, T - feat_up.shape[-1]), mode='replicate')
            upsampled.append(feat_up.transpose(1, 2))  # (B, T, d_model)

        return upsampled, scale_pooled


# ============================================================================
# Sub-module 2: Cross-Scale Attention
# ============================================================================

class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力: 粗尺度引导细尺度, 细尺度丰富粗尺度.

    双向交互:
      - Coarse→Fine: queries from fine, keys/values from coarse
      - Fine→Coarse: queries from coarse, keys/values from fine
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _attention(self, q, k, v) -> torch.Tensor:
        """Scaled dot-product attention."""
        B, H, N, D = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return attn @ v

    def forward(self, multi_scale_features: list) -> torch.Tensor:
        """
        Args:
            multi_scale_features: list of (B, T, d_model), ordered fine→coarse
        Returns:
            fused: (B, T, d_model) cross-scale attended features
        """
        # 取最细和最粗尺度做交叉注意力
        fine = multi_scale_features[0]   # (B, T, d_model)
        coarse = multi_scale_features[-1]  # (B, T, d_model)

        B, T_f, D = fine.shape
        _, T_c, _ = coarse.shape

        # 如果时序长度不同 (因上采样截断), 对齐
        if T_f != T_c:
            if T_c > T_f:
                coarse = coarse[:, :T_f, :]
            else:
                coarse = F.pad(coarse, (0, 0, 0, T_f - T_c), mode='replicate')

        # Reshape for multi-head
        def reshape(x): return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q_fine = reshape(self.q_proj(fine))
        k_fine = reshape(self.k_proj(fine))
        v_fine = reshape(self.v_proj(fine))

        q_coarse = reshape(self.q_proj(coarse))
        k_coarse = reshape(self.k_proj(coarse))
        v_coarse = reshape(self.v_proj(coarse))

        # Coarse→Fine: fine queries coarse (coarse guides fine)
        cf_attn = self._attention(q_fine, k_coarse, v_coarse)
        # Fine→Coarse: coarse queries fine (fine details enrich coarse)
        fc_attn = self._attention(q_coarse, k_fine, v_fine)

        # 融合
        cf_out = cf_attn.transpose(1, 2).reshape(B, T_f, D)
        fc_out = fc_attn.transpose(1, 2).reshape(B, T_c, D)

        if T_f != T_c:
            if T_c > T_f:
                fc_out = fc_out[:, :T_f, :]
            else:
                fc_out = F.pad(fc_out, (0, 0, 0, T_f - T_c), mode='replicate')

        fused = self.out_proj(cf_out + fc_out)
        fused = self.norm(fused + fine)  # residual + norm

        return fused


# ============================================================================
# Sub-module 3: Adaptive Fusion
# ============================================================================

class AdaptiveFusion(nn.Module):
    """
    自适应多尺度融合: 根据轨迹动态特性学习每层权重.

    直觉: 平稳飞行时细尺度更重要, 急剧机动时粗尺度 (宏观趋势) 更重要.
    """
    def __init__(self, d_model: int = 256, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # 动态权重预测器
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_scales),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, scale_features: list, scale_pooled: list
    ) -> torch.Tensor:
        """
        Args:
            scale_features: list of (B, T, d_model)
            scale_pooled: list of (B, d_model)
        Returns:
            fused: (B, T, d_model) 自适应加权融合特征
        """
        # 拼接所有尺度的池化特征来预测权重
        pooled_cat = torch.cat(scale_pooled, dim=-1)  # (B, d_model * num_scales)
        weights = self.weight_predictor(pooled_cat)    # (B, num_scales)

        # 加权融合
        fused = torch.zeros_like(scale_features[0])
        for s in range(self.num_scales):
            w = weights[:, s:s+1].unsqueeze(-1)  # (B, 1, 1)
            fused = fused + w * scale_features[s]

        return fused


# ============================================================================
# Main Module: Enhanced IA-DTP
# ============================================================================

class IntentAwareDTP(nn.Module):
    """
    Intent-Aware Dynamic Temporal Pyramid (Enhanced).

    三阶段流程:
      1. Multi-scale temporal pyramid → multi-scale features
      2. Cross-scale attention → coarse↔fine bidirectional interaction
      3. Adaptive fusion + intent classification + anchor generation

    Args:
        d_model: 特征维度 (from EMam-SE)
        num_classes: 意图类别数
        hidden_dim: 意图分类头隐层维度
        num_scales: 时序金字塔层数
        num_heads: 跨尺度注意力头数
    """
    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = NUM_INTENT_CLASSES,
        hidden_dim: int = 128,
        num_scales: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # 多尺度时序金字塔
        self.pyramid = MultiScaleTemporalPyramid(
            d_model=d_model, num_scales=num_scales
        )

        # 跨尺度注意力
        self.cross_attn = CrossScaleAttention(
            d_model=d_model, num_heads=num_heads
        )

        # 自适应融合
        self.fusion = AdaptiveFusion(
            d_model=d_model, num_scales=num_scales
        )

        # 意图分类头
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 全局锚点生成器 (融合多尺度上下文 + 意图信息)
        self.anchor_proj = nn.Sequential(
            nn.Linear(d_model * 2 + num_classes, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 特征增强: 将意图 + 锚点信息注入回时序特征
        self.enhance = nn.Sequential(
            nn.Linear(d_model + num_classes + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        encoded_features: torch.Tensor,
        historical_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_features: (B, T, d_model) EMam-SE 编码特征
            historical_trajectory: (B, T, 6) 原始轨迹 (可选, 用于运动学分析)
        Returns:
            global_anchor: (B, 1, d_model)
            intent_logits: (B, num_classes)
            intent_weights: (B, num_classes)
            enhanced_features: (B, T, d_model)
        """
        B, T, D = encoded_features.shape

        # ================================================================
        # Stage 1: 多尺度时序金字塔
        # ================================================================
        scale_features, scale_pooled = self.pyramid(encoded_features)
        # scale_features: list of (B, T, d_model), [fine, mid, coarse]
        # scale_pooled:  list of (B, d_model)

        # ================================================================
        # Stage 2: 跨尺度注意力 (双向交互)
        # ================================================================
        cross_attended = self.cross_attn(scale_features)  # (B, T, d_model)

        # ================================================================
        # Stage 3: 自适应融合
        # ================================================================
        fused_features = self.fusion(scale_features, scale_pooled)  # (B, T, d_model)

        # 将跨尺度注意力结果与自适应融合结果结合
        combined = cross_attended + fused_features  # (B, T, d_model)

        # ================================================================
        # Stage 4: 意图分类
        # ================================================================
        # 全局池化 + 分类
        global_feat = combined.mean(dim=1)  # (B, d_model)
        intent_logits = self.intent_head(global_feat)  # (B, num_classes)
        intent_probs = F.softmax(intent_logits, dim=-1)

        # ================================================================
        # Stage 5: 全局锚点生成 (多尺度上下文 + 意图先验)
        # ================================================================
        # 粗尺度全局信息 (宏观趋势)
        coarse_global = scale_pooled[-1]  # (B, d_model)

        # 拼接: 精细全局 + 粗糙全局 + 意图分布
        anchor_input = torch.cat([
            global_feat,          # (B, d_model) — fine-grained
            coarse_global,        # (B, d_model) — coarse trend
            intent_probs,         # (B, num_classes) — intent prior
        ], dim=-1)  # (B, 2*d_model + num_classes)

        global_anchor = self.anchor_proj(anchor_input).unsqueeze(1)  # (B, 1, d_model)

        # ================================================================
        # Stage 6: 时序特征增强 (注入意图 + 锚点)
        # ================================================================
        intent_tiled = intent_probs.unsqueeze(1).expand(-1, T, -1)    # (B, T, num_classes)
        anchor_tiled = global_anchor.expand(-1, T, -1)                # (B, T, d_model)
        enhance_input = torch.cat([combined, intent_tiled, anchor_tiled], dim=-1)
        enhanced_features = self.enhance(enhance_input)  # (B, T, d_model)

        return {
            'global_anchor': global_anchor,
            'intent_logits': intent_logits,
            'intent_weights': intent_probs,
            'enhanced_features': enhanced_features,
        }
