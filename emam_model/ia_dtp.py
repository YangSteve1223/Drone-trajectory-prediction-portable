"""Intent-Aware Dynamic Temporal Pyramid: multi-scale residual features + cross-scale attention
for implicit flight intent perception and anchor generation."""

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
    Multi-scale temporal pyramid: extract features at multiple time resolutions.

    Scales:
      - Scale 0: full resolution T (micro maneuver detail)
      - Scale 1: stride-2 downsample T/2 (meso maneuver pattern)
      - Scale 2: stride-4 downsample T/4 (macro intent trend)

    Args:
        d_model: feature dim
        num_scales: pyramid levels (default: 3)
        kernel_size: conv kernel size per level
    """
    def __init__(self, d_model: int = 256, num_scales: int = 3, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales

        # Downsampling conv per scale (stride=2)
        self.down_convs = nn.ModuleList()
        for s in range(num_scales - 1):
            self.down_convs.append(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=1)
            )

        # Per-scale processing conv + residual block
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
            # High-frequency residual (diff of raw features to catch maneuver bursts)
            self.scale_residuals.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2),
                    nn.SiLU(),
                    nn.Conv1d(d_model // 2, d_model, kernel_size=1),
                )
            )

        # Upsampling projection (restore original resolution)
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
            x: (B, T, d_model) encoded features
        Returns:
            fused: (B, T, d_model) multi-scale fused features
            scale_features: list of (B, d_model) per-scale pooled features
        """
        B, T, D = x.shape
        x_t = x.transpose(1, 2)  # (B, d_model, T)

        # Build pyramid
        pyramid = [x_t]  # Scale 0: full resolution
        for down_conv in self.down_convs:
            pyramid.append(down_conv(pyramid[-1]))

        # Per-level processing + high-frequency residual
        processed = []
        scale_pooled = []
        for s, (conv, res_block) in enumerate(zip(self.scale_convs, self.scale_residuals)):
            feat = pyramid[s]  # (B, d_model, T_s)
            # Main path
            main = conv(feat)
            # High-frequency residual: adjacent-frame diff amplifies maneuver bursts
            feat_diff = feat[:, :, 1:] - feat[:, :, :-1]  # (B, d_model, T_s-1)
            feat_diff_padded = F.pad(feat_diff, (1, 0), mode='replicate')
            hf_residual = res_block(feat_diff_padded)
            # Residual fusion
            out = main + hf_residual
            out = out + feat  # skip connection
            processed.append(out)
            # Pool to fixed dim
            scale_pooled.append(out.mean(dim=-1))  # (B, d_model)

        # Upsample back to original T
        upsampled = [processed[0].transpose(1, 2)]  # Scale 0: (B, T, d_model)
        for s in range(1, self.num_scales):
            up_proj = self.up_projs[s - 1]
            feat_up = up_proj(processed[s])  # (B, d_model, T)
            # Truncate or pad to match original T
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
    Cross-scale attention: coarse guides fine, fine enriches coarse.

    Bidirectional:
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
        # Use finest and coarsest scales for cross attention
        fine = multi_scale_features[0]   # (B, T, d_model)
        coarse = multi_scale_features[-1]  # (B, T, d_model)

        B, T_f, D = fine.shape
        _, T_c, _ = coarse.shape

        # Align time length if different (from upsample truncation)
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

        # Fuse
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
    Adaptive multi-scale fusion: learn per-scale weights from trajectory dynamics.

    Intuition: fine scale matters more in stable flight, coarse scale (macro trend)
    matters more in sharp maneuvers.
    """
    def __init__(self, d_model: int = 256, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # Dynamic weight predictor
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
            fused: (B, T, d_model) adaptively weighted fused features
        """
        # Concat all scale pooled features to predict weights
        pooled_cat = torch.cat(scale_pooled, dim=-1)  # (B, d_model * num_scales)
        weights = self.weight_predictor(pooled_cat)    # (B, num_scales)

        # Weighted fusion
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

    Three stages:
      1. Multi-scale temporal pyramid → multi-scale features
      2. Cross-scale attention → coarse↔fine bidirectional interaction
      3. Adaptive fusion + intent classification + anchor generation

    Args:
        d_model: feature dim (from EMam-SE)
        num_classes: number of intent classes
        hidden_dim: intent head hidden dim
        num_scales: pyramid levels
        num_heads: cross-scale attention heads
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

        # Multi-scale temporal pyramid
        self.pyramid = MultiScaleTemporalPyramid(
            d_model=d_model, num_scales=num_scales
        )

        # Cross-scale attention
        self.cross_attn = CrossScaleAttention(
            d_model=d_model, num_heads=num_heads
        )

        # Adaptive fusion
        self.fusion = AdaptiveFusion(
            d_model=d_model, num_scales=num_scales
        )

        # Intent classification head
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Global anchor generator (fuses multi-scale context + intent info)
        self.anchor_proj = nn.Sequential(
            nn.Linear(d_model * 2 + num_classes, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Feature enhancement: inject intent + anchor back into temporal features
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
            encoded_features: (B, T, d_model) EMam-SE encoded features
            historical_trajectory: (B, T, 6) raw trajectory (optional, for kinematics)
        Returns:
            global_anchor: (B, 1, d_model)
            intent_logits: (B, num_classes)
            intent_weights: (B, num_classes)
            enhanced_features: (B, T, d_model)
        """
        B, T, D = encoded_features.shape

        # ================================================================
        # Stage 1: multi-scale temporal pyramid
        # ================================================================
        scale_features, scale_pooled = self.pyramid(encoded_features)
        # scale_features: list of (B, T, d_model), [fine, mid, coarse]
        # scale_pooled:  list of (B, d_model)

        # ================================================================
        # Stage 2: cross-scale attention (bidirectional)
        # ================================================================
        cross_attended = self.cross_attn(scale_features)  # (B, T, d_model)

        # ================================================================
        # Stage 3: adaptive fusion
        # ================================================================
        fused_features = self.fusion(scale_features, scale_pooled)  # (B, T, d_model)

        # Combine cross-scale attention with adaptive fusion
        combined = cross_attended + fused_features  # (B, T, d_model)

        # ================================================================
        # Stage 4: intent classification
        # ================================================================
        # Global pooling + classify
        global_feat = combined.mean(dim=1)  # (B, d_model)
        intent_logits = self.intent_head(global_feat)  # (B, num_classes)
        intent_probs = F.softmax(intent_logits, dim=-1)

        # ================================================================
        # Stage 5: global anchor generation (multi-scale context + intent prior)
        # ================================================================
        # Coarse-scale global info (macro trend)
        coarse_global = scale_pooled[-1]  # (B, d_model)

        # Concat: fine global + coarse global + intent distribution
        anchor_input = torch.cat([
            global_feat,          # (B, d_model) — fine-grained
            coarse_global,        # (B, d_model) — coarse trend
            intent_probs,         # (B, num_classes) — intent prior
        ], dim=-1)  # (B, 2*d_model + num_classes)

        global_anchor = self.anchor_proj(anchor_input).unsqueeze(1)  # (B, 1, d_model)

        # ================================================================
        # Stage 6: temporal feature enhancement (inject intent + anchor)
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
