"""
Enhanced Mamba with Squeeze-and-Excitation (EMam-SE)
Input: Trajectory Sequence → Spatiotemporal Encoded Features
1. Linear Projection: (B,T,6) → (B,T,D)
2. LayerNorm
3. 1D Causal Conv (kernel=3)
4. EnhancedMambaBlock × n_layers
5. MultiScaleDWConv (3/5/7)
6. SE Channel Attention
7. Gate + LayerNorm + Dropout
8. Output Projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

# Attempt to import CUDA-accelerated selective scan from mamba-ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit): x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu(x)


class GRLU(nn.Module):
    """Gated Recurrent Linear Unit"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba variant)
    
    实现输入依赖的选择性状态空间扫描:
        h_t = exp(dt_t * A) * h_{t-1} + B_t * x_t   (状态更新)
        y_t = C_t * h_t + D * x_t                    (输出投影)
    
    其中 dt_t 由输入 x_t 经 softplus 计算得到，使 A 矩阵离散化参数
    每步不同，从而赋予模型对序列内容的选择性注意能力。
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 3, bias=False)
        # A 使用 log-space 参数化: A = -exp(A_log)，保证始终为负
        # 负 A 使 SSM 状态自然衰减 (exp(dt * (-|A|)) < 1)，防止递归爆炸
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init()

    def _init(self):
        # A_log 初始化为使 A 在 [-2, -0.1] 范围内 (不同通道不同衰减速率)
        # 均值为 -0.5: A_log ~ N(ln(0.5), small_std)
        nn.init.normal_(self.A_log, mean=-0.7, std=0.5)  # A ≈ -exp(-0.7) ≈ -0.5
        nn.init.ones_(self.D)
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_inner = self.d_inner

        # 输入投影: x → [x_gate, x_dt_raw, x_inner]
        xz = self.in_proj(x)
        x_gate, x_dt_raw, x_inner = xz.chunk(3, dim=-1)

        # 因果卷积 (保持时间顺序)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_act = silu(x_conv)  # (B, T, d_inner)

        # 时间增量: softplus 确保正值, 加偏置使初始 dt ≈ 1.0
        # NOTE: 在 autocast 下 softplus 可能降为 FP16, 需要保持 FP32
        with torch.amp.autocast('cuda', enabled=False):
            if x_dt_raw.dtype == torch.float16:
                x_dt_raw_fp32 = x_dt_raw.float()
            else:
                x_dt_raw_fp32 = x_dt_raw
            dt = F.softplus(x_dt_raw_fp32 + 1.0)

        # === 选择性 SSM 递归扫描 ===
        # CRITICAL: SSM scan uses torch.exp which overflows in FP16.
        # Force FP32 for numerical stability.
        # A = -exp(A_log) 保证始终为负，防止递归状态爆炸
        A_neg = -torch.exp(self.A_log)  # (d_inner, d_state), 所有值 < 0
        if _HAS_MAMBA_SSM and x.is_cuda:
            y = selective_scan_fn(
                x_act, dt, A_neg, self.D.unsqueeze(-1),
                delta_softplus=False,
            )
        else:
            if x.is_cuda and T > 50 and not _HAS_MAMBA_SSM:
                warnings.warn(
                    "CUDA available but mamba-ssm not installed. "
                    "Falling back to Python-loop SSM scan (slow for long sequences). "
                    "Install with: pip install mamba-ssm",
                    UserWarning, stacklevel=2
                )
            with torch.amp.autocast('cuda', enabled=False):
                y = _selective_ssm_scan(
                    x_act.float() if x_act.dtype == torch.float16 else x_act,
                    dt,
                    A_neg,
                    self.D,
                    self.d_state,
                )
            if x.dtype == torch.float16:
                y = y.half()

        # 门控: z ⊙ y  (Mamba 的 gating 机制)
        gate = torch.sigmoid(x_gate)
        y = y * gate

        output = self.out_proj(y)
        return output


class MambaBlock(nn.Module):
    """Mamba Block with Gated Linear Unit"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.grlu = GRLU(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.grlu(x)
        x = self.dropout(x)
        return residual + x


class MultiScaleDWConv1D(nn.Module):
    """
    Multi-Scale Depthwise Conv (kernel=3,5,7)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv_3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv_5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.fusion = nn.Conv1d(dim * 3, dim, kernel_size=1)
        self._init()

    def _init(self):
        for conv in [self.conv_3, self.conv_5, self.conv_7]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        o3 = self.conv_3(x)
        o5 = self.conv_5(x)
        o7 = self.conv_7(x)
        out = torch.cat([o3, o5, o7], dim=1)
        out = self.fusion(out)
        return out.transpose(1, 2)


def _selective_ssm_scan(
    x_act,      # (B, T, d_inner)
    dt,         # (B, T, d_inner)  softplus 输出, 始终 > 0
    A_mat,      # (d_inner, d_state)  -exp(A_log), 始终 < 0
    D_vec,      # (d_inner,)
    d_state,
):
    """选择性 SSM 递归扫描 (eager mode, pre-allocated output).

    A < 0 保证 exp(dt * A) ≤ 1，状态自然衰减，不会爆炸。
    """
    B, T, d_inner = x_act.shape
    device = x_act.device
    h = torch.zeros(B, d_inner, d_state, device=device, dtype=torch.float32)
    output = torch.empty(B, T, d_inner, device=device, dtype=torch.float32)

    # dt 由 softplus 输出，理论上 > 0；额外 clamp 防止异常值
    dt = dt.clamp(min=1e-6, max=50.0)

    # 预分配中间 tensor，避免每次循环分配 (减少 autograd 跟踪的中间节点)
    D_vec_b = D_vec.view(1, d_inner)  # (1, d_inner) 广播视图
    A_mat_b = A_mat.unsqueeze(0)       # (1, d_inner, d_state) 广播视图

    for t in range(T):
        dt_t = dt[:, t, :]                                          # (B, d_inner)
        # dt_t > 0, A_mat < 0 → dt_t * A_mat < 0 → exp(<0) ∈ (0, 1]
        log_A_d = (dt_t.unsqueeze(-1) * A_mat_b).clamp_(-50.0, 50.0)
        A_d = torch.exp(log_A_d)                                     # (B, d_inner, d_state)
        x_t = x_act[:, t, :]
        # in-place 更新: h = A_d ⊙ h + x_t   (广播: x_t → (B, d_inner, 1))
        h.mul_(A_d).add_(x_t.unsqueeze(-1))
        output[:, t, :] = h.sum(dim=-1) + D_vec_b * x_t
    return output


class SEChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            SiLU(),
            nn.Linear(dim // reduction, dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        w = self.squeeze(x_t).squeeze(-1)
        w = self.excitation(w)
        return x * w.unsqueeze(1)


class EnhancedMambaSE(nn.Module):
    """
    Enhanced Mamba with SE (EMam-SE)
    1. Linear Projection: (B,T,6) → (B,T,D)
    2. LayerNorm
    3. Causal Conv (kernel=3)
    4. Mamba blocks + MultiScaleDWConv + SE
    5. Final gating + dropout
    6. Output Projection
    """
    def __init__(self, input_dim: int = 6, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.causal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.ms_dwconv = MultiScaleDWConv1D(d_model)
        self.se_attention = SEChannelAttention(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.output_proj = nn.Linear(d_model, d_model)
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)
        x_conv = self.causal_conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = self.act(x_conv)
        x = x + x_conv

        for block in self.mamba_blocks:
            x = block(x)
            if hasattr(self, 'ms_dwconv'):
                x_ms = self.ms_dwconv(x)
                x = x + x_ms
                x = self.se_attention(x)

        gate = torch.sigmoid(self.gate_proj(x))
        x = self.norm(x)
        x = gate * x
        x = self.dropout(x)
        return self.output_proj(x)


class EMAMSEWithOutput(nn.Module):
    """EMam-SE with output pooling"""
    def __init__(self, input_dim: int = 6, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.core = EnhancedMambaSE(input_dim, d_model, d_state, d_conv, expand, n_layers, dropout)

    def forward(self, x):
        features = self.core(x)
        pooled = features.mean(dim=1)
        return features, pooled
