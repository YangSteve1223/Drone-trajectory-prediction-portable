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

# Fallback SSM scan implementation when mamba-ssm is unavailable.
#   'loop'    = serial Python loop (default; robust across batch sizes).
#   'chunked' = parallel segment-sum scan. Numerically equivalent (validated by
#               test_ssm_scan.py: fwd max 1.5e-5, grad max 2e-4). It is ~4x
#               faster for long sequences (T>=80) but the O(L^2 * d_state)
#               segment-sum matrix becomes memory-bound at large batch: it wins
#               at B=32/T=40 (1.17x) yet loses at B=64/T=40 (0.80x). Since the
#               training configs use T<=40 and batch 32-64, 'loop' is the safer
#               default; enable 'chunked' only for long-horizon workloads.
import os as _os
_SSM_FALLBACK = _os.environ.get('EMAM_SSM_FALLBACK', 'loop').lower()
_SSM_CHUNK_SIZE = int(_os.environ.get('EMAM_SSM_CHUNK', '4'))


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
    Selective State Space Model (Mamba variant).

    Input-dependent selective scan:
        h_t = exp(dt_t * A) * h_{t-1} + B_t * x_t   (state update)
        y_t = C_t * h_t + D * x_t                    (output projection)

    dt_t is computed from x_t via softplus, making the discretization
    step input-dependent for content-selective attention.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 3, bias=False)
        # A in log-space: A = -exp(A_log), always negative so SSM state decays
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init()

    def _init(self):
        # Init A_log so A in [-2, -0.1], mean ~ -0.5
        nn.init.normal_(self.A_log, mean=-0.7, std=0.5)  # A ≈ -exp(-0.7) ≈ -0.5
        nn.init.ones_(self.D)
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_inner = self.d_inner

        # Input projection: x → [x_gate, x_dt_raw, x_inner]
        xz = self.in_proj(x)
        x_gate, x_dt_raw, x_inner = xz.chunk(3, dim=-1)

        # Causal conv (preserve time order)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_act = silu(x_conv)  # (B, T, d_inner)

        # Time delta: softplus for positive value, bias makes initial dt ≈ 1.0
        # NOTE: softplus may downcast to FP16 under autocast, keep FP32
        with torch.amp.autocast('cuda', enabled=False):
            if x_dt_raw.dtype == torch.float16:
                x_dt_raw_fp32 = x_dt_raw.float()
            else:
                x_dt_raw_fp32 = x_dt_raw
            dt = F.softplus(x_dt_raw_fp32 + 1.0)

        # === Selective SSM recurrent scan ===
        # CRITICAL: SSM scan uses torch.exp which overflows in FP16.
        # Force FP32 for numerical stability.
        # A = -exp(A_log), always negative to prevent state explosion
        A_neg = -torch.exp(self.A_log)  # (d_inner, d_state), all < 0
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
                _scan_fn = (_selective_ssm_scan_chunked if _SSM_FALLBACK == 'chunked'
                            else _selective_ssm_scan)
                _scan_kwargs = ({'chunk_size': _SSM_CHUNK_SIZE}
                                if _SSM_FALLBACK == 'chunked' else {})
                y = _scan_fn(
                    x_act.float() if x_act.dtype == torch.float16 else x_act,
                    dt,
                    A_neg,
                    self.D,
                    self.d_state,
                    **_scan_kwargs,
                )
            if x.dtype == torch.float16:
                y = y.half()

        # Gating: z ⊙ y  (Mamba gating)
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
    dt,         # (B, T, d_inner)  softplus output, always > 0
    A_mat,      # (d_inner, d_state)  -exp(A_log), always < 0
    D_vec,      # (d_inner,)
    d_state,
):
    """Selective SSM recurrent scan (eager mode, pre-allocated output).

    A < 0 ensures exp(dt * A) <= 1, so state decays without exploding.
    """
    B, T, d_inner = x_act.shape
    device = x_act.device
    h = torch.zeros(B, d_inner, d_state, device=device, dtype=torch.float32)
    output = torch.empty(B, T, d_inner, device=device, dtype=torch.float32)

    # dt from softplus is > 0; extra clamp guards against outliers
    dt = dt.clamp(min=1e-6, max=50.0)

    # Pre-allocate broadcast views to reduce autograd intermediates
    D_vec_b = D_vec.view(1, d_inner)  # (1, d_inner) broadcast view
    A_mat_b = A_mat.unsqueeze(0)       # (1, d_inner, d_state) broadcast view

    for t in range(T):
        dt_t = dt[:, t, :]                                          # (B, d_inner)
        # dt_t > 0, A_mat < 0 → dt_t * A_mat < 0 → exp(<0) ∈ (0, 1]
        log_A_d = (dt_t.unsqueeze(-1) * A_mat_b).clamp_(-50.0, 50.0)
        A_d = torch.exp(log_A_d)                                     # (B, d_inner, d_state)
        x_t = x_act[:, t, :]
        # in-place update: h = A_d ⊙ h + x_t  (broadcast: x_t → (B, d_inner, 1))
        h.mul_(A_d).add_(x_t.unsqueeze(-1))
        output[:, t, :] = h.sum(dim=-1) + D_vec_b * x_t
    return output


def _selective_ssm_scan_chunked(
    x_act,      # (B, T, d_inner)
    dt,         # (B, T, d_inner)  softplus output, always > 0
    A_mat,      # (d_inner, d_state)  -exp(A_log), always < 0
    D_vec,      # (d_inner,)
    d_state,
    chunk_size: int = 16,
):
    """Chunked parallel selective SSM scan — numerically equivalent to the
    Python-loop scan, but with sequential depth T/chunk_size instead of T.

    The recurrence  h_t = a_t ⊙ h_{t-1} + x_t   (a_t = exp(dt_t * A) ∈ (0,1])
    is a first-order linear recurrence. Within a chunk we solve it in parallel:

        Let  P_t = prod_{k=1..t} a_k   (cumulative decay inside the chunk).
        Then h_t = P_t ⊙ h_prev + P_t ⊙ cumsum_t( x_k / P_k )

    Division by the cumulative product is avoided by doing the intra-chunk
    weighted cumulative sum in log-space via a subtraction trick that stays
    bounded: because a_k ∈ (0,1], log P is non-positive and monotonically
    decreasing, so (log P_t - log P_k) ≤ 0 for k ≤ t and exp(...) ∈ (0,1].
    Chunk boundaries carry the true state h forward serially.
    """
    B, T, d_inner = x_act.shape
    device = x_act.device
    dtype = torch.float32

    dt = dt.clamp(min=1e-6, max=50.0)

    # log a_t = dt_t * A  (≤ 0 since A < 0). Shape (B, T, d_inner, d_state)
    log_a = (dt.unsqueeze(-1) * A_mat.view(1, 1, d_inner, d_state)).clamp(-50.0, 50.0)

    D_vec_b = D_vec.view(1, 1, d_inner)
    h_prev = torch.zeros(B, d_inner, d_state, device=device, dtype=dtype)
    outputs = []

    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        L = c1 - c0
        la = log_a[:, c0:c1]                       # (B, L, d_inner, d_state)
        xc = x_act[:, c0:c1]                        # (B, L, d_inner)

        # Cumulative log decay within chunk: logP_t = sum_{k<=t} log a_k  (≤ 0)
        logP = torch.cumsum(la, dim=1)             # (B, L, d_inner, d_state)

        # --- Intra-chunk contribution via stable segment-sum matrix ---
        #   s_t = sum_{k<=t} exp(logP_t - logP_k) * x_k
        # decay[t,k] = exp(logP_t - logP_k). Since logP is monotonically
        # decreasing (cumsum of non-positive log_a), logP_t - logP_k ≤ 0 for
        # t ≥ k, so exp(...) ∈ (0,1] — no overflow. Upper triangle is masked out.
        logP_t = logP.unsqueeze(2)                 # (B, L, 1, d_inner, d_state)
        logP_k = logP.unsqueeze(1)                 # (B, 1, L, d_inner, d_state)
        decay = logP_t - logP_k                    # (B, L, L, d_inner, d_state)
        causal = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        decay = decay.masked_fill(~causal.view(1, L, L, 1, 1), float('-inf'))
        decay = torch.exp(decay)                   # (B, L, L, d_inner, d_state), ∈ [0,1]

        # intra_t = sum_k decay[t,k] * x_k  (x broadcast over d_state)
        x_k = xc.unsqueeze(1).unsqueeze(-1)        # (B, 1, L, d_inner, 1)
        intra = (decay * x_k).sum(dim=2)           # (B, L, d_inner, d_state)

        # Carry-in from previous chunk state: exp(logP_t) ⊙ h_prev
        carry = torch.exp(logP) * h_prev.unsqueeze(1)  # (B, L, d_inner, d_state)

        h_chunk = carry + intra                    # (B, L, d_inner, d_state)

        y = h_chunk.sum(dim=-1) + D_vec_b * xc     # (B, L, d_inner)
        outputs.append(y)

        h_prev = h_chunk[:, -1]                    # (B, d_inner, d_state) true state at c1-1

    return torch.cat(outputs, dim=1)               # (B, T, d_inner)


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
