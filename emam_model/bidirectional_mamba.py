"""Bidirectional selective SSM encoder with gated forward/backward fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BidirectionalSelectiveSSM(nn.Module):
    """Bidirectional SSM with input-dependent B/C selective mechanism."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # === Input projection: x → [x_conv, x_dt] ===
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=True)

        # === SSM core parameters ===
        # A: state evolution matrix (d_inner, d_state)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        # D: skip-connection matrix (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # === Local causal convolution ===
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # === Output projection ===
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize SSM parameters."""
        # A matrix: negative values ensure state stability (exp(-|A|) ∈ (0,1))
        nn.init.xavier_uniform_(self.A)
        with torch.no_grad():
            self.A.copy_(-torch.abs(self.A))
        # D matrix: all ones
        nn.init.ones_(self.D)
        # Conv layer init
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1d.bias)

    def _ssm_scan(
        self,
        x_conv: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        SSM sequence scan.

        Args:
            x_conv: (B, T, d_inner) features after convolution
            reverse: whether to scan in reverse

        Returns:
            output: (B, T, d_inner)
        """
        B, T, d_inner = x_conv.shape
        
        if reverse:
            x_conv = torch.flip(x_conv, dims=[1])
        
        # === Selective parameter computation ===
        # dt: softplus ensures positive values
        dt = F.softplus(x_conv)  # (B, T, d_inner)

        # A discretization: A_dis[b,t,i,k] = exp(dt[b,t,i] * A[i,k])
        # Result shape: (B, T, d_inner, d_state)
        # Numerical stability: dt*A may cause exp explosion, clamp to [-50, 50]
        dt_A = torch.einsum('btd,dn->btdn', dt, self.A)
        dt_A = dt_A.clamp(min=-50.0, max=50.0)
        A_dis = torch.exp(dt_A)

        # Selective B: B_t = x_conv (used directly, no extra projection)
        # Selective C: C_t = x_conv (used directly, no extra projection)
        B_t = x_conv  # (B, T, d_inner)
        C_t = x_conv  # (B, T, d_inner)

        # === SSM scan ===
        # Initialize hidden state: h_0 = 0
        h = torch.zeros(B, self.d_inner, self.d_state,
                       device=x_conv.device, dtype=x_conv.dtype)

        outputs = []
        for t in range(T):
            A_t = A_dis[:, t]        # (B, d_inner, d_state)
            B_t_t = B_t[:, t]        # (B, d_inner)
            C_t_t = C_t[:, t]        # (B, d_inner)

            # h_term[b,i] = Σ_k A_t[b,i,k] * h[b,i,k]
            h_term = torch.einsum('bik,bik->bi', A_t, h)  # (B, d_inner)

            # Consistent with emam_se.py: h_new = h_term * A_t + B_t * C_t
            h_new = h_term.unsqueeze(-1) * A_t + B_t_t.unsqueeze(-1) * C_t_t.unsqueeze(-1)
            h_new = h_new.clamp(-10, 10)  # numerical stability

            # y_t[b,i] = Σ_k C_t[b,i] * h_new[b,i,k] + D[i] * C_t[b,i]
            y_t = torch.einsum('bi,bik->bi', C_t_t, h_new) + self.D * C_t_t
            
            outputs.append(y_t)
            h = h_new
        
        output = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        
        if reverse:
            output = torch.flip(output, dims=[1])
        
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional SSM forward pass.

        Args:
            x: (B, T, D)

        Returns:
            forward_out: (B, T, D) forward scan result
            backward_out: (B, T, D) backward scan result
        """
        B, T, D = x.shape

        # === Input projection ===
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_conv_raw, x_dt_raw = xz.chunk(2, dim=-1)

        # === Preprocessing ===
        # Combine x_conv_raw and x_dt_raw to compute dt
        dt_input = x_conv_raw * torch.sigmoid(x_dt_raw)

        # === Causal convolution ===
        x_conv = dt_input.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # truncate
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)  # SiLU activation

        # === Bidirectional SSM scan ===
        forward_out = self._ssm_scan(x_conv, reverse=False)
        backward_out = self._ssm_scan(x_conv, reverse=True)

        # === Output projection ===
        forward_out = self.out_proj(forward_out)
        backward_out = self.out_proj(backward_out)
        
        return forward_out, backward_out


class GatedFusion(nn.Module):
    """
    Gated fusion layer.

    Fuses forward and backward SSM features:
        output = σ(w_g) * forward + (1 - σ(w_g)) * backward

    Numerical stability:
        - Sigmoid ensures output in [0,1] range
        - Avoids extreme gate values that cause vanishing gradients
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Xavier init: initial gate ≈0.5 (balanced forward/backward), faster convergence
        # zero init gives sigmoid(0)=0.5 but smaller gradients and slower convergence
        self.gate_weight = nn.Parameter(torch.zeros(d_model))
        # Xavier uniform for dim=64: range ≈ ±√(6/(1+64)) ≈ ±0.30, sigmoid centered at 0.5±0.07
        # Using normal instead: std=0.5 → sigmoid output spread ≈ 0.2, more noticeable gradients
        nn.init.normal_(self.gate_weight, mean=0.0, std=0.5)
        
    def forward(
        self,
        forward_features: torch.Tensor,
        backward_features: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_weight)  # (D,)
        gate = gate.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        fused = gate * forward_features + (1 - gate) * backward_features
        return fused


class BidirectionalMambaEncoder(nn.Module):
    """
    Bidirectional Mamba encoder.

    Uses bidirectional selective SSM for temporal feature extraction:
    - Forward: scan from t=0 to t=T-1
    - Backward: scan from t=T-1 to t=0 (time-reversed)
    - Gate fusion: learnable gating network

    Architecture:
        Input → Bi-SSM → Gated Fusion → Output
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        fusion_type: str = 'gate',
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.fusion_type = fusion_type

        # === Bidirectional SSM core ===
        self.bi_ssm = BidirectionalSelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # === Fusion layer ===
        if fusion_type == 'gate':
            self.fusion = GatedFusion(d_model)
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        elif fusion_type == 'add':
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # === Normalization and Dropout ===
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        # === Residual projection ===
        self.residual_proj = nn.Linear(d_model, d_model, bias=False) if expand != 1 else None

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) input sequence
            return_intermediate: whether to return forward/backward features

        Returns:
            forward_features: (B, T, D) forward scan features
            backward_features: (B, T, D) backward scan features
            fused: (B, T, D) fused features
        """
        # Residual connection
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        # === Bidirectional SSM ===
        forward_out, backward_out = self.bi_ssm(x)

        # === Fusion ===
        if self.fusion_type == 'gate':
            fused = self.fusion(forward_out, backward_out)
        elif self.fusion_type == 'concat':
            fused = self.fusion(torch.cat([forward_out, backward_out], dim=-1))
        elif self.fusion_type == 'add':
            fused = forward_out + backward_out

        # === Residual + normalization ===
        fused = fused + residual
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        if return_intermediate:
            return forward_out, backward_out, fused
        return forward_out, backward_out, fused

    def get_fusion_weight(self) -> torch.Tensor:
        """Get fusion gate weights (for analysis)."""
        if isinstance(self.fusion, GatedFusion):
            return torch.sigmoid(self.fusion.gate_weight)
        return None
