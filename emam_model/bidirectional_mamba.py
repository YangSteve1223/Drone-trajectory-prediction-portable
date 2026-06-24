"""
Bidirectional Mamba Encoder - 完整实现

来源论文: Motion Mamba / Mamba (SSM)
核心机制: 双向选择性状态空间模型
    - 前向扫描: t = 0 → T-1, h_init = 0
    - 后向扫描: t = T-1 → 0, h_init = 0 (时间翻转)
    - 门控融合: fused = gate * forward + (1 - gate) * backward

Mamba 选择性 SSM 核心公式:
    选择性机制: B_t = linear(x_t), C_t = linear(x_t) (输入依赖)
    A_discrete = exp(dt * ΔA), dt = softplus(log_dt)
    状态更新: h_t = A_discrete · h_{t-1} + B_t · x_t
    输出: y_t = einsum(C_t, h_t) + D · x_t

与原论文Motion Mamba差异:
    - Motion Mamba BSM: 沿通道维度扫描 (C, B, T) 排列
    - EMam-SE: 沿时间维度扫描 (B, T, D)，更符合轨迹预测需求
    - 本实现: 使用输入依赖的选择性B/C（符合Mamba原论文）

集成位置: 并行于 EMam-SE，残差增强 encoded 特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BidirectionalSelectiveSSM(nn.Module):
    """
    双向选择性状态空间模型
    
    实现双向SSM扫描，分别捕获前向和后向的时间依赖关系。
    基于Mamba选择性机制，B和C都是输入依赖的。
    
    核心公式:
        A_discrete = exp(dt * A), 其中 dt = softplus(x_dt)
        B_t = linear(x_inner), C_t = linear(x_inner)  # 选择性
        h_t = A_discrete · h_{t-1} + B_t · x_inner
        y_t = C_t · h_t + D · x_inner
    """
    
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

        # === 输入投影: x → [x_conv, x_dt] ===
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=True)

        # === SSM核心参数 ===
        # A: 状态演化矩阵 (d_inner, d_state)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        # D: 跳接矩阵 (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # === 局部因果卷积 ===
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # === 输出投影 ===
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        """初始化SSM参数"""
        # A矩阵: 取负值保证状态稳定 (exp(-|A|) ∈ (0,1))
        nn.init.xavier_uniform_(self.A)
        with torch.no_grad():
            self.A.copy_(-torch.abs(self.A))
        # D矩阵: 全1
        nn.init.ones_(self.D)
        # 卷积层初始化
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1d.bias)

    def _ssm_scan(
        self,
        x_conv: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        SSM序列扫描
        
        Args:
            x_conv: (B, T, d_inner) 卷积后的特征
            reverse: 是否反向扫描
        
        Returns:
            output: (B, T, d_inner)
        """
        B, T, d_inner = x_conv.shape
        
        if reverse:
            x_conv = torch.flip(x_conv, dims=[1])
        
        # === 选择性参数计算 ===
        # dt: softplus确保正值
        dt = F.softplus(x_conv)  # (B, T, d_inner)
        
        # A离散化: A_dis[b,t,i,k] = exp(dt[b,t,i] * A[i,k])
        # 结果形状: (B, T, d_inner, d_state)
        # 数值稳定性: dt*A 可能过大导致 exp 爆炸，clamp 到 [-50, 50]
        dt_A = torch.einsum('btd,dn->btdn', dt, self.A)
        dt_A = dt_A.clamp(min=-50.0, max=50.0)
        A_dis = torch.exp(dt_A)
        
        # 选择性B: B_t = x_conv (直接使用，无额外投影)
        # 选择性C: C_t = x_conv (直接使用，无额外投影)
        B_t = x_conv  # (B, T, d_inner)
        C_t = x_conv  # (B, T, d_inner)
        
        # === SSM扫描 ===
        # 初始化隐状态: h_0 = 0
        h = torch.zeros(B, self.d_inner, self.d_state, 
                       device=x_conv.device, dtype=x_conv.dtype)
        
        outputs = []
        for t in range(T):
            A_t = A_dis[:, t]        # (B, d_inner, d_state)
            B_t_t = B_t[:, t]        # (B, d_inner)
            C_t_t = C_t[:, t]        # (B, d_inner)

            # h_term[b,i] = Σ_k A_t[b,i,k] * h[b,i,k]
            h_term = torch.einsum('bik,bik->bi', A_t, h)  # (B, d_inner)

            # h_new[b,i,k] = h_term[b,i] * A_t[b,i,k] + B_t[b,i] * C_t[b,i]
            # 与 emam_se.py 一致: h_new = h_term * A_t + B_t * C_t
            # B_t_t.unsqueeze(-1): (B, d_inner, 1), C_t_t.unsqueeze(-1): (B, d_inner, 1)
            # 结果: (B, d_inner, d_state)
            h_new = h_term.unsqueeze(-1) * A_t + B_t_t.unsqueeze(-1) * C_t_t.unsqueeze(-1)
            h_new = h_new.clamp(-10, 10)  # 数值稳定性

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
        双向SSM前向传播
        
        Args:
            x: (B, T, D)
        
        Returns:
            forward_out: (B, T, D) 前向扫描结果
            backward_out: (B, T, D) 后向扫描结果
        """
        B, T, D = x.shape
        
        # === 输入投影 ===
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_conv_raw, x_dt_raw = xz.chunk(2, dim=-1)
        
        # === 预处理 ===
        # 结合x_conv_raw和x_dt_raw计算dt
        dt_input = x_conv_raw * torch.sigmoid(x_dt_raw)
        
        # === 因果卷积 ===
        x_conv = dt_input.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # 截断
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)  # SiLU激活
        
        # === 双向SSM扫描 ===
        forward_out = self._ssm_scan(x_conv, reverse=False)
        backward_out = self._ssm_scan(x_conv, reverse=True)
        
        # === 输出投影 ===
        forward_out = self.out_proj(forward_out)
        backward_out = self.out_proj(backward_out)
        
        return forward_out, backward_out


class GatedFusion(nn.Module):
    """
    门控融合层
    
    融合前向和后向SSM特征:
        output = σ(w_g) * forward + (1 - σ(w_g)) * backward
    
    数值稳定性:
        - 使用sigmoid确保输出在[0,1]范围
        - 避免极端的门控值导致梯度消失
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Xavier初始化: 初始门控值≈0.5（前向后向均衡），加速收敛
        # zero init 会导致 sigmoid(0)=0.5，梯度更小收敛更慢
        self.gate_weight = nn.Parameter(torch.zeros(d_model))
        # Xavier uniform 对 dim=64: range ≈ ±√(6/(1+64)) ≈ ±0.30, sigmoid 后集中在 0.5±0.07
        # 改用正态分布: std=0.5 → sigmoid 输出 spread ≈ 0.2, 梯度更明显
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
    双向 Mamba 编码器
    
    使用双向选择性SSM进行时序特征提取:
    - Forward: 从 t=0 扫描到 t=T-1
    - Backward: 从 t=T-1 扫描到 t=0 (时间翻转)
    - Gate fusion: 可学习门控网络
    
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
        
        # === 双向SSM核心 ===
        self.bi_ssm = BidirectionalSelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # === 融合层 ===
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
        
        # === 归一化和Dropout ===
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # === 残差投影 ===
        self.residual_proj = nn.Linear(d_model, d_model, bias=False) if expand != 1 else None

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) 输入序列
            return_intermediate: 是否返回前向/后向特征
        
        Returns:
            forward_features: (B, T, D) 前向扫描特征
            backward_features: (B, T, D) 后向扫描特征
            fused: (B, T, D) 融合特征
        """
        # 残差连接
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # === 双向SSM ===
        forward_out, backward_out = self.bi_ssm(x)
        
        # === 融合 ===
        if self.fusion_type == 'gate':
            fused = self.fusion(forward_out, backward_out)
        elif self.fusion_type == 'concat':
            fused = self.fusion(torch.cat([forward_out, backward_out], dim=-1))
        elif self.fusion_type == 'add':
            fused = forward_out + backward_out
        
        # === 残差连接 + 归一化 ===
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
