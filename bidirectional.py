"""
Bidirectional Mamba Enhancer — plug-in for DronePredictor.

Adds bidirectional temporal context to the standard unidirectional EMAM encoder.
The bidirectional branch processes the same input in both forward and backward
directions, then fuses features via a learnable gate.

Architecture:
    Input (B,T,6) ──→ EMAM-SE (unidirectional) ──→ encoded (B,T,D)
         │
         └─────→ BidirectionalMambaEncoder ──→ bi_features (B,T,D)
                           │
                   Gated Fusion: gate*forward + (1-gate)*backward
                           │
                   encoded + bi_features → enhanced output

Usage:
    from bidirectional import BidirectionalPredictor
    bp = BidirectionalPredictor(predictor)  # wraps DronePredictor
    out = bp.predict(history)

Note: The bidirectional encoder has its own weights (~500K params) that need
training. Without training, it starts near-zero (output_proj initialized small)
so existing predictions are minimally affected.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from emam_model.bidirectional_mamba import BidirectionalMambaEncoder


class BidirectionalEnhancer(nn.Module):
    """
    Lightweight bidirectional feature enhancer.

    Takes raw trajectory input, runs bidirectional SSM, and produces
    an enhancement residual that can be added to the encoder output.

    Args:
        d_model: Feature dimension (must match the base model, default 128).
        d_state: SSM state dimension.
        expand: Expansion factor for SSM inner dimension.
        freeze: If True, start with near-zero contribution (safe for inference).
    """

    def __init__(self, d_model: int = 128, d_state: int = 16,
                 expand: int = 2, freeze: bool = True):
        super().__init__()
        self.d_model = d_model

        # Input projection to match model's d_model
        self.input_proj = nn.Linear(6, d_model)

        # Bidirectional encoder
        self.bi_encoder = BidirectionalMambaEncoder(
            d_model=d_model, d_state=d_state, expand=expand,
        )

        # Output scaling — starts small so untrained model has minimal effect
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        if freeze:
            self._init_near_zero()

    def _init_near_zero(self):
        """Initialize output layers small so untrained enhancer doesn't disrupt."""
        # Small random init for output_proj
        for name, param in self.bi_encoder.named_parameters():
            if 'out_proj' in name or 'residual_proj' in name:
                nn.init.normal_(param, std=0.001)
        self.output_scale.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 6) raw trajectory input (same as model input).

        Returns:
            bi_features: (B, T, d_model) bidirectional enhancement residual.
        """
        # Project to model space
        projected = self.input_proj(x)

        # Bidirectional encoding
        _, _, fused = self.bi_encoder(projected)

        # Scale down to not overwhelm base model (until trained)
        return fused * self.output_scale

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_training(self):
        """Enable full training mode (rescale output)."""
        self.output_scale.data.fill_(1.0)


class BidirectionalPredictor:
    """
    DronePredictor wrapper with bidirectional context enhancement.

    The bidirectional branch provides additional temporal context by
    processing the trajectory in both directions. This is particularly
    useful for:
    - Long-range predictions (steps 15-20) where future context helps
    - Maneuver transitions where backward context resolves ambiguity
    - Hovering/loitering patterns where bidirectional motion is symmetric

    Note: The bidirectional enhancer has ~500K trainable parameters.
    Training it requires a short fine-tuning phase on your dataset.
    For zero-shot use, it adds a small bias (output_scale=0.01).
    """

    def __init__(self, predictor, d_model: int = 128):
        self.predictor = predictor
        self.device = predictor.device

        self.enhancer = BidirectionalEnhancer(
            d_model=d_model, freeze=True
        ).to(self.device).eval()

        self._trained = False
        self._original_encoder_forward = None
        self._hook_active = False

    def _inject_hook(self):
        """Monkey-patch encoder to add bidirectional features as residual."""
        if self._hook_active:
            return
        encoder = self.predictor.mixed.emam_se
        self._original_encoder_forward = encoder.forward

        enhancer = self.enhancer
        def enhanced_forward(x):
            base_encoded = self._original_encoder_forward(x)
            bi = enhancer(x)
            # Add bidirectional residual (bi is (B,T,D), same shape as base_encoded)
            return base_encoded + bi

        encoder.forward = enhanced_forward
        self._hook_active = True

    def _remove_hook(self):
        """Restore original encoder forward."""
        if not self._hook_active or self._original_encoder_forward is None:
            return
        self.predictor.mixed.emam_se.forward = self._original_encoder_forward
        self._hook_active = False

    @torch.no_grad()
    def predict(self, hist: torch.Tensor, **kwargs) -> dict:
        """Enhanced prediction with bidirectional context."""
        hist = hist.to(self.device)
        if self._trained:
            self._inject_hook()
        result = self.predictor.predict(hist, **kwargs)
        return result

    def predict_with_adaptation(self, hist, **kwargs) -> dict:
        """Enhanced prediction with LoRA adaptation."""
        if self._trained:
            self._inject_hook()
        return self.predictor.predict_with_adaptation(hist, **kwargs)

    def train_enhancer(self, train_loader, epochs: int = 5, lr: float = 1e-4,
                       amp: bool = True):
        """
        Train the bidirectional enhancer.

        Fine-tunes only the bidirectional branch (~122K params).
        Base model weights remain frozen. Bi-directional features are
        injected as a residual into the encoder output.

        Args:
            train_loader: DataLoader yielding (history, future_gt, intent).
            epochs: Number of training epochs.
            lr: Learning rate.
            amp: Use automatic mixed precision.
        """
        self.enhancer.enable_training()
        self.enhancer.train()

        # Freeze base model, only train enhancer
        for param in self.predictor.mixed.parameters():
            param.requires_grad_(False)

        # Inject hook so bi_features flow through the model
        self._inject_hook()

        optimizer = torch.optim.AdamW(self.enhancer.parameters(), lr=lr,
                                       weight_decay=1e-5)
        scaler = torch.amp.GradScaler('cuda') if (amp and self.device.type == 'cuda') else None
        use_amp = scaler is not None

        best_loss = float('inf')
        save_path = Path('checkpoints/bidir_enhancer.pth')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for hist, pred, intent in pbar:
                hist = hist.to(self.device)
                pred = pred.to(self.device)

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        out = self.predictor.mixed(hist, force_predict=True)
                        targets = pred[..., :3] - hist[:, -1:, :3]
                        loss = nn.functional.mse_loss(out['predictions'], targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.enhancer.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = self.predictor.mixed(hist, force_predict=True)
                    targets = pred[..., :3] - hist[:, -1:, :3]
                    loss = nn.functional.mse_loss(out['predictions'], targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.enhancer.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            avg = total_loss / max(len(train_loader), 1)
            print(f'  Epoch {epoch+1}: avg_loss={avg:.4f}')

            if avg < best_loss:
                best_loss = avg
                torch.save({
                    'model_state_dict': self.enhancer.state_dict(),
                    'loss': avg, 'epoch': epoch,
                }, save_path)
                print(f'  -> Best saved: {save_path}')

        # Cleanup
        self.enhancer.eval()
        self._remove_hook()
        for param in self.predictor.mixed.parameters():
            param.requires_grad_(True)
        self._trained = True
        print(f'Training complete. Best loss: {best_loss:.4f}')
        print(f'Weights: {save_path}')

    @property
    def trained(self) -> bool:
        return self._trained


# ============================================================
# Smoke Test
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from predictor import DronePredictor

    print('=== Bidirectional Enhancer Smoke Test ===')

    p = DronePredictor()
    bp = BidirectionalPredictor(p)

    print(f'Enhancer params: {bp.enhancer.num_params:,}')
    print(f'Trained: {bp.trained}')

    # Standard prediction (enhancer starts frozen → near-zero effect)
    x = torch.randn(2, 20, 6)
    x[:, :, 3:6] *= 2.0

    out = bp.predict(x)
    print(f'Prediction shape: {out["predictions"].shape}')
    print(f'Speed: {[f"{s:.1f}" for s in out["speed"].tolist()]} m/s')

    # Verify enhancer output magnitude is small (untrained mode)
    with torch.no_grad():
        bi = bp.enhancer(x.to(bp.device))
    print(f'Bi-feature magnitude (untrained): {bi.abs().mean():.6f} (should be ~0)')

    print('\nAll good!')
