"""
LoRA (Low-Rank Adaptation) adapter for EMAM drone trajectory predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy


class LoRALinear(nn.Module):
    """Wraps nn.Linear with a low-rank adapter: y = Wx + (alpha/r) * B @ A @ x."""

    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base weights
        base_layer.weight.requires_grad_(False)
        if base_layer.bias is not None:
            base_layer.bias.requires_grad_(False)

        # LoRA matrices: A ~ Kaiming, B ~ zeros (initially LoRA output = base output)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(in_features, r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(r, out_features, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen)
        base_out = self.base_layer(x)

        # LoRA contribution: (x @ A) @ B * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A  # (..., in) @ (in, r) -> (..., r)
        lora_out = lora_out @ self.lora_B               # (..., r) @ (r, out) -> (..., out)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    @property
    def weight(self):
        """Effective weight matrix: W + B^T A^T * scaling."""
        return self.base_layer.weight + (self.lora_B.T @ self.lora_A.T).T * self.scaling

    @property
    def bias(self):
        return self.base_layer.bias

    def merge(self):
        """Merge LoRA into base weights so the plain Linear reproduces the adapted output.

        forward does y = x@W.T + (x@A@B)*scaling, with A:(in,r), B:(r,out).
        nn.Linear stores W:(out,in) and computes x@W.T, so the equivalent weight
        delta is (A@B).T * scaling  ->  shape (out,in). (The previous
        (B@A.T).T formulation had mismatched shapes and never ran.)
        """
        delta = (self.lora_A.data @ self.lora_B.data).T * self.scaling   # (out, in)
        self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def unmerge(self, saved_delta: Optional[torch.Tensor] = None):
        """Remove LoRA delta from base weights."""
        if saved_delta is not None:
            self.base_layer.weight.data -= saved_delta.to(self.base_layer.weight.dtype)


# Default target layers (upstream-only fallback for LoRAAdapter when the caller
# passes no explicit targets — e.g. the smoke test below).
#
# The AUTHORITATIVE production config lives in online_config.ONLINE_LORA_TARGETS /
# ONLINE_HEAD_TARGETS; deploy.py / adapter_manager / online_learner all use that and
# never fall back to these. These defaults are kept only so `create_lora_model(model)`
# works standalone, and they intentionally mirror the validated upstream-only setup.
#
# NOTE: `ua_pgd.neural_decoder.delta_head` is deliberately EXCLUDED. The delta_head
# processes each of the 20 prediction steps independently (ua_pgd.py), so adapting it
# amplifies per-step differences and produces zigzag trajectories. Do not add it back.
DEFAULT_LORA_TARGETS = [
    'emam_se.mamba_blocks.0.ssm.in_proj',
    'emam_se.mamba_blocks.0.ssm.out_proj',
    'emam_se.mamba_blocks.1.ssm.in_proj',
    'emam_se.mamba_blocks.1.ssm.out_proj',
]

DEFAULT_HEAD_TARGETS = [
    'ua_pgd.anchor_to_pos.2',
]


class LoRAAdapter(nn.Module):
    """Injects LoRA into target layers of a TrajectoryPredictor and manages Group A/B params."""

    def __init__(self, model: nn.Module, r: int = 4, alpha: float = 4.0,
                 dropout: float = 0.0,
                 lora_targets=None,
                 head_targets: List[str] = None):
        super().__init__()
        self.r = r
        self.alpha = alpha
        # lora_targets may be:
        #   - a list of str paths (uniform rank r), or
        #   - a list of (path, rank) tuples, or
        #   - a dict {path: rank}   -> per-target ranks.
        # alpha per target defaults to 2*rank (matches the offline eval scripts).
        raw = lora_targets if lora_targets is not None else DEFAULT_LORA_TARGETS
        self._target_ranks = self._normalize_targets(raw, r)
        self.lora_targets = list(self._target_ranks.keys())
        self.head_targets = head_targets or DEFAULT_HEAD_TARGETS

        self.model = model
        self.lora_layers: Dict[str, LoRALinear] = {}
        self.head_params: Dict[str, nn.Parameter] = {}
        self._original_layers: Dict[str, nn.Module] = {}
        self._original_head_states: Dict[str, torch.Tensor] = {}
        self._active = False

    @staticmethod
    def _normalize_targets(raw, default_r: int) -> Dict[str, int]:
        """Normalize lora_targets (list[str] | list[(path,rank)] | dict) -> {path: rank}."""
        out: Dict[str, int] = {}
        if isinstance(raw, dict):
            return dict(raw)
        for item in raw:
            if isinstance(item, (tuple, list)):
                path, rank = item[0], int(item[1])
            else:
                path, rank = item, default_r
            out[path] = rank
        return out

    def _resolve_module(self, path: str) -> nn.Module:
        """Resolve dotted path to a submodule, e.g. 'emam_se.mamba_blocks.0.ssm.in_proj'."""
        parts = path.split('.')
        obj = self.model
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _set_module(self, path: str, module: nn.Module):
        """Replace a submodule at the given dotted path."""
        parts = path.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    def activate(self):
        """Replace target layers with LoRALinear and enable head param gradients."""
        if self._active:
            return

        # Group B: Inject LoRA into SSM projection layers (per-target rank)
        for path in self.lora_targets:
            original = self._resolve_module(path)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"{path} is not nn.Linear, got {type(original)}")
            self._original_layers[path] = original
            rank = self._target_ranks[path]
            # per-target alpha = 2*rank when ranks are non-uniform, else use self.alpha
            alpha = self.alpha if len(set(self._target_ranks.values())) == 1 else rank * 2.0
            lora_layer = LoRALinear(original, r=rank, alpha=alpha)
            self._set_module(path, lora_layer)
            self.lora_layers[path] = lora_layer

        # Group A: Enable full finetuning of small output heads
        for path in self.head_targets:
            layer = self._resolve_module(path)
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"{path} is not nn.Linear, got {type(layer)}")
            self._original_head_states[path] = {
                'weight': layer.weight.data.clone(),
                'bias': layer.bias.data.clone() if layer.bias is not None else None,
            }
            layer.weight.requires_grad_(True)
            if layer.bias is not None:
                layer.bias.requires_grad_(True)
            self.head_params[f'{path}.weight'] = layer.weight
            if layer.bias is not None:
                self.head_params[f'{path}.bias'] = layer.bias

        self._active = True

    def deactivate(self):
        """Restore original layers and freeze head parameters."""
        if not self._active:
            return

        for path, original in self._original_layers.items():
            self._set_module(path, original)

        for path, state in self._original_head_states.items():
            layer = self._resolve_module(path)
            layer.weight.data.copy_(state['weight'])
            if state['bias'] is not None:
                layer.bias.data.copy_(state['bias'])
            layer.weight.requires_grad_(False)
            if layer.bias is not None:
                layer.bias.requires_grad_(False)

        self._original_layers.clear()
        self._original_head_states.clear()
        self.head_params.clear()
        self.lora_layers.clear()
        self._active = False

    def get_lora_state(self) -> Dict:
        """Export LoRA A, B matrices for serialization."""
        return {path: {'A': layer.lora_A.data.clone(),
                       'B': layer.lora_B.data.clone()}
                for path, layer in self.lora_layers.items()}

    def get_head_state(self) -> Dict:
        """Export head layer weights for serialization."""
        state = {}
        for path in self.head_targets:
            layer = self._resolve_module(path)
            state[f'{path}.weight'] = layer.weight.data.clone()
            if layer.bias is not None:
                state[f'{path}.bias'] = layer.bias.data.clone()
        return state

    def load_lora_state(self, lora_state: Dict):
        """Load LoRA A, B matrices from a saved state dict."""
        for path, matrices in lora_state.items():
            if path in self.lora_layers:
                self.lora_layers[path].lora_A.data.copy_(matrices['A'])
                self.lora_layers[path].lora_B.data.copy_(matrices['B'])

    def load_head_state(self, head_state: Dict):
        """Load head weights from a saved state dict."""
        for key, tensor in head_state.items():
            path, attr = key.rsplit('.', 1)
            layer = self._resolve_module(path)
            if attr == 'weight':
                layer.weight.data.copy_(tensor)
            elif attr == 'bias' and layer.bias is not None:
                layer.bias.data.copy_(tensor)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return all trainable parameters: LoRA A, B plus head params."""
        params = []
        for layer in self.lora_layers.values():
            params.extend([layer.lora_A, layer.lora_B])
        params.extend(self.head_params.values())
        return params

    def reset(self):
        """Reset all LoRA and head parameters to their initial state."""
        for layer in self.lora_layers.values():
            layer.reset_lora_parameters()
        for path, state in self._original_head_states.items():
            layer = self._resolve_module(path)
            layer.weight.data.copy_(state['weight'])
            if state['bias'] is not None:
                layer.bias.data.copy_(state['bias'])

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    @property
    def active(self) -> bool:
        return self._active


def merge_lora(adapter: LoRAAdapter):
    """Merge LoRA weights into the base model permanently."""
    for layer in adapter.lora_layers.values():
        layer.merge()


def unmerge_lora(adapter: LoRAAdapter, saved_deltas: Dict[str, torch.Tensor]):
    """Remove LoRA contribution from the base model."""
    for path, delta in saved_deltas.items():
        if path in adapter.lora_layers:
            adapter.lora_layers[path].unmerge(delta)


def create_lora_model(model: nn.Module, r: int = 4, alpha: float = 4.0,
                      lora_targets: List[str] = None,
                      head_targets: List[str] = None) -> LoRAAdapter:
    """Create and activate a LoRA adapter on the given model."""
    adapter = LoRAAdapter(model, r=r, alpha=alpha,
                          lora_targets=lora_targets, head_targets=head_targets)
    adapter.activate()
    return adapter


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from emam_model import TrajectoryPredictor

    print('=== LoRA Smoke Test ===')

    # Build model
    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).eval()
    print(f'Base model params: {sum(p.numel() for p in model.parameters()):,}')

    # Create LoRA adapter
    adapter = create_lora_model(model, r=4)
    print(f'LoRA trainable params: {adapter.num_params:,}')

    # Test forward pass
    x = torch.randn(2, 20, 6)
    with torch.no_grad():
        out_base = model(x, force_predict=True)

    # Reset LoRA (B=0) -> output should match base
    adapter.reset()
    with torch.no_grad():
        out_lora_zero = model(x, force_predict=True)
    match = torch.allclose(out_base['predictions'], out_lora_zero['predictions'], atol=1e-5)
    print(f'Zero-init B -> output matches base: {match}')

    # Random LoRA should change output. Both A and B must be non-zero — with the
    # zero-init B alone the low-rank term (x@A)@B stays zero and the output is
    # unchanged, so randomizing A only would not exercise this path.
    for layer in adapter.lora_layers.values():
        nn.init.normal_(layer.lora_A, std=0.1)
        nn.init.normal_(layer.lora_B, std=0.1)
    with torch.no_grad():
        out_lora_rand = model(x, force_predict=True)
    diff = (out_lora_rand['predictions'] - out_base['predictions']).abs().mean().item()
    print(f'Random LoRA -> mean prediction diff: {diff:.6f} (should be > 0)')

    # Deactivate and verify restoration
    adapter.deactivate()
    with torch.no_grad():
        out_restored = model(x, force_predict=True)
    restored_match = torch.allclose(out_base['predictions'], out_restored['predictions'], atol=1e-5)
    print(f'Deactivate -> output restored: {restored_match}')

    # State export/import. Set a known non-trivial state, snapshot it and its output,
    # then reset (which zeroes B -> output collapses back to base) and reload. The
    # reloaded output must match the snapshot, and must differ from the reset state.
    adapter.activate()
    for layer in adapter.lora_layers.values():
        nn.init.normal_(layer.lora_A, std=0.1)
        nn.init.normal_(layer.lora_B, std=0.1)
    lora_state = adapter.get_lora_state()
    head_state = adapter.get_head_state()
    with torch.no_grad():
        out_snapshot = model(x, force_predict=True)

    adapter.reset()
    with torch.no_grad():
        out_after_reset = model(x, force_predict=True)
    reset_changed = not torch.allclose(out_snapshot['predictions'], out_after_reset['predictions'], atol=1e-5)

    adapter.load_lora_state(lora_state)
    adapter.load_head_state(head_state)
    with torch.no_grad():
        out_reloaded = model(x, force_predict=True)
    reload_match = torch.allclose(out_snapshot['predictions'], out_reloaded['predictions'], atol=1e-5)
    print(f'Reset actually clears LoRA: {reset_changed}')
    print(f'State save/load -> output preserved: {reload_match}')

    adapter.deactivate()
    assert match and restored_match and reset_changed and reload_match, 'LoRA smoke test FAILED'
    print(f'\nAll tests passed!')
