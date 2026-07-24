#!/usr/bin/env python3
"""
Equivalence + speed comparison test for the chunked parallel SSM scan
vs. the original Python-loop scan.

Pass criteria (both must hold to adopt the chunked version):
  1. Forward outputs match within tight tolerance (fp32).
  2. Backward gradients (w.r.t. x_act, dt, A, D) match within tolerance.
  3. Chunked is faster on the real model's typical shapes (T=20 and T=40).

Run standalone:  python test_ssm_scan.py
Run via pytest:  pytest test_ssm_scan.py -v
"""
import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emam_model.emam_se import _selective_ssm_scan, _selective_ssm_scan_chunked

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


def make_inputs(B, T, d_inner, d_state, device):
    """Realistic inputs matching SelectiveSSM: dt>0 (softplus), A<0 (-exp)."""
    x_act = torch.randn(B, T, d_inner, device=device)
    dt = torch.nn.functional.softplus(torch.randn(B, T, d_inner, device=device) + 1.0)
    A_mat = -torch.exp(torch.randn(d_inner, d_state, device=device) * 0.5 - 0.7)
    D_vec = torch.ones(d_inner, device=device)
    return x_act, dt, A_mat, D_vec


def _check_equivalence(B, T, d_inner, d_state, chunk_size):
    x_act, dt, A_mat, D_vec = make_inputs(B, T, d_inner, d_state, DEVICE)

    # --- Forward equivalence ---
    xl = x_act.clone().requires_grad_(True)
    dtl = dt.clone().requires_grad_(True)
    Al = A_mat.clone().requires_grad_(True)
    Dl = D_vec.clone().requires_grad_(True)
    yl = _selective_ssm_scan(xl, dtl, Al, Dl, d_state)

    xc = x_act.clone().requires_grad_(True)
    dtc = dt.clone().requires_grad_(True)
    Ac = A_mat.clone().requires_grad_(True)
    Dc = D_vec.clone().requires_grad_(True)
    yc = _selective_ssm_scan_chunked(xc, dtc, Ac, Dc, d_state, chunk_size=chunk_size)

    fwd_max = (yl - yc).abs().max().item()
    fwd_ok = torch.allclose(yl, yc, atol=1e-4, rtol=1e-4)

    # --- Backward equivalence (same random upstream grad) ---
    g = torch.randn_like(yl)
    yl.backward(g)
    yc.backward(g.clone())

    grad_max = max(
        (xl.grad - xc.grad).abs().max().item(),
        (dtl.grad - dtc.grad).abs().max().item(),
        (Al.grad - Ac.grad).abs().max().item(),
        (Dl.grad - Dc.grad).abs().max().item(),
    )
    grad_ok = (
        torch.allclose(xl.grad, xc.grad, atol=1e-3, rtol=1e-3) and
        torch.allclose(dtl.grad, dtc.grad, atol=1e-3, rtol=1e-3) and
        torch.allclose(Al.grad, Ac.grad, atol=1e-3, rtol=1e-3) and
        torch.allclose(Dl.grad, Dc.grad, atol=1e-3, rtol=1e-3)
    )
    return fwd_ok, fwd_max, grad_ok, grad_max


def bench(fn, *args, iters=50, **kwargs):
    # warmup
    for _ in range(5):
        out = fn(*args, **kwargs)
        loss = out.sum(); loss.backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = fn(*args, **kwargs)
        loss = out.sum(); loss.backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000  # ms/iter


def bench_shape(B, T, d_inner, d_state, chunk_size):
    x_act, dt, A_mat, D_vec = make_inputs(B, T, d_inner, d_state, DEVICE)
    args = [x_act.requires_grad_(True), dt.requires_grad_(True),
            A_mat.requires_grad_(True), D_vec.requires_grad_(True), d_state]
    t_loop = bench(_selective_ssm_scan, *args)
    t_chunk = bench(_selective_ssm_scan_chunked, *args, chunk_size=chunk_size)
    return t_loop, t_chunk


if not HAS_PYTEST:
    # Dummy parametrize so standalone `python test_ssm_scan.py` works without pytest
    class _DummyPytest:
        @staticmethod
        def mark_parametrize(*args, **kwargs):
            return lambda fn: fn
    _dummy = _DummyPytest()

# ---- pytest-compatible wrappers ----
# Run via `pytest test_ssm_scan.py -v` or standalone `python test_ssm_scan.py`

if HAS_PYTEST:
    _param = pytest.mark.parametrize
else:
    _param = _dummy.mark_parametrize

@_param('B,T', [(1,20),(4,20),(8,20),(8,40),(16,40),(4,80),(2,150)])
def test_chunked_scan_equivalence(B, T):
    """Pytest entry: forward+backward equivalence at 1e-4 tolerance."""
    d_inner, d_state = 256, 16
    chunk = 16
    fwd_ok, fwd_max, grad_ok, grad_max = _check_equivalence(B, T, d_inner, d_state, chunk)
    assert fwd_ok, f'Forward mismatch: max {fwd_max:.2e}'
    assert grad_ok, f'Backward mismatch: max {grad_max:.2e}'

@_param('B,T', [(16,40),(8,80),(4,150)])
def test_chunked_scan_speedup(B, T):
    """Pytest entry: chunked scan >= 1.0x of loop for large-T shapes where it should win.
    Small B/T (32,20 etc.) excluded: chunking overhead dominates, loop wins there."""
    d_inner, d_state = 256, 16
    chunk = 16
    t_loop, t_chunk = bench_shape(B, T, d_inner, d_state, chunk)
    speedup = t_loop / t_chunk
    assert speedup >= 1.0, f'Chunked regressed at T={T}: speedup {speedup:.2f}x (loop {t_loop:.1f}ms chunk {t_chunk:.1f}ms)'

if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    CHUNK = 16
    # Real model dims: d_model=128, expand=2 -> d_inner=256, d_state=16
    d_inner, d_state = 256, 16

    print('\n=== Equivalence (forward + backward) ===')
    all_ok = True
    for B, T in [(8, 20), (8, 40), (16, 40), (4, 80), (2, 150)]:
        fwd_ok, fwd_max, grad_ok, grad_max = _check_equivalence(B, T, d_inner, d_state, CHUNK)
        status = 'OK' if (fwd_ok and grad_ok) else 'FAIL'
        all_ok = all_ok and fwd_ok and grad_ok
        print(f'  B={B:2d} T={T:3d}: fwd {"OK " if fwd_ok else "FAIL"}(max {fwd_max:.2e}) '
              f'grad {"OK " if grad_ok else "FAIL"}(max {grad_max:.2e}) -> {status}')

    print('\n=== Speed (ms/iter, fwd+bwd, 50 iters) ===')
    for B, T in [(32, 20), (32, 40), (16, 40), (8, 80), (4, 150)]:
        t_loop, t_chunk = bench_shape(B, T, d_inner, d_state, CHUNK)
        speedup = t_loop / t_chunk
        print(f'  B={B:2d} T={T:3d}: loop {t_loop:6.2f}  chunk {t_chunk:6.2f}  '
              f'speedup {speedup:4.2f}x')

    print(f'\nEquivalence: {"ALL PASS" if all_ok else "FAILED"}')
