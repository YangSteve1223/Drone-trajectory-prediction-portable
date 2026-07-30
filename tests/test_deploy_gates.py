#!/usr/bin/env python3
"""Regression tests for DeployedLowPredictor's three gates and the online-learning
resident adapter.

Covers the two bugs fixed during development that have no other guard:
  1. The gate routing (length / global-toggle / speed) must pick the right base.
  2. The online learner must keep its adapter RESIDENT across updates — after
     enough (hist, gt) observations the per-drone LoRA (lora_B) must become
     non-zero. An earlier version deactivated the adapter after every update and
     silently discarded the learned weights (lora_B stayed 0).

Runs on CPU in a few seconds. Run from the project root:
`python tests/test_deploy_gates.py` or `pytest tests/test_deploy_gates.py`.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import torch

from deploy import DeployedLowPredictor
from online_learner import OnlineLearnerConfig


def _make_hist(speed: float, n: int = 40) -> torch.Tensor:
    """A (1, n, 6) history whose last-5-frame mean velocity norm is ~speed."""
    hist = torch.zeros(1, n, 6)
    # constant velocity along x -> position ramps, speed = |vx|
    hist[0, :, 3] = speed
    hist[0, :, 0] = torch.arange(n, dtype=torch.float32) * speed
    return hist


def _make_gt(n: int = 20) -> torch.Tensor:
    return torch.randn(1, n, 3) * 0.1


def _build(**kw):
    # tiny accumulation so learning triggers within a short test flight
    d = DeployedLowPredictor(device='cpu', **kw)
    d.learner.config = OnlineLearnerConfig(
        accumulation_steps=4, device='cpu', conf_threshold=0.0)
    return d


def test_length_gate_closed_below_threshold():
    d = _build(online_min_frames=60)  # default use_global=False
    out = d.predict(_make_hist(1.0), drone_id='d1', frames_seen=30)
    assert out['length_gate_open'] is False
    assert out['online_active'] is False
    assert tuple(out['predictions'].shape) == (1, 20, 3)


def test_length_gate_open_above_threshold():
    d = _build(online_min_frames=60)  # default use_global=False
    out = d.predict(_make_hist(1.0), drone_id='d1', frames_seen=120)
    assert out['length_gate_open'] is True
    assert out['online_active'] is True


def test_default_is_no_global():
    """Default DeployedLowPredictor has use_global=False."""
    d = _build()  # no use_global kwarg
    out = d.predict(_make_hist(1.0), frames_seen=10)
    assert out['base'] == 'plain'
    assert out['global_active'] is False
    assert d.base_global is None


def test_speed_gate_disables_global_when_fast():
    d = _build(use_global=True, global_max_speed=4.0)
    slow = d.predict(_make_hist(1.0), frames_seen=10)   # within 0-3 m/s domain
    fast = d.predict(_make_hist(9.0), frames_seen=10)   # out of domain
    assert slow['base'] == 'global' and slow['global_active'] is True
    assert fast['base'] == 'plain' and fast['global_active'] is False


def test_global_toggle_off_never_uses_global():
    d = _build(use_global=False)
    out = d.predict(_make_hist(1.0), frames_seen=10)
    assert out['base'] == 'plain'
    assert out['global_active'] is False


def test_online_learner_keeps_adapter_resident_and_learns():
    """After enough observations the resident per-drone LoRA must be non-zero."""
    d = _build(online_min_frames=1)  # default use_global=False
    drone = 'learner_drone'
    updated_any = False
    for t in range(12):
        out = d.predict(_make_hist(1.0), drone_id=drone,
                        ground_truth=_make_gt(), frames_seen=100, timestep=t)
        updated_any = updated_any or out['updated']

    assert updated_any, 'online learner never triggered an update'
    # adapter must still be resident (not deactivated after update)
    assert d.mgr.active_drone == drone
    assert d.mgr.adapter is not None

    lora_b_energy = sum(
        layer.lora_B.abs().sum().item()
        for layer in d.mgr.adapter.lora_layers.values())
    assert lora_b_energy > 0.0, (
        'lora_B is all-zero: adapter was reset/deactivated, learning discarded')


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print(f'=== DeployedLowPredictor gate + online-learning tests ({len(tests)}) ===')
    for fn in tests:
        fn()
        print(f'  PASS  {fn.__name__}')
    print('\nAll tests passed!')


if __name__ == '__main__':
    _run_all()
