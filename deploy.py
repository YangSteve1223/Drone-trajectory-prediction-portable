#!/usr/bin/env python3
"""
Deployment entry for LONG-trajectory prediction with per-drone online learning.

Separate from predictor.DronePredictor (which handles short/general inference with
Z-correction + soft-fusion). This entry owns the online-learning stack and its
three gates:

  1) LENGTH gate    — per-drone online learning only turns on once the drone has
                      streamed >= online_min_frames. Short flights just use the
                      plain 40-frame base (no LoRA).
  2) GLOBAL toggle  — use_global switches the shared low-speed global LoRA
                      (dir_lora_40) on/off entirely.
  3) SPEED gate     — even with use_global=True, the global LoRA is disabled when
                      current speed > global_max_speed (it was trained on 0-3 m/s
                      real DJI flight; faster motion is out of its domain).

Implementation note: the global LoRA is merged into weights and cannot be
un-merged at runtime, so we hold TWO low bases — base_plain and base_global — and
route to whichever the gates select. Per-drone LoRA stacks on the active base.
"""

import torch
import warnings
from pathlib import Path

import online_config as OC
from adapter_manager import DroneAdapterManager
from online_learner import OnlineLearner, OnlineLearnerConfig


class DeployedLowPredictor:
    def __init__(self, device=None,
                 use_global: bool = True,
                 global_max_speed: float = OC.GLOBAL_MAX_SPEED,
                 online_min_frames: int = OC.ONLINE_MIN_FRAMES,
                 checkpoint_dir: str = 'weights/online_adapters',
                 accumulation_steps: int = 10):
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_global = use_global
        self.global_max_speed = global_max_speed
        self.online_min_frames = online_min_frames

        # Two LOW bases: plain and global-merged (global merged once at load).
        self.base_plain = OC.build_base_model(device=str(self.device))
        self.base_global = None
        if use_global:
            self.base_global = OC.build_base_model(device=str(self.device))
            ok = OC.merge_global_lora(self.base_global, device=str(self.device))
            if not ok:
                warnings.warn(f'Global LoRA {OC.GLOBAL_LORA_FILE} not found; '
                              'global layer disabled.')
                self.use_global = False
                self.base_global = None

        # Per-drone online learning attaches to the plain base (the per-drone LoRA
        # is base-agnostic; using one manager keeps a single adapter store).
        self.mgr = DroneAdapterManager(self.base_plain, checkpoint_dir=checkpoint_dir)
        cfg = OnlineLearnerConfig(accumulation_steps=accumulation_steps,
                                  device=str(self.device), conf_threshold=0.0)
        self.learner = OnlineLearner(self.mgr, cfg)
        self._active_drone = None
        self._session_base_tag = None

    @staticmethod
    def _speed(hist):
        vel = hist[:, :, 3:6]
        return torch.norm(vel[:, -5:, :], dim=2).mean(dim=1)

    def _select_base(self, speed_val: float):
        """Global toggle + speed gate -> which base to use, and a tag."""
        if self.use_global and self.base_global is not None and speed_val <= self.global_max_speed:
            return self.base_global, 'global'
        return self.base_plain, 'plain'

    def predict(self, hist: torch.Tensor, drone_id: str = None,
                ground_truth: torch.Tensor = None,
                frames_seen: int = None, timestep: int = 0) -> dict:
        """Predict with the three gates.

        hist:         (1, 40, 6) recent history window.
        drone_id:     enables per-drone online LoRA when set.
        ground_truth: (1, 20, 3) future displacement for online learning (optional).
        frames_seen:  total frames the drone has streamed (for the length gate).
                      If None, inferred as hist length (i.e. gate open).
        """
        hist = hist.to(self.device)
        if hist.dim() != 3:
            hist = hist.unsqueeze(0)
        speed_val = float(self._speed(hist).item())
        base, base_tag = self._select_base(speed_val)

        # LENGTH gate: only personalize on long-enough flights.
        n_seen = frames_seen if frames_seen is not None else hist.shape[1]
        length_ok = n_seen >= self.online_min_frames

        # The online adapter is bound to a single base for the whole drone session
        # (chosen by the gates the first time the length gate opens). Per-frame base
        # flipping would deactivate/reactivate the adapter and wipe in-memory
        # learning, so the base is fixed per session.
        adapted = False
        if drone_id and length_ok:
            if self._active_drone != drone_id:
                if self.mgr.active_drone is not None:
                    self.mgr.deactivate()
                self.mgr.base_model = base           # bind once
                self._session_base_tag = base_tag
                self._active_drone = drone_id
            adapted = True

        # Online learning step. learner.observe() FULLY owns adapter activation and
        # keeps it resident (do NOT call mgr.activate() here — double management
        # reloads stale disk weights and wipes learning). After observe, the trained
        # adapter is resident on mgr.adapter.model.
        updated = False
        if adapted and ground_truth is not None:
            gt = ground_truth.to(self.device)
            if gt.dim() == 3:
                gt = gt[0]
            updated = self.learner.observe(
                drone_id, hist[0].cpu(), gt.cpu(), confidence=1.0, timestep=timestep)

        # Choose model for prediction.
        if adapted:
            if self.mgr.active_drone == drone_id and self.mgr.adapter is not None:
                # learner has this drone's adapter resident (with in-memory learning)
                model = self.mgr.adapter.model
            elif ground_truth is None and self.mgr.has_adapter(drone_id):
                # inference-only path (no learning this call): load saved adapter
                self.mgr.activate(drone_id)
                model = self.mgr.adapter.model
            else:
                # still buffering (learner hasn't triggered its first update yet):
                # predict from the base. Do NOT activate — that would create a
                # fresh/disk adapter and clobber the learner's pending state.
                model = base
            eff_base_tag = getattr(self, '_session_base_tag', base_tag)
        else:
            model = base
            eff_base_tag = base_tag

        with torch.no_grad():
            out = model(hist, force_predict=True)

        return {
            'predictions': out['predictions'],
            'speed': speed_val,
            'base': eff_base_tag,
            'global_active': eff_base_tag == 'global',
            'online_active': adapted,
            'length_gate_open': length_ok,
            'updated': updated,
        }

    def save(self):
        self.learner.save_all()


if __name__ == '__main__':
    import numpy as np
    print('=== DeployedLowPredictor gate smoke test ===')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    dp = DeployedLowPredictor(device=dev, use_global=True)

    # 1) LENGTH gate: short flight -> no online learning
    h = torch.randn(1, 40, 6) * 0.3
    r = dp.predict(h, drone_id='d1', frames_seen=30)   # below 60
    print(f'  short flight (30f): online_active={r["online_active"]} '
          f'length_gate_open={r["length_gate_open"]} (expect False/False)')

    # 2) LENGTH gate open: long flight -> online learning on
    r = dp.predict(h, drone_id='d1', frames_seen=120)
    print(f'  long flight (120f): online_active={r["online_active"]} (expect True)')

    # 3) SPEED gate: low speed -> global base (fresh drone, no drone_id so no session lock)
    h_slow = torch.zeros(1, 40, 6); h_slow[:, :, 3] = 1.0   # 1 m/s
    r = dp.predict(h_slow, frames_seen=120)
    print(f'  slow (1 m/s): base={r["base"]} global_active={r["global_active"]} (expect global/True)')

    # 4) SPEED gate: fast -> plain base (global off)
    h_fast = torch.zeros(1, 40, 6); h_fast[:, :, 3] = 8.0   # 8 m/s
    r = dp.predict(h_fast, frames_seen=120)
    print(f'  fast (8 m/s): base={r["base"]} global_active={r["global_active"]} (expect plain/False)')

    # 5) GLOBAL toggle off
    dp2 = DeployedLowPredictor(device=dev, use_global=False)
    r = dp2.predict(h_slow, drone_id='d2', frames_seen=120)
    print(f'  use_global=False, slow: base={r["base"]} (expect plain)')

    import shutil
    shutil.rmtree('weights/online_adapters', ignore_errors=True)
    print('Done.')
