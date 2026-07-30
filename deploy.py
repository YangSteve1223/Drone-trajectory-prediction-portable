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
                      (dir_lora_40) on/off entirely. DISABLED by default (global
                      LoRA was trained on the full window set = data leakage;
                      cross-flight generalization is only +7.3% FDE).
  3) SPEED gate     — even with use_global=True, the global LoRA is disabled when
                      current speed > global_max_speed (it was trained on 0-3 m/s
                      real DJI flight; faster motion is out of its domain).

  4) DIRECTION fallback — if the predicted final displacement direction deviates
     >90 deg from the constant-velocity baseline, fall back to const-vel. This
     catches the rare catastrophic failures (~1.1% of windows) that per-drone
     LoRA cannot rescue.

Multi-head (K=5, WTA): opt-in via use_multihead=True. Uses the pre-trained
K=5 decoder; at inference the highest-confidence hypothesis is selected.
When combined with the direction fallback, catastrophic windows drop to near zero.
"""

import torch
import torch.nn.functional as F
import warnings
import math
from pathlib import Path

import online_config as OC
from adapter_manager import DroneAdapterManager
from online_learner import OnlineLearner, OnlineLearnerConfig


class DeployedLowPredictor:
    def __init__(self, device=None,
                 use_global: bool = False,
                 use_multihead: bool = False,
                 global_max_speed: float = OC.GLOBAL_MAX_SPEED,
                 online_min_frames: int = OC.ONLINE_MIN_FRAMES,
                 checkpoint_dir: str = 'weights/online_adapters',
                 accumulation_steps: int = 10,
                 direction_fallback: bool = True,
                 direction_threshold_deg: float = 90.0):
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_global = use_global
        self.use_multihead = use_multihead
        self.global_max_speed = global_max_speed
        self.online_min_frames = online_min_frames
        self.direction_fallback = direction_fallback
        self.direction_threshold_deg = direction_threshold_deg
        self.pred_len = OC.PRED_LEN
        self.dt = 0.2

        # Base model(s). When use_global=False (default), only base_plain is built.
        # When use_global=True (opt-in), a second base_global carries the merged
        # dir_lora_40 weights, and _select_base() routes between them per the speed gate.
        if use_multihead:
            self.base_plain = OC.build_multihead_base(device=str(self.device), K=5)
        else:
            self.base_plain = OC.build_base_model(device=str(self.device))
        self.base_global = None
        if use_global:
            if use_multihead:
                self.base_global = OC.build_multihead_base(device=str(self.device), K=5)
            else:
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

    def _const_vel_prediction(self, hist):
        """Constant-velocity displacement prediction (meters), for direction fallback."""
        last_vel = hist[:, -1, 3:6]
        vel_recent = hist[:, -3:, 3:6]
        w = torch.tensor([0.2, 0.3, 0.5], device=hist.device)
        last_vel_smooth = (vel_recent * w.view(1, 3, 1)).sum(dim=1)
        step_indices = torch.arange(1, self.pred_len + 1, device=hist.device).float()
        step_indices = step_indices.view(1, -1, 1) * self.dt
        return last_vel_smooth.unsqueeze(1) * step_indices  # (B, pred_len, 3) meters

    def _apply_direction_fallback(self, predictions, hist):
        """If prediction direction deviates >threshold from const-vel, use const-vel."""
        cv_pred = self._const_vel_prediction(hist)
        pred_dir = predictions[:, -1, :2]
        cv_dir = cv_pred[:, -1, :2]
        cos_sim = F.cosine_similarity(pred_dir + 1e-8, cv_dir + 1e-8, dim=-1)
        angle = torch.acos(cos_sim.clamp(-0.999, 0.999)) * (180.0 / math.pi)
        mask = (angle > self.direction_threshold_deg).float()
        mask_3d = mask.unsqueeze(-1).unsqueeze(-1)
        safe_pred = predictions * (1 - mask_3d) + cv_pred * mask_3d
        return safe_pred, mask.bool()

    def _forward_model(self, model, hist):
        """Forward pass — uses multi_head path when the decoder supports it."""
        if self.use_multihead and hasattr(model.ua_pgd.neural_decoder, 'delta_heads'):
            scale = hist.new_tensor([100., 100., 100., 10., 10., 10.])
            h_norm = hist / scale.unsqueeze(0).unsqueeze(0)
            encoded = model.emam_se(h_norm)
            dtp_out = model.ia_dtp(encoded, historical_trajectory=h_norm)
            return model.ua_pgd.forward_multi_head(
                encoded_feat=encoded,
                global_anchor=dtp_out['global_anchor'],
                historical_trajectory=h_norm,
                intent_weights=dtp_out.get('intent_weights'),
            )
        else:
            return model(hist, force_predict=True)

    def predict(self, hist: torch.Tensor, drone_id: str = None,
                ground_truth: torch.Tensor = None,
                frames_seen: int = None, timestep: int = 0) -> dict:
        """Predict with the three gates + direction fallback.

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
        # (chosen by the gates the first time the length gate opens).
        adapted = False
        if drone_id and length_ok:
            if self._active_drone != drone_id:
                if self.mgr.active_drone is not None:
                    self.mgr.deactivate()
                self.mgr.base_model = base           # bind once
                self._session_base_tag = base_tag
                self._active_drone = drone_id
            adapted = True

        # Online learning step.
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
                model = self.mgr.adapter.model
            elif ground_truth is None and self.mgr.has_adapter(drone_id):
                self.mgr.activate(drone_id)
                model = self.mgr.adapter.model
            else:
                model = base
            eff_base_tag = getattr(self, '_session_base_tag', base_tag)
        else:
            model = base
            eff_base_tag = base_tag

        with torch.no_grad():
            out = self._forward_model(model, hist)

        predictions = out['predictions']

        # Direction fallback: if model points >90 deg away from const-vel, use const-vel.
        fell_back = None
        if self.direction_fallback:
            predictions, fell_back = self._apply_direction_fallback(predictions, hist)

        return {
            'predictions': predictions,
            'speed': speed_val,
            'base': eff_base_tag,
            'global_active': eff_base_tag == 'global',
            'online_active': adapted,
            'length_gate_open': length_ok,
            'updated': updated,
            'direction_fell_back': fell_back,
            'multihead_active': self.use_multihead,
            'all_predictions': out.get('all_predictions'),     # (K, B, P, 3) if multi-head
            'confidences': out.get('confidences'),             # (B, K) if multi-head
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

    # 3) SPEED gate: low speed -> global base
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

    # 6) Multi-head + direction fallback
    print('\n--- Multi-head + Direction Fallback ---')
    try:
        dp3 = DeployedLowPredictor(device=dev, use_multihead=True, direction_fallback=True)
        h_test = torch.randn(1, 40, 6) * 0.3
        r = dp3.predict(h_test, frames_seen=120)
        print(f'  multihead_active={r["multihead_active"]}')
        print(f'  predictions shape: {r["predictions"].shape}')
        print(f'  all_predictions shape: {r["all_predictions"].shape}')
        print(f'  confidences: {r["confidences"]}')
        print(f'  direction_fell_back: {r["direction_fell_back"]}')
    except Exception as e:
        print(f'  Multi-head test skipped: {e}')

    # 7) Direction fallback on deliberately bad history
    print('\n--- Direction Fallback: opposite-direction test ---')
    dp4 = DeployedLowPredictor(device=dev, use_multihead=False, direction_fallback=True,
                               direction_threshold_deg=30)  # low threshold to force fallback
    h_opp = torch.zeros(1, 40, 6)
    h_opp[0, :, 0] = torch.arange(40) * 0.2       # moving +X
    h_opp[0, :, 3] = 1.0                            # vx = 1 m/s
    r = dp4.predict(h_opp, frames_seen=10)
    fb = r['direction_fell_back']
    print(f'  fell_back={fb} (shape={fb.shape if fb is not None else None})')
    if fb is not None:
        print(f'  any fallback: {fb.any().item()}')

    import shutil
    shutil.rmtree('weights/online_adapters', ignore_errors=True)
    print('\nDone.')
