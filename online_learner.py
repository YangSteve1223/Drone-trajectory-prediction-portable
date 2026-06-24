"""
Online Continual Learner for per-drone LoRA adaptation.

Orchestrates the online learning loop:
  1. Accumulate confident trajectory observations
  2. When K observations collected: run gradient update
  3. Mix in replay buffer samples for catastrophic forgetting protection
  4. CUSUM escalation when prediction degrades
  5. Periodic persistence

Usage:
    learner = OnlineLearner(adapter_manager)
    learner.observe('drone_001', history, future_gt, confidence=0.85)
    # ... after K observations, update happens automatically
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time

from adapter_manager import (
    DroneAdapterManager, TrajectorySample,
    ReplayBuffer, CUSUMDetector,
)
from lora import LoRAAdapter


# ============================================================
# Configuration
# ============================================================

@dataclass
class OnlineLearnerConfig:
    """Hyperparameters for online continual learning."""

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    # Accumulation
    accumulation_steps: int = 5     # K: trajectories before one update
    replay_ratio: float = 0.4       # Fraction of batch from replay buffer

    # Regularization
    lora_l2_penalty: float = 0.01   # Pull LoRA weights toward zero

    # Confidence gating
    conf_threshold: float = 0.75    # Minimum confidence to accumulate
    conf_threshold_escalated: float = 0.5  # Lower during CUSUM escalation
    lr_escalation_factor: float = 3.0      # lr multiplier during escalation

    # CUSUM
    cusum_threshold: float = 3.0
    cusum_drift: float = 1.0

    # Persistence
    save_every: int = 10            # Save adapter every N updates

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Online Learner
# ============================================================

class OnlineLearner:
    """
    Orchestrates per-drone online learning.

    Maintains per-drone:
      - Optimizer (AdamW)
      - Accumulation buffer (pending observations)
      - Replay buffer (past trajectories)
      - CUSUM detector (degradation monitoring)
      - Update counter
    """

    def __init__(self, adapter_manager: DroneAdapterManager,
                 config: OnlineLearnerConfig = None):
        self.mgr = adapter_manager
        self.config = config or OnlineLearnerConfig()
        self.device = torch.device(self.config.device)

        # Per-drone state (not persisted to disk — rebuilt on load)
        self._optimizers: Dict[str, optim.AdamW] = {}
        self._buffers: Dict[str, List[TrajectorySample]] = {}       # accumulation
        self._replay: Dict[str, ReplayBuffer] = {}
        self._cusum: Dict[str, CUSUMDetector] = {}
        self._update_counters: Dict[str, int] = {}
        self._escalated: Dict[str, bool] = {}

    # ---- Observe & Update ----

    def observe(self, drone_id: str, history: torch.Tensor,
                future_gt: torch.Tensor, confidence: float = 1.0,
                intent: int = 0, timestep: int = 0) -> bool:
        """
        Record a trajectory observation for a drone.
        Triggers an update if accumulation buffer reaches K.

        Args:
            drone_id: Unique drone identifier.
            history: (20, 6) past trajectory.
            future_gt: (20, 3) ground truth future displacement.
            confidence: Model confidence at prediction time [0, 1].
            intent: Intent label (optional).
            timestep: Observation timestep (optional).

        Returns:
            True if an update was performed.
        """
        # Check confidence threshold
        threshold = self._get_threshold(drone_id)
        if confidence < threshold:
            return False

        # Ensure drone is active and has state
        self._ensure_state(drone_id)

        # Create sample
        sample = TrajectorySample(
            history=history.detach().cpu(),
            future_gt=future_gt.detach().cpu(),
            intent=intent, timestep=timestep,
            confidence=confidence,
        )

        # Accumulate
        self._buffers[drone_id].append(sample)

        # Check if we should update
        if len(self._buffers[drone_id]) >= self.config.accumulation_steps:
            loss = self._update(drone_id)
            return True
        return False

    def force_update(self, drone_id: str) -> Optional[float]:
        """Force an update regardless of accumulation count."""
        if len(self._buffers.get(drone_id, [])) == 0:
            return None
        return self._update(drone_id)

    # ---- Internal: Update Step ----

    def _update(self, drone_id: str) -> Optional[float]:
        """Execute one gradient update step for a drone."""
        cfg = self.config
        buffer = self._buffers[drone_id]
        replay = self._replay[drone_id]
        cusum = self._cusum[drone_id]

        if len(buffer) == 0:
            return None

        # 1. Build batch: K current + M replay samples
        k_current = min(len(buffer), cfg.accumulation_steps)
        m_replay = min(int(k_current * cfg.replay_ratio), len(replay))
        current_samples = buffer[:k_current]
        replay_samples = replay.sample(m_replay) if m_replay > 0 else []

        # 2. Activate adapter for this drone and rebind optimizer params
        was_active = (self.mgr.active_drone == drone_id)
        if not was_active:
            self.mgr.activate(drone_id)
            # Refresh optimizer param references (params recreated on each activate)
            if drone_id in self._optimizers:
                trainable = self.mgr.adapter.get_trainable_params()
                self._optimizers[drone_id].param_groups[0]['params'] = trainable

        # 3. Forward pass
        histories = torch.stack([s.history for s in current_samples + replay_samples])
        futures = torch.stack([s.future_gt for s in current_samples + replay_samples])

        # Get model's device (may differ from learner's config device)
        model_device = next(self.mgr.base_model.parameters()).device
        histories = histories.to(model_device)
        futures = futures.to(model_device)

        self.mgr.adapter.model.eval()  # Keep dropout off during adaptation

        out = self.mgr.adapter.model(histories, force_predict=True)
        predictions = out['predictions']

        # 4. Loss
        mse_loss = F.mse_loss(predictions, futures)

        # L2 penalty on LoRA params
        l2_penalty = torch.tensor(0.0, device=model_device)
        for layer in self.mgr.adapter.lora_layers.values():
            l2_penalty += layer.lora_A.norm() ** 2 + layer.lora_B.norm() ** 2

        loss = mse_loss + cfg.lora_l2_penalty * l2_penalty

        # 5. Backward with NaN protection
        if torch.isnan(loss) or torch.isinf(loss):
            # Reset buffer to avoid repeated NaN
            buffer.clear()
            if not was_active:
                self.mgr.deactivate()
            return None

        optimizer = self._optimizers[drone_id]

        # Escalated LR during CUSUM
        if self._escalated.get(drone_id, False):
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.lr * cfg.lr_escalation_factor
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.lr

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        trainable = self.mgr.adapter.get_trainable_params()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)

        # NaN gradient check
        has_nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in trainable
        )
        if has_nan_grad:
            optimizer.zero_grad()
            buffer.clear()
            if not was_active:
                self.mgr.deactivate()
            return None

        optimizer.step()

        loss_val = loss.item()

        # 6. Update replay buffer with current samples
        for s in current_samples:
            replay.push(s)

        # 7. Clear accumulation buffer
        buffer.clear()

        # 8. Update CUSUM with prediction error
        with torch.no_grad():
            pred_err = F.mse_loss(predictions[:k_current], futures[:k_current]).item()
        escalated = cusum.update(pred_err)
        self._escalated[drone_id] = escalated

        # 9. Persist periodically (adapter weights only; optimizer rebuilt on load)
        self._update_counters[drone_id] += 1
        if self._update_counters[drone_id] % cfg.save_every == 0:
            self.mgr.save(drone_id, replay_buffer=replay, cusum=cusum)

        # 10. Restore previous adapter state
        if not was_active:
            self.mgr.deactivate()

        return loss_val

    # ---- State Management ----

    def _ensure_state(self, drone_id: str):
        """Initialize per-drone state if first observation."""
        if drone_id not in self._optimizers:
            # Activate to create adapter, then deactivate
            self.mgr.activate(drone_id)

            # Create optimizer
            trainable = self.mgr.adapter.get_trainable_params()
            self._optimizers[drone_id] = optim.AdamW(
                trainable, lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

            # Try to restore saved state (may not exist for new drone)
            has_saved = self.mgr.has_adapter(drone_id)
            if has_saved:
                try:
                    full_state = self.mgr.load_full_state(drone_id)
                    if full_state.optimizer_state:
                        self._optimizers[drone_id].load_state_dict(
                            full_state.optimizer_state)
                    if full_state.replay_buffer_data:
                        self._replay[drone_id] = ReplayBuffer.from_list(
                            full_state.replay_buffer_data)
                    else:
                        self._replay[drone_id] = ReplayBuffer(capacity=20)
                    self._cusum[drone_id] = CUSUMDetector(
                        threshold=self.config.cusum_threshold,
                        drift=self.config.cusum_drift,
                    )
                    if full_state.cusum_state:
                        self._cusum[drone_id].load_state_dict(full_state.cusum_state)
                    self._update_counters[drone_id] = full_state.metadata.get(
                        'num_updates', 0)
                except Exception:
                    has_saved = False  # Fall through to fresh init

            if not has_saved:
                self._replay[drone_id] = ReplayBuffer(capacity=20)
                self._cusum[drone_id] = CUSUMDetector(
                    threshold=self.config.cusum_threshold,
                    drift=self.config.cusum_drift,
                )
                self._update_counters[drone_id] = 0
                # Save initial zero-state to disk so future activates can reload
                self.mgr.save(drone_id)

            self._buffers[drone_id] = []
            self._escalated[drone_id] = False

            self.mgr.deactivate()

    def _get_threshold(self, drone_id: str) -> float:
        """Get confidence threshold for a drone (may be escalated)."""
        if self._escalated.get(drone_id, False):
            return self.config.conf_threshold_escalated
        return self.config.conf_threshold

    # ---- Management ----

    def reset_drone(self, drone_id: str):
        """Reset all state for a drone (delete adapter, start fresh)."""
        if self.mgr.active_drone == drone_id:
            self.mgr.deactivate()
        self.mgr.delete(drone_id)
        self._optimizers.pop(drone_id, None)
        self._buffers.pop(drone_id, None)
        self._replay.pop(drone_id, None)
        self._cusum.pop(drone_id, None)
        self._update_counters.pop(drone_id, None)
        self._escalated.pop(drone_id, None)

    def save_all(self):
        """Save all active drone adapter weights to disk."""
        for drone_id in list(self._optimizers.keys()):
            if drone_id in self._replay and drone_id in self._cusum:
                self.mgr.activate(drone_id)
                self.mgr.save(drone_id,
                              replay_buffer=self._replay[drone_id],
                              cusum=self._cusum[drone_id])
                self.mgr.deactivate()

    def get_status(self, drone_id: str) -> Dict:
        """Get status for a drone."""
        return {
            'drone_id': drone_id,
            'buffer_size': len(self._buffers.get(drone_id, [])),
            'replay_size': len(self._replay.get(drone_id, ReplayBuffer())),
            'num_updates': self._update_counters.get(drone_id, 0),
            'escalated': self._escalated.get(drone_id, False),
            'has_adapter': self.mgr.has_adapter(drone_id),
            'cusum_active': self._cusum.get(drone_id, CUSUMDetector()).active,
        }

    def summary(self) -> str:
        drones = list(self._optimizers.keys())
        if not drones:
            return 'No drones tracked.'
        lines = [f'{len(drones)} drones tracked:']
        for d in sorted(drones):
            s = self.get_status(d)
            lines.append(f"  {d}: updates={s['num_updates']}, "
                         f"buf={s['buffer_size']}/{self.config.accumulation_steps}, "
                         f"replay={s['replay_size']}, "
                         f"escalated={s['escalated']}")
        return '\n'.join(lines)


# ============================================================
# Smoke Test
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from emam_model import TrajectoryPredictor
    import shutil

    print('=== Online Learner Smoke Test ===')

    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).eval()

    # Use CPU for smoke test (avoids CUDA OOM from training session)
    model = model.cpu()
    mgr = DroneAdapterManager(model, checkpoint_dir='test_adapters_ol')
    learner = OnlineLearner(mgr)

    # Simulate online learning for one drone
    drone = 'test_drone'
    losses = []

    print(f'\nSimulating online learning for {drone}...')
    for i in range(30):
        hist = torch.randn(20, 6)
        future_gt = torch.randn(20, 3) * 0.1
        updated = learner.observe(drone, hist, future_gt, confidence=0.9)

        if updated:
            status = learner.get_status(drone)
            losses.append(f'update#{status["num_updates"]}')
            print(f'  Step {i}: UPDATE (total={status["num_updates"]}, '
                  f'replay={status["replay_size"]}, escalated={status["escalated"]})')

    status = learner.get_status(drone)
    print(f'\nFinal: {learner.summary()}')

    # Cleanup
    shutil.rmtree('test_adapters_ol', ignore_errors=True)
    print('\nSmoke test complete!')
