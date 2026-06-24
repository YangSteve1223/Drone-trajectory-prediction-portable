"""
Online Continual Learner for per-drone LoRA adaptation with replay buffer and CUSUM escalation.
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


@dataclass
class OnlineLearnerConfig:
    """Hyperparameters for online continual learning."""

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    # Accumulation
    accumulation_steps: int = 5
    replay_ratio: float = 0.4

    # Regularization
    lora_l2_penalty: float = 0.01

    # Confidence gating
    conf_threshold: float = 0.75
    conf_threshold_escalated: float = 0.5
    lr_escalation_factor: float = 3.0

    # CUSUM
    cusum_threshold: float = 3.0
    cusum_drift: float = 1.0

    # Persistence
    save_every: int = 10

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class OnlineLearner:
    """
    Orchestrates per-drone online learning with optimizer, accumulation buffer,
    replay buffer, and CUSUM degradation monitoring.
    """

    def __init__(self, adapter_manager: DroneAdapterManager,
                 config: OnlineLearnerConfig = None):
        self.mgr = adapter_manager
        self.config = config or OnlineLearnerConfig()
        self.device = torch.device(self.config.device)

        # Per-drone state (not persisted to disk, rebuilt on load)
        self._optimizers: Dict[str, optim.AdamW] = {}
        self._buffers: Dict[str, List[TrajectorySample]] = {}
        self._replay: Dict[str, ReplayBuffer] = {}
        self._cusum: Dict[str, CUSUMDetector] = {}
        self._update_counters: Dict[str, int] = {}
        self._escalated: Dict[str, bool] = {}

    def observe(self, drone_id: str, history: torch.Tensor,
                future_gt: torch.Tensor, confidence: float = 1.0,
                intent: int = 0, timestep: int = 0) -> bool:
        """Record a trajectory observation. Triggers update when buffer reaches K."""
        threshold = self._get_threshold(drone_id)
        if confidence < threshold:
            return False

        self._ensure_state(drone_id)

        sample = TrajectorySample(
            history=history.detach().cpu(),
            future_gt=future_gt.detach().cpu(),
            intent=intent, timestep=timestep,
            confidence=confidence,
        )

        self._buffers[drone_id].append(sample)

        if len(self._buffers[drone_id]) >= self.config.accumulation_steps:
            loss = self._update(drone_id)
            return True
        return False

    def force_update(self, drone_id: str) -> Optional[float]:
        """Force an update regardless of accumulation count."""
        if len(self._buffers.get(drone_id, [])) == 0:
            return None
        return self._update(drone_id)

    def _update(self, drone_id: str) -> Optional[float]:
        """Execute one gradient update step: build batch, forward, backward, update buffers."""
        cfg = self.config
        buffer = self._buffers[drone_id]
        replay = self._replay[drone_id]
        cusum = self._cusum[drone_id]

        if len(buffer) == 0:
            return None

        # Build batch: K current + M replay samples
        k_current = min(len(buffer), cfg.accumulation_steps)
        m_replay = min(int(k_current * cfg.replay_ratio), len(replay))
        current_samples = buffer[:k_current]
        replay_samples = replay.sample(m_replay) if m_replay > 0 else []

        # Activate adapter and rebind optimizer params
        was_active = (self.mgr.active_drone == drone_id)
        if not was_active:
            self.mgr.activate(drone_id)
            if drone_id in self._optimizers:
                trainable = self.mgr.adapter.get_trainable_params()
                self._optimizers[drone_id].param_groups[0]['params'] = trainable

        # Forward pass
        histories = torch.stack([s.history for s in current_samples + replay_samples])
        futures = torch.stack([s.future_gt for s in current_samples + replay_samples])

        model_device = next(self.mgr.base_model.parameters()).device
        histories = histories.to(model_device)
        futures = futures.to(model_device)

        self.mgr.adapter.model.eval()

        out = self.mgr.adapter.model(histories, force_predict=True)
        predictions = out['predictions']

        # MSE loss + L2 penalty on LoRA params
        mse_loss = F.mse_loss(predictions, futures)

        l2_penalty = torch.tensor(0.0, device=model_device)
        for layer in self.mgr.adapter.lora_layers.values():
            l2_penalty += layer.lora_A.norm() ** 2 + layer.lora_B.norm() ** 2

        loss = mse_loss + cfg.lora_l2_penalty * l2_penalty

        # NaN protection
        if torch.isnan(loss) or torch.isinf(loss):
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

        # Update replay buffer with current samples
        for s in current_samples:
            replay.push(s)

        buffer.clear()

        # Update CUSUM with prediction error
        with torch.no_grad():
            pred_err = F.mse_loss(predictions[:k_current], futures[:k_current]).item()
        escalated = cusum.update(pred_err)
        self._escalated[drone_id] = escalated

        # Persist periodically
        self._update_counters[drone_id] += 1
        if self._update_counters[drone_id] % cfg.save_every == 0:
            self.mgr.save(drone_id, replay_buffer=replay, cusum=cusum)

        if not was_active:
            self.mgr.deactivate()

        return loss_val

    def _ensure_state(self, drone_id: str):
        """Initialize per-drone state on first observation."""
        if drone_id not in self._optimizers:
            self.mgr.activate(drone_id)

            trainable = self.mgr.adapter.get_trainable_params()
            self._optimizers[drone_id] = optim.AdamW(
                trainable, lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

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
                    has_saved = False

            if not has_saved:
                self._replay[drone_id] = ReplayBuffer(capacity=20)
                self._cusum[drone_id] = CUSUMDetector(
                    threshold=self.config.cusum_threshold,
                    drift=self.config.cusum_drift,
                )
                self._update_counters[drone_id] = 0
                self.mgr.save(drone_id)

            self._buffers[drone_id] = []
            self._escalated[drone_id] = False

            self.mgr.deactivate()

    def _get_threshold(self, drone_id: str) -> float:
        """Get confidence threshold (lowered during CUSUM escalation)."""
        if self._escalated.get(drone_id, False):
            return self.config.conf_threshold_escalated
        return self.config.conf_threshold

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

    model = model.cpu()
    mgr = DroneAdapterManager(model, checkpoint_dir='test_adapters_ol')
    learner = OnlineLearner(mgr)

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

    shutil.rmtree('test_adapters_ol', ignore_errors=True)
    print('\nSmoke test complete!')
