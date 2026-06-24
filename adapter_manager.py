"""
Drone Adapter Manager for per-drone LoRA state management, replay buffer, and CUSUM monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
import time
import os

from lora import LoRAAdapter, DEFAULT_LORA_TARGETS, DEFAULT_HEAD_TARGETS


@dataclass
class TrajectorySample:
    """Single trajectory observation for online learning."""
    history: torch.Tensor      # (20, 6)
    future_gt: torch.Tensor    # (20, 3)
    intent: int = 0
    timestep: int = 0
    confidence: float = 1.0

    def to(self, device):
        self.history = self.history.to(device)
        self.future_gt = self.future_gt.to(device)
        return self


class ReplayBuffer:
    """Fixed-capacity FIFO buffer for experience replay."""

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, sample: TrajectorySample):
        self.buffer.append(sample)

    def sample(self, n: int) -> List[TrajectorySample]:
        """Random sample n items (or fewer if buffer is smaller)."""
        import random
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)

    def to_list(self) -> List[Dict]:
        """Serialize for checkpoint."""
        return [{'history': s.history.cpu(), 'future_gt': s.future_gt.cpu(),
                 'intent': s.intent, 'timestep': s.timestep,
                 'confidence': s.confidence}
                for s in self.buffer]

    @classmethod
    def from_list(cls, data: List[Dict], capacity: int = 20):
        buf = cls(capacity)
        for d in data:
            buf.push(TrajectorySample(
                history=d['history'], future_gt=d['future_gt'],
                intent=d.get('intent', 0), timestep=d.get('timestep', 0),
                confidence=d.get('confidence', 1.0),
            ))
        return buf


class CUSUMDetector:
    """
    Two-sided CUSUM for detecting sustained prediction degradation.
    Accumulates when error exceeds baseline; triggers escalation.
    """

    def __init__(self, threshold: float = 3.0, drift: float = 1.0,
                 window: int = 50):
        self.threshold = threshold
        self.drift = drift
        self.window = window
        self.reset()

    def reset(self):
        self.pos_sum = 0.0
        self.neg_sum = 0.0
        self.error_history: deque = deque(maxlen=self.window)
        self.mean_error = 0.0
        self.n_updates = 0
        self.triggered = False

    def update(self, error: float) -> bool:
        """Update CUSUM with new prediction error. Returns True if triggered."""
        self.error_history.append(error)
        if len(self.error_history) < 5:
            return False

        # Baseline: running mean of recent errors
        self.mean_error = sum(self.error_history) / len(self.error_history)

        # CUSUM update: deviation = error - mean_error
        deviation = error - self.mean_error
        self.pos_sum = max(0.0, self.pos_sum + deviation - self.drift)
        self.neg_sum = max(0.0, self.neg_sum - deviation - self.drift)
        self.n_updates += 1

        triggered = self.pos_sum > self.threshold or self.neg_sum > self.threshold
        if triggered and not self.triggered:
            self.triggered = True
        if not triggered:
            self.triggered = False

        return triggered

    @property
    def active(self) -> bool:
        return self.triggered

    def state_dict(self) -> Dict:
        return {
            'pos_sum': self.pos_sum, 'neg_sum': self.neg_sum,
            'error_history': list(self.error_history),
            'mean_error': self.mean_error, 'n_updates': self.n_updates,
            'triggered': self.triggered,
        }

    def load_state_dict(self, d: Dict):
        self.pos_sum = d['pos_sum']
        self.neg_sum = d['neg_sum']
        self.error_history = deque(d['error_history'], maxlen=self.window)
        self.mean_error = d['mean_error']
        self.n_updates = d['n_updates']
        self.triggered = d['triggered']


@dataclass
class DroneState:
    """All persisted state for one drone."""
    drone_id: str
    lora_state: Dict = field(default_factory=dict)
    head_state: Dict = field(default_factory=dict)
    optimizer_state: Dict = field(default_factory=dict)
    replay_buffer_data: List[Dict] = field(default_factory=list)
    cusum_state: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class DroneAdapterManager:
    """Manages LoRA adapters for multiple drones: create, load, save, delete."""

    def __init__(self, base_model: nn.Module,
                 checkpoint_dir: str = 'checkpoints/adapters',
                 r: int = 4, alpha: float = 4.0,
                 lora_targets: List[str] = None,
                 head_targets: List[str] = None):
        self.base_model = base_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.r = r
        self.alpha = alpha
        self.lora_targets = lora_targets or DEFAULT_LORA_TARGETS
        self.head_targets = head_targets or DEFAULT_HEAD_TARGETS

        self._active_drone: Optional[str] = None
        self._adapter: Optional[LoRAAdapter] = None

        self._index_path = self.checkpoint_dir / 'index.json'
        self._index: Dict[str, Dict] = self._load_index()

    def _load_index(self) -> Dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def _drone_path(self, drone_id: str) -> Path:
        return self.checkpoint_dir / f'{drone_id}.pt'

    def activate(self, drone_id: str) -> bool:
        """Apply this drone's LoRA adapter to the base model. Returns True if newly created."""
        if self._active_drone is not None:
            self.deactivate()

        is_new = not self._drone_path(drone_id).exists()

        self._adapter = LoRAAdapter(
            self.base_model, r=self.r, alpha=self.alpha,
            lora_targets=self.lora_targets, head_targets=self.head_targets,
        )
        self._adapter.activate()

        if not is_new:
            self.load(drone_id)

        self._active_drone = drone_id

        now = time.strftime('%Y-%m-%dT%H:%M:%S')
        if drone_id not in self._index:
            self._index[drone_id] = {'created': now, 'num_updates': 0}
        self._index[drone_id]['last_active'] = now
        self._save_index()

        return is_new

    def deactivate(self):
        """Remove LoRA adapter and restore base model."""
        if self._adapter is not None:
            self._adapter.deactivate()
            self._adapter = None
        self._active_drone = None

    @property
    def active_drone(self) -> Optional[str]:
        return self._active_drone

    @property
    def adapter(self) -> Optional[LoRAAdapter]:
        return self._adapter

    def save(self, drone_id: str, optimizer=None, replay_buffer=None,
             cusum=None):
        """Save drone adapter state to disk."""
        if self._adapter is None:
            raise RuntimeError("No adapter active. Call activate() first.")

        data = {
            'drone_id': drone_id,
            'lora_state': self._adapter.get_lora_state(),
            'head_state': self._adapter.get_head_state(),
            'version': 1,
        }

        if optimizer is not None:
            data['optimizer_state'] = optimizer.state_dict()
        if replay_buffer is not None:
            data['replay_buffer'] = replay_buffer.to_list()
        if cusum is not None:
            data['cusum_state'] = cusum.state_dict()

        torch.save(data, self._drone_path(drone_id))

        self._index[drone_id]['last_saved'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        if 'num_updates' in self._index[drone_id]:
            self._index[drone_id]['num_updates'] += 1
        self._save_index()

    def load(self, drone_id: str):
        """Load drone adapter state from disk into active adapter."""
        if self._adapter is None:
            raise RuntimeError("No adapter active. Call activate() first.")

        path = self._drone_path(drone_id)
        if not path.exists():
            raise FileNotFoundError(f"No adapter found for {drone_id}: {path}")

        data = torch.load(path, map_location='cpu')
        self._adapter.load_lora_state(data['lora_state'])
        self._adapter.load_head_state(data['head_state'])

    def load_full_state(self, drone_id: str) -> DroneState:
        """Load all state for a drone (adapter + optimizer + replay + cusum)."""
        path = self._drone_path(drone_id)
        if not path.exists():
            raise FileNotFoundError(f"No adapter found for {drone_id}: {path}")

        data = torch.load(path, map_location='cpu')
        return DroneState(
            drone_id=drone_id,
            lora_state=data.get('lora_state', {}),
            head_state=data.get('head_state', {}),
            optimizer_state=data.get('optimizer_state', {}),
            replay_buffer_data=data.get('replay_buffer', []),
            cusum_state=data.get('cusum_state', {}),
            metadata=self._index.get(drone_id, {}),
        )

    def delete(self, drone_id: str):
        """Delete a drone's adapter from disk."""
        if self._active_drone == drone_id:
            self.deactivate()
        path = self._drone_path(drone_id)
        if path.exists():
            path.unlink()
        self._index.pop(drone_id, None)
        self._save_index()

    def list_drones(self) -> List[Dict]:
        """List all tracked drones with metadata."""
        result = []
        for drone_id, meta in self._index.items():
            entry = {'drone_id': drone_id, **meta}
            entry['on_disk'] = self._drone_path(drone_id).exists()
            try:
                entry['size_kb'] = self._drone_path(drone_id).stat().st_size / 1024
            except OSError:
                entry['size_kb'] = 0
            result.append(entry)
        return result

    def has_adapter(self, drone_id: str) -> bool:
        return self._drone_path(drone_id).exists()

    def num_drones(self) -> int:
        return len(self._index)

    def get_trainable_params(self) -> List[nn.Parameter]:
        if self._adapter is None:
            raise RuntimeError("No adapter active.")
        return self._adapter.get_trainable_params()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from emam_model import TrajectoryPredictor

    print('=== Adapter Manager Smoke Test ===')

    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).eval()

    mgr = DroneAdapterManager(model, checkpoint_dir='test_adapters')

    # CRUD
    print('\n--- CRUD ---')
    for drone_id in ['drone_A', 'drone_B']:
        is_new = mgr.activate(drone_id)
        print(f'  activate({drone_id}): new={is_new}, active={mgr.active_drone}')
        mgr.save(drone_id)
        mgr.deactivate()

    # Load and verify
    mgr.activate('drone_A')
    print(f'  Loaded drone_A: adapter params={mgr.adapter.num_params}')
    mgr.deactivate()

    # List drones
    drones = mgr.list_drones()
    for d in drones:
        print(f'  {d["drone_id"]}: created={d.get("created","?")}, on_disk={d["on_disk"]}, size={d["size_kb"]:.1f}KB')

    # Replay buffer
    print('\n--- Replay Buffer ---')
    buf = ReplayBuffer(capacity=20)
    for i in range(25):
        buf.push(TrajectorySample(
            history=torch.randn(20, 6),
            future_gt=torch.randn(20, 3),
            timestep=i,
        ))
    print(f'  Buffer size: {len(buf)} (cap=20)')
    samples = buf.sample(3)
    print(f'  Sampled {len(samples)} trajectories')

    # CUSUM
    print('\n--- CUSUM ---')
    cusum = CUSUMDetector(threshold=3.0)
    triggered = False
    for i in range(100):
        error = 1.5 if i < 70 else 3.5
        triggered = cusum.update(error)
        if triggered:
            print(f'  CUSUM triggered at step {i} (error={error})')
            break
    if not triggered:
        print(f'  CUSUM not triggered (error_history={len(cusum.error_history)})')

    import shutil
    shutil.rmtree('test_adapters', ignore_errors=True)
    print('\nAll tests passed!')
