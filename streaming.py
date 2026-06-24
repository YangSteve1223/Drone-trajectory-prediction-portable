"""
Streaming Trajectory Predictor with rolling history buffer for real-time frame-by-frame inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from collections import deque


class RollingBuffer:
    """Fixed-size rolling window for trajectory history."""

    def __init__(self, history_len: int = 20, input_dim: int = 6):
        self.history_len = history_len
        self.input_dim = input_dim
        self.buffer: deque = deque(maxlen=history_len)
        self._full = False

    def push(self, frame: torch.Tensor) -> bool:
        """Add a new frame. Returns True when buffer is full (ready to predict)."""
        if frame.dim() == 2:
            frame = frame.squeeze(0)
        self.buffer.append(frame.clone())
        if len(self.buffer) >= self.history_len:
            self._full = True
        return self._full

    def get_history(self) -> torch.Tensor:
        """Get full history as (1, history_len, input_dim)."""
        if len(self.buffer) < self.history_len:
            raise RuntimeError(
                f"Buffer not full: {len(self.buffer)}/{self.history_len} frames"
            )
        return torch.stack(list(self.buffer), dim=0).unsqueeze(0)

    def reset(self):
        self.buffer.clear()
        self._full = False

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        return self._full


class StreamingPredictor:
    """
    Real-time streaming trajectory predictor.
    Maintains a rolling 20-frame history buffer; produces a 20-frame future prediction on each new frame.
    """

    def __init__(self, predictor, history_len: int = 20):
        self.predictor = predictor
        self.history_len = history_len
        self.device = predictor.device

        self.buffer = RollingBuffer(history_len, input_dim=6)
        self._frame_count = 0
        self._last_prediction: Optional[torch.Tensor] = None
        self._prediction_age = 0

    def reset(self):
        """Reset state for a new flight episode."""
        self.buffer.reset()
        self._frame_count = 0
        self._last_prediction = None
        self._prediction_age = 0

    def update(self, frame: torch.Tensor,
               drone_id: str = None,
               ground_truth: torch.Tensor = None) -> Optional[Dict]:
        """Process one new frame and return prediction if buffer is ready."""
        self._frame_count += 1
        self._prediction_age += 1

        ready = self.buffer.push(frame)
        if not ready:
            return None

        history = self.buffer.get_history().to(self.device)

        # Use adaptation if available
        if drone_id and self.predictor.adaptation_enabled:
            gt_tensor = ground_truth.unsqueeze(0) if ground_truth is not None and ground_truth.dim() == 2 else ground_truth
            result = self.predictor.predict_with_adaptation(
                history, drone_id=drone_id, ground_truth=gt_tensor,
                timestep=self._frame_count,
            )
        else:
            result = self.predictor.predict(history)

        self._last_prediction = result['predictions']
        self._prediction_age = 0

        return result

    def predict_now(self) -> Optional[torch.Tensor]:
        """Get the most recent prediction without new data. Returns None if expired."""
        if self._last_prediction is None:
            return None
        if self._prediction_age >= 20:
            return None
        return self._last_prediction[:, self._prediction_age:, :]

    def get_state(self) -> Dict:
        """Export streaming state for checkpointing."""
        return {
            'frame_count': self._frame_count,
            'buffer': list(self.buffer.buffer),
            'last_prediction': self._last_prediction.clone() if self._last_prediction is not None else None,
            'prediction_age': self._prediction_age,
        }

    def load_state(self, state: Dict):
        """Restore streaming state from checkpoint."""
        self._frame_count = state['frame_count']
        self.buffer.buffer = deque(state['buffer'], maxlen=self.history_len)
        self.buffer._full = len(self.buffer.buffer) >= self.history_len
        self._last_prediction = state['last_prediction']
        self._prediction_age = state['prediction_age']

    @property
    def ready(self) -> bool:
        return self.buffer.ready

    @property
    def frame_count(self) -> int:
        return self._frame_count


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from predictor import DronePredictor

    print('=== Streaming Predictor Smoke Test ===')

    p = DronePredictor()
    sp = StreamingPredictor(p)

    # Simulate 30 frames at ~1.5 m/s
    pos = torch.zeros(3)
    preds = []
    for i in range(30):
        vel = torch.randn(3) * 0.5 + 1.5
        pos = pos + vel * 0.2  # 5Hz, dt=0.2s
        frame = torch.cat([pos.clone(), vel.clone()])

        result = sp.update(frame)
        if result is not None:
            preds.append(result)
            print(f"  Frame {i}: buffer ready, pred shape={result['predictions'].shape}, "
                  f"speed={result['speed'].item():.1f} m/s")

    print(f"\nPredictions made: {len(preds)} (frames 20-29, one per new frame)")
    print(f"Frame count: {sp.frame_count}")

    # Test reset
    sp.reset()
    print(f"After reset: ready={sp.ready}, frames={sp.frame_count}")

    print('\nAll good!')
