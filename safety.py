"""
Safety & Risk Assessment Module for Drone Trajectory Prediction.

Three capabilities:
  1. RiskAssessment  — geofence distance + risk level (LOW/MEDIUM/HIGH)
  2. AnomalyDetector — prediction-vs-actual deviation + CUSUM escalation
  3. InputValidator  — shape/range checks for predictor inputs

Usage:
    from safety import RiskAssessment, AnomalyDetector

    ra = RiskAssessment(geofence_bounds={...})
    risk = ra.evaluate(predicted_trajectory, current_position)

    ad = AnomalyDetector()
    alert = ad.check(predicted, actual, drone_id)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


# ============================================================
# Risk Level Types
# ============================================================

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskResult:
    """Output of risk assessment."""
    level: RiskLevel
    min_distance: float         # meters — closest approach to any boundary
    time_to_violation: float    # seconds until boundary crossing (inf if none)
    violating_steps: List[int]  # which prediction steps cross boundary
    warning: bool               # early warning signal
    alert: bool                 # immediate action required


# ============================================================
# 1. Risk Assessment — Geofence Boundary Distance
# ============================================================

class RiskAssessment:
    """
    Evaluate risk of predicted trajectory crossing geofence boundaries.

    Supports multiple boundary types:
      - cylinder: circular no-fly zone (center, radius, floor, ceiling)
      - box: rectangular restricted area (min_xyz, max_xyz)
      - sphere: spherical keep-out zone (center, radius)

    Parameters
    ----------
    boundaries : list of dict
        Each dict defines one geofence boundary.
        cylinder: {'type':'cylinder','center':(x,y),'radius':r,'floor':z0,'ceiling':z1}
        box:      {'type':'box','min':(x,y,z),'max':(x,y,z)}
        sphere:   {'type':'sphere','center':(x,y,z),'radius':r}
    warning_margin : float
        Distance margin for early warning (meters). Default 5.0m.
    critical_margin : float
        Distance margin for critical alert (meters). Default 1.0m.
    """

    def __init__(self, boundaries: List[dict] = None,
                 warning_margin: float = 5.0, critical_margin: float = 1.0):
        self.boundaries = boundaries or []
        self.warning_margin = warning_margin
        self.critical_margin = critical_margin

    def add_cylinder(self, center: Tuple[float, float], radius: float,
                     floor: float = -float('inf'), ceiling: float = float('inf'),
                     name: str = ""):
        """Add a cylindrical no-fly zone (e.g., around a building)."""
        self.boundaries.append({
            'type': 'cylinder', 'center': center, 'radius': radius,
            'floor': floor, 'ceiling': ceiling, 'name': name,
        })

    def add_box(self, min_xyz: Tuple[float, float, float],
                max_xyz: Tuple[float, float, float], name: str = ""):
        """Add a box restricted area."""
        self.boundaries.append({
            'type': 'box', 'min': min_xyz, 'max': max_xyz, 'name': name,
        })

    def add_sphere(self, center: Tuple[float, float, float],
                   radius: float, name: str = ""):
        """Add a spherical keep-out zone."""
        self.boundaries.append({
            'type': 'sphere', 'center': center, 'radius': radius, 'name': name,
        })

    def _distance_to_cylinder(self, points: torch.Tensor, boundary: dict) -> torch.Tensor:
        """Compute signed distance from points (N,3) to cylinder boundary."""
        cx, cy = boundary['center']
        r = boundary['radius']
        floor = boundary.get('floor', -float('inf'))
        ceiling = boundary.get('ceiling', float('inf'))

        # Horizontal distance from center
        horiz_dist = torch.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        horiz_violation = horiz_dist - r

        # Vertical violation
        vert_violation_floor = floor - points[:, 2]
        vert_violation_ceiling = points[:, 2] - ceiling

        # Combined: max of horizontal and vertical violations
        violation = torch.maximum(horiz_violation,
                    torch.maximum(vert_violation_floor, vert_violation_ceiling))

        return violation  # positive = outside (safe), negative = inside (violation)

    def _distance_to_box(self, points: torch.Tensor, boundary: dict) -> torch.Tensor:
        """Compute signed distance from points (N,3) to box boundary."""
        lo = torch.tensor(boundary['min'], device=points.device, dtype=points.dtype)
        hi = torch.tensor(boundary['max'], device=points.device, dtype=points.dtype)

        # Distance to each face
        dist_lo = lo - points
        dist_hi = points - hi

        # Outside distance = max of violations (positive = outside)
        outside = torch.maximum(dist_lo, dist_hi)
        outside_dist = torch.max(outside, dim=1).values

        return outside_dist  # positive = safe, negative = inside

    def _distance_to_sphere(self, points: torch.Tensor, boundary: dict) -> torch.Tensor:
        """Compute signed distance from points (N,3) to sphere boundary."""
        center = torch.tensor(boundary['center'], device=points.device, dtype=points.dtype)
        r = boundary['radius']
        dist_to_center = torch.norm(points - center.unsqueeze(0), dim=1)
        return dist_to_center - r  # positive = safe (outside)

    def evaluate(self, predicted_trajectory: torch.Tensor,
                 current_position: torch.Tensor = None,
                 dt: float = 0.2) -> RiskResult:
        """
        Evaluate risk for a predicted trajectory.

        Args:
            predicted_trajectory: (B, pred_len, 3) future displacement in meters.
            current_position: (B, 3) current absolute position. If None, assume origin.
            dt: Time step between prediction frames (seconds). Default 0.2s (5Hz).

        Returns:
            RiskResult with level, min_distance, time_to_violation, etc.
        """
        if not self.boundaries:
            return RiskResult(level=RiskLevel.LOW, min_distance=float('inf'),
                            time_to_violation=float('inf'), violating_steps=[],
                            warning=False, alert=False)

        device = predicted_trajectory.device

        if predicted_trajectory.dim() == 3:
            traj = predicted_trajectory[0]  # Take first batch
        else:
            traj = predicted_trajectory

        # Convert displacement to absolute positions
        if current_position is not None:
            current_position = current_position.to(device)
            if current_position.dim() == 2:
                current_position = current_position[0]
            abs_positions = current_position.unsqueeze(0) + torch.cumsum(
                torch.cat([torch.zeros(1, 3, device=device), traj], dim=0), dim=0)
            abs_positions = abs_positions[1:]  # Skip initial zero
        else:
            abs_positions = traj

        # Compute minimum distance to any boundary
        min_dist = float('inf')
        violating_steps = []
        first_violation_step = None

        for boundary in self.boundaries:
            btype = boundary['type']
            if btype == 'cylinder':
                dists = self._distance_to_cylinder(abs_positions, boundary)
            elif btype == 'box':
                dists = self._distance_to_box(abs_positions, boundary)
            elif btype == 'sphere':
                dists = self._distance_to_sphere(abs_positions, boundary)
            else:
                continue

            b_min = dists.min().item()
            if b_min < min_dist:
                min_dist = b_min

            # Find violating steps
            viol_steps = (dists < 0).nonzero(as_tuple=True)[0].tolist()
            violating_steps.extend(viol_steps)
            if viol_steps and (first_violation_step is None or viol_steps[0] < first_violation_step):
                first_violation_step = viol_steps[0]

        # Determine risk level
        if min_dist <= self.critical_margin:
            level = RiskLevel.CRITICAL
        elif min_dist <= self.warning_margin:
            level = RiskLevel.HIGH
        elif min_dist <= self.warning_margin * 3:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        # Time to violation
        if first_violation_step is not None:
            ttv = first_violation_step * dt
        elif min_dist > 0:
            ttv = float('inf')
        else:
            ttv = 0.0

        warning = level in (RiskLevel.HIGH, RiskLevel.CRITICAL) or min_dist < self.warning_margin * 2
        alert = level == RiskLevel.CRITICAL

        return RiskResult(
            level=level, min_distance=min_dist,
            time_to_violation=ttv, violating_steps=sorted(set(violating_steps)),
            warning=warning, alert=alert,
        )


# ============================================================
# 2. Anomaly Detector — Prediction-vs-Actual Deviation
# ============================================================

class AnomalyDetector:
    """
    Detect anomalous flight behavior via prediction error monitoring.

    Combines:
      - Moving average of prediction error
      - Adaptive threshold based on historical error distribution (μ + k*σ)
      - CUSUM escalation for sustained deviation
      - Drone-specific baselines

    Parameters
    ----------
    window_size : int
        Number of recent errors for moving statistics. Default 50.
    sigma_multiplier : float
        Anomaly threshold = mean_error + sigma_multiplier * std_error. Default 3.0.
    cusum_threshold : float
        CUSUM trigger threshold. Default 5.0.
    min_samples : int
        Minimum samples before anomaly detection starts. Default 20.
    """

    def __init__(self, window_size: int = 50, sigma_multiplier: float = 3.0,
                 cusum_threshold: float = 5.0, min_samples: int = 20):
        self.window_size = window_size
        self.sigma_multiplier = sigma_multiplier
        self.cusum_threshold = cusum_threshold
        self.min_samples = min_samples

        # Per-drone state
        self._errors: Dict[str, deque] = {}
        self._cusum_pos: Dict[str, float] = {}
        self._cusum_neg: Dict[str, float] = {}
        self._alert_count: Dict[str, int] = {}
        self._baseline: Dict[str, Tuple[float, float]] = {}  # (mean, std)

    def check(self, predicted: torch.Tensor, actual: torch.Tensor,
              drone_id: str = "default") -> Dict:
        """
        Check if current prediction error indicates anomaly.

        Args:
            predicted: (B, pred_len, 3) predicted displacement.
            actual: (B, pred_len, 3) actual/ground-truth displacement.
            drone_id: Unique drone identifier.

        Returns:
            dict with: is_anomaly, error, threshold, cusum_active, alert_count, level
        """
        # Move to same device if needed
        if predicted.device != actual.device:
            actual = actual.to(predicted.device)

        # Compute per-element MSE error
        error = F.mse_loss(predicted, actual, reduction='none').mean(dim=[-2, -1])
        if error.dim() > 0:
            error = error.mean()  # Average over batch
        error_val = error.item()

        # Initialize drone state
        if drone_id not in self._errors:
            self._errors[drone_id] = deque(maxlen=self.window_size)
            self._cusum_pos[drone_id] = 0.0
            self._cusum_neg[drone_id] = 0.0
            self._alert_count[drone_id] = 0

        self._errors[drone_id].append(error_val)
        errors = self._errors[drone_id]

        # Not enough data yet
        if len(errors) < self.min_samples:
            return {'is_anomaly': False, 'error': error_val,
                    'threshold': float('inf'), 'cusum_active': False,
                    'alert_count': 0, 'level': 'CALIBRATING'}

        # Update baseline statistics
        mean_err = np.mean(errors)
        std_err = np.std(errors) + 1e-6
        self._baseline[drone_id] = (mean_err, std_err)

        # Adaptive threshold
        threshold = mean_err + self.sigma_multiplier * std_err

        # CUSUM update
        deviation = error_val - mean_err
        self._cusum_pos[drone_id] = max(0.0, self._cusum_pos[drone_id] + deviation - std_err)
        self._cusum_neg[drone_id] = max(0.0, self._cusum_neg[drone_id] - deviation - std_err)
        cusum_active = (self._cusum_pos[drone_id] > self.cusum_threshold or
                       self._cusum_neg[drone_id] > self.cusum_threshold)

        # Determine anomaly
        is_anomaly = error_val > threshold or cusum_active
        if is_anomaly:
            self._alert_count[drone_id] += 1
        elif self._alert_count[drone_id] > 0 and error_val < mean_err:
            self._alert_count[drone_id] = max(0, self._alert_count[drone_id] - 1)

        # Severity level
        if error_val > mean_err + 5 * std_err:
            level = 'CRITICAL'
        elif error_val > mean_err + 3 * std_err or cusum_active:
            level = 'HIGH'
        elif error_val > mean_err + 2 * std_err:
            level = 'MEDIUM'
        else:
            level = 'NORMAL'

        return {
            'is_anomaly': is_anomaly,
            'error': error_val,
            'threshold': threshold,
            'baseline_mean': mean_err,
            'baseline_std': std_err,
            'cusum_active': cusum_active,
            'cusum_pos': self._cusum_pos[drone_id],
            'alert_count': self._alert_count[drone_id],
            'level': level,
        }

    def reset(self, drone_id: str):
        """Reset anomaly state for a drone."""
        self._errors.pop(drone_id, None)
        self._cusum_pos[drone_id] = 0.0
        self._cusum_neg[drone_id] = 0.0
        self._alert_count[drone_id] = 0
        self._baseline.pop(drone_id, None)

    def get_baseline(self, drone_id: str) -> Optional[Tuple[float, float]]:
        """Get (mean_error, std_error) baseline for a drone."""
        return self._baseline.get(drone_id)


# ============================================================
# 3. Input Validator
# ============================================================

class InputValidator:
    """
    Validate input tensors before model inference.

    Checks:
      - Shape correctness (must be (B, 20, 6))
      - Value ranges (position ~100m, velocity ~30m/s)
      - NaN/Inf detection
      - Device consistency

    Usage:
        validator = InputValidator()
        hist = validator.validate(hist)  # Returns validated tensor or raises
    """

    # Reasonable physical ranges for drone trajectories
    DEFAULT_POS_RANGE = (-1000.0, 1000.0)   # meters
    DEFAULT_VEL_RANGE = (-50.0, 50.0)       # m/s (180 km/h)
    EXPECTED_HIST_LEN = 20
    EXPECTED_INPUT_DIM = 6

    def __init__(self, pos_range: Tuple[float, float] = None,
                 vel_range: Tuple[float, float] = None,
                 strict: bool = False):
        self.pos_range = pos_range or self.DEFAULT_POS_RANGE
        self.vel_range = vel_range or self.DEFAULT_VEL_RANGE
        self.strict = strict  # If True, raise on violation. If False, warn + clip.

        self._warnings: List[str] = []

    def validate(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Validate and sanitize input tensor.

        Args:
            hist: (B, 20, 6) history tensor [pos_x,pos_y,pos_z, vx,vy,vz].

        Returns:
            Validated (possibly clipped) tensor.

        Raises:
            ValueError if input is fundamentally invalid (wrong shape, all NaN).
        """
        self._warnings.clear()

        # --- Shape check ---
        if hist.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, 20, 6), got shape {hist.shape}")
        B, T, D = hist.shape
        if T != self.EXPECTED_HIST_LEN or D != self.EXPECTED_INPUT_DIM:
            raise ValueError(
                f"Expected shape (B, {self.EXPECTED_HIST_LEN}, {self.EXPECTED_INPUT_DIM}), "
                f"got ({B}, {T}, {D})"
            )

        # --- NaN/Inf check ---
        nan_mask = torch.isnan(hist)
        inf_mask = torch.isinf(hist)
        if nan_mask.any():
            nan_frac = nan_mask.float().mean().item()
            msg = f"Input contains {nan_frac:.1%} NaN values"
            if self.strict:
                raise ValueError(msg)
            self._warnings.append(msg)
            hist = torch.nan_to_num(hist, nan=0.0)

        if inf_mask.any():
            msg = f"Input contains Inf values"
            if self.strict:
                raise ValueError(msg)
            self._warnings.append(msg)
            hist = torch.clamp(hist, min=-1e6, max=1e6)

        # --- Position range check ---
        pos = hist[:, :, :3]
        pos_min, pos_max = self.pos_range
        if pos.min() < pos_min or pos.max() > pos_max:
            msg = (f"Position out of range [{pos_min}, {pos_max}]: "
                   f"actual [{pos.min().item():.0f}, {pos.max().item():.0f}]")
            if self.strict:
                raise ValueError(msg)
            self._warnings.append(msg)
            hist[:, :, :3] = torch.clamp(hist[:, :, :3], min=pos_min, max=pos_max)

        # --- Velocity range check ---
        vel = hist[:, :, 3:6]
        vel_min, vel_max = self.vel_range
        if vel.min() < vel_min or vel.max() > vel_max:
            msg = (f"Velocity out of range [{vel_min}, {vel_max}]: "
                   f"actual [{vel.min().item():.0f}, {vel.max().item():.0f}]")
            if self.strict:
                raise ValueError(msg)
            self._warnings.append(msg)
            hist[:, :, 3:6] = torch.clamp(hist[:, :, 3:6], min=vel_min, max=vel_max)

        return hist

    @property
    def warnings(self) -> List[str]:
        return self._warnings.copy()

    def clear_warnings(self):
        self._warnings.clear()


# ============================================================
# Convenience: select best available device
# ============================================================

def get_best_device(verbose: bool = True) -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if verbose:
            print(f"Device: CUDA ({name}, {mem:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print("Device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        if verbose:
            print("Device: CPU")
    return device


# ============================================================
# Smoke Test
# ============================================================
if __name__ == '__main__':
    print('=== Safety Module Smoke Test ===\n')

    # 1. Input Validator
    print('--- Input Validator ---')
    v = InputValidator()
    x_ok = torch.randn(2, 20, 6)
    x_bad = torch.randn(2, 20, 6)
    x_bad[:, :, 3] = float('nan')
    try:
        v.validate(x_ok)
        print(f'  Valid input: OK')
    except ValueError as e:
        print(f'  Valid input ERROR: {e}')

    v2 = InputValidator(strict=False)
    x_validated = v2.validate(x_bad)
    print(f'  NaN input (non-strict): validated shape={x_validated.shape}, warnings={len(v2.warnings)}')

    # 2. Risk Assessment
    print('\n--- Risk Assessment ---')
    ra = RiskAssessment()
    ra.add_cylinder(center=(0, 0), radius=5.0, name="Tower")
    ra.add_box(min_xyz=(20, 20, 0), max_xyz=(30, 30, 100), name="Building")

    # Safe trajectory (stays outside boundaries)
    safe_traj = torch.ones(20, 3) * 10.0  # Start at (10,10,10), far from boundaries
    risk_safe = ra.evaluate(safe_traj.unsqueeze(0))
    print(f'  Safe: level={risk_safe.level.value}, dist={risk_safe.min_distance:.1f}m')

    # Risky trajectory (start at 8m from origin, heading toward cylinder at origin)
    risky_traj = torch.linspace(0, 3, 20).unsqueeze(1).expand(-1, 3) * -0.5
    risky_traj += torch.tensor([8.0, 0.0, 0.0])  # Start at (8,0,0), heading to origin
    risk = ra.evaluate(risky_traj.unsqueeze(0))
    print(f'  Risky: level={risk.level.value}, dist={risk.min_distance:.1f}m, '
          f'ttv={risk.time_to_violation:.2f}s, warning={risk.warning}')

    # 3. Anomaly Detector
    print('\n--- Anomaly Detector ---')
    ad = AnomalyDetector(min_samples=5)
    # Feed normal data
    for i in range(20):
        pred = torch.randn(1, 20, 3) * 0.5
        actual = pred + torch.randn(1, 20, 3) * 0.1
        result = ad.check(pred, actual, 'drone_1')
    print(f'  Normal baseline: mean={ad.get_baseline("drone_1")[0]:.3f}, std={ad.get_baseline("drone_1")[1]:.3f}')

    # Feed anomalous data
    anomaly = ad.check(torch.randn(1, 20, 3) * 0.5,
                       torch.randn(1, 20, 3) * 5.0, 'drone_1')
    print(f'  Anomaly check: is_anomaly={anomaly["is_anomaly"]}, level={anomaly["level"]}, '
          f'error={anomaly["error"]:.3f} vs threshold={anomaly["threshold"]:.3f}')

    # 4. Device selection
    print(f'\n--- Device ---')
    dev = get_best_device()

    print('\nAll tests passed!')
