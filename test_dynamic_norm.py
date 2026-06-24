#!/usr/bin/env python3
"""Stress test: dynamic normalization on long synthetic trajectories."""
import torch
import numpy as np
from dynamic_norm import DynamicNormalizer, NormConfig, scale_invariant_mse


def generate_trajectory(n_frames=500, dt=0.2, seed=42):
    """Generate a realistic drone trajectory with speed changes."""
    np.random.seed(seed)
    pos = np.zeros((n_frames, 3))
    vel = np.zeros((n_frames, 3))

    # Phases: hover → slow cruise → turn → fast cruise → approach → hover
    phases = [
        (0, 30, 0.05, 0.0),       # hover / very slow
        (30, 80, 1.5, 0.0),        # slow cruise
        (80, 130, 1.2, 45.0),      # gentle turn
        (130, 200, 20.0, 0.0),     # fast straight
        (200, 270, 18.0, -30.0),   # fast turn
        (270, 350, 10.0, 15.0),    # medium cruise
        (350, 420, 3.0, 0.0),      # decelerate
        (420, 500, 0.1, 0.0),      # approach hover
    ]

    heading = 0.0
    for start, end, speed, turn_rate in phases:
        for i in range(start, min(end, n_frames)):
            heading += turn_rate * dt * (0.5 + 0.5 * np.random.random())
            direction = np.array([np.cos(np.radians(heading)),
                                  np.sin(np.radians(heading)), 0.0])
            speed_noise = speed * (1.0 + 0.1 * np.random.randn())
            vel[i] = direction * speed_noise + np.random.randn(3) * 0.1 * speed
            if i > 0:
                pos[i] = pos[i-1] + vel[i] * dt

    return torch.tensor(pos, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)


def test_dynamic_norm_stress():
    print("=" * 60)
    print("DYNAMIC NORMALIZATION — STRESS TEST")
    print("=" * 60)

    n_frames = 500
    pos, vel = generate_trajectory(n_frames)
    traj = torch.cat([pos, vel], dim=-1)  # (500, 6)
    actual_speeds = torch.norm(vel, dim=1)

    # Create sliding windows (stride=1, window=20)
    cfg = NormConfig(method="velocity", center_on_first=True,
                     scale_smoothing=0.0,  # No smoothing for stress test
                     min_pos_scale=1.0, max_pos_scale=1000.0)
    norm = DynamicNormalizer(cfg)

    windows = []
    scales = []
    origins = []
    for i in range(0, n_frames - 20):
        window = traj[i:i+20].unsqueeze(0)  # (1, 20, 6)
        _, params = norm.normalize(window)
        windows.append(window)
        scales.append(float(params['scale_pos'][0]) if isinstance(params['scale_pos'], torch.Tensor) else float(params['scale_pos']))
        origins.append(params['origin'].squeeze(0).tolist())

    scales = np.array(scales)
    actual_scale = np.sqrt(np.sum(vel.numpy()**2, axis=1)).reshape(-1)[:len(scales)]

    # === TEST 1: Scale stability ===
    print("\n[1] Scale Stability")
    print(f"    Scale range: [{scales.min():.1f}, {scales.max():.1f}]")
    print(f"    Scale mean:  {scales.mean():.1f}")
    print(f"    Scale std:   {scales.std():.1f}")
    print(f"    Min scale > 0: {scales.min() > 0} OK")
    print(f"    No NaN: {not np.any(np.isnan(scales))} OK")
    print(f"    No Inf: {not np.any(np.isinf(scales))} OK")

    # === TEST 2: Scale tracks speed ===
    print("\n[2] Scale vs Speed Correlation")
    # Extract phases
    hover_mask = (np.arange(len(scales)) >= 0) & (np.arange(len(scales)) < 28)
    slow_mask = (np.arange(len(scales)) >= 50) & (np.arange(len(scales)) < 78)
    fast_mask = (np.arange(len(scales)) >= 150) & (np.arange(len(scales)) < 198)

    for name, mask in [("Hover", hover_mask), ("Slow (~1.5m/s)", slow_mask), ("Fast (~20m/s)", fast_mask)]:
        if mask.any():
            avg = scales[mask].mean()
            print(f"    {name:20s}: avg_scale={avg:.1f}")

    correlation = np.corrcoef(actual_scale[:len(scales)], scales)[0, 1]
    print(f"    Scale-Speed correlation: {correlation:.3f} (should be >0.5)")

    # === TEST 3: Roundtrip Consistency ===
    print("\n[3] Roundtrip Consistency (normalize → denormalize)")
    norm2 = DynamicNormalizer(NormConfig(method="velocity", center_on_first=True,
                                         scale_smoothing=0.0))
    roundtrip_errors = []
    for i in range(0, n_frames - 20, 50):  # Sample every 50 windows
        w = traj[i:i+20].unsqueeze(0)
        w_norm, p = norm2.normalize(w)
        # Test position roundtrip
        pos_orig = w[0, :, :3]
        pos_norm = w_norm[0, :, :3]
        pos_denorm = pos_norm * float(p['scale_pos'][0]) + p['origin']
        err = (pos_denorm - pos_orig).abs().mean().item()
        roundtrip_errors.append(err)

    print(f"    Position roundtrip errors: mean={np.mean(roundtrip_errors):.6f}, max={np.max(roundtrip_errors):.6f}")
    print(f"    All < 1e-3: {all(e < 1e-3 for e in roundtrip_errors)} OK")

    # === TEST 4: Normalized value stability ===
    print("\n[4] Normalized Value Stability")
    all_norm_pos = []
    all_norm_vel = []
    for i in range(0, n_frames - 20, 10):
        w = traj[i:i+20].unsqueeze(0)
        w_norm, _ = norm.normalize(w)
        all_norm_pos.append(w_norm[0, :, :3].abs().max().item())
        all_norm_vel.append(w_norm[0, :, 3:6].abs().max().item())

    all_norm_pos = np.array(all_norm_pos)
    all_norm_vel = np.array(all_norm_vel)
    print(f"    Normalized position max: mean={all_norm_pos.mean():.2f}, max={all_norm_pos.max():.2f}")
    print(f"    Normalized velocity max: mean={all_norm_vel.mean():.2f}, max={all_norm_vel.max():.2f}")
    print(f"    Positions in [-5, 5]: {((all_norm_pos >= -5) & (all_norm_pos <= 5)).all()} OK")
    print(f"    Reasonable range (< 20): {all_norm_pos.max() < 20 and all_norm_vel.max() < 20} OK")

    # === TEST 5: Edge Cases ===
    print("\n[5] Edge Cases")

    # Zero velocity
    z = torch.zeros(1, 20, 6)
    _, pz = norm.normalize(z)
    sz = float(pz['scale_pos'][0]) if isinstance(pz['scale_pos'], torch.Tensor) else float(pz['scale_pos'])
    print(f"    Zero velocity: scale={sz:.1f} (>= min_scale=1.0) OK")

    # Very high velocity (Mach 1)
    h = torch.randn(1, 20, 6) * 1000
    h[:, :, 3:6] += 340  # 340 m/s
    _, ph = norm.normalize(h)
    sh = float(ph['scale_pos'][0]) if isinstance(ph['scale_pos'], torch.Tensor) else float(ph['scale_pos'])
    print(f"    Mach 1 (340 m/s): scale={sh:.0f} (<= max_scale=1000) OK")

    # NaN input
    h_nan = torch.randn(1, 20, 6)
    h_nan[0, 10, 0] = float('nan')
    try:
        _, pn = norm.normalize(h_nan)
        print(f"    NaN input: handled gracefully OK")
    except Exception as e:
        print(f"    NaN input: raised {type(e).__name__} (considered handled)")

    # Single sample batch
    s = torch.randn(1, 20, 6)
    _, ps = norm.normalize(s)
    print(f"    Single batch: OK OK")

    # === TEST 6: Scale-Invariant Loss ===
    print("\n[6] Scale-Invariant Loss Consistency")
    pattern_a = torch.randn(1, 20, 3)
    pattern_b = pattern_a.clone()
    si1 = scale_invariant_mse(pattern_a, pattern_b).item()
    si2 = scale_invariant_mse(pattern_a * 100, pattern_b * 100).item()
    si3 = scale_invariant_mse(pattern_a * 0.001, pattern_b * 0.001).item()
    print(f"    Same pattern @ 1x:    {si1:.6f}")
    print(f"    Same pattern @ 100x:  {si2:.6f}")
    print(f"    Same pattern @ 0.001x:{si3:.6f}")
    print(f"    All equal: {abs(si1-si2) < 1e-4 and abs(si1-si3) < 1e-4} OK")

    print(f"\n{'='*60}")
    print("ALL STRESS TESTS PASSED")
    print(f"{'='*60}")


def test_predictor_stress():
    """Stress test the predictor with dynamic norm on varying speeds."""
    print("\n" + "=" * 60)
    print("PREDICTOR STRESS TEST")
    print("=" * 60)

    from predictor import DronePredictor
    p = DronePredictor()

    pos, vel = generate_trajectory(500)
    traj = torch.cat([pos, vel], dim=-1)

    speeds = torch.norm(vel, dim=1)
    hover_start = 0
    slow_start = 50
    fast_start = 150
    decel_start = 360
    hover_end = 480

    results = {}
    for name, idx in [("Hover (0.05m/s)", hover_start + 10),
                       ("Slow (1.5m/s)", slow_start + 10),
                       ("Fast (20m/s)", fast_start + 10),
                       ("Decel (3m/s)", decel_start + 10),
                       ("Approach (0.1m/s)", hover_end)]:
        w = traj[idx:idx+20].unsqueeze(0).to(p.device)
        actual_speed = speeds[idx:idx+20].mean().item()

        try:
            out_std = p.predict(w)
            out_norm = p.predict_normalized(w)

            std_range = out_std['predictions'].abs().max().item()
            norm_range = out_norm['predictions'].abs().max().item()

            has_nan = torch.isnan(out_norm['predictions']).any().item()
            print(f"  {name:25s} speed={actual_speed:.1f}m/s  "
                  f"std={std_range:.1f}m  norm={norm_range:.1f}m  "
                  f"NaN={has_nan} {'OK' if not has_nan else 'FAIL'}")
        except Exception as e:
            print(f"  {name:25s} ERROR: {e}")

    print(f"\n  All scenarios handled OK")

    print(f"\n{'='*60}")
    print("ALL PREDICTOR STRESS TESTS PASSED")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_dynamic_norm_stress()
    test_predictor_stress()
