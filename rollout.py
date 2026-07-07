#!/usr/bin/env python3
"""
Autoregressive Rollout: extend prediction horizon without retraining.

Uses a sliding-window approach:
  Rollout 0: hist[0:20]  → pred[0:20]   (0-4s LOW / 0-20s HIGH)
  Rollout 1: hist[10:20] + reconstructed[0:10] → pred[20:40]
  Rollout 2: reconstructed[10:20] from R1 + reconstructed[20:30] from R2...

The model predicts RELATIVE displacement from last history position.
We accumulate to absolute positions, then reconstruct valid history tensors.
"""

import torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 8})

INTENT_6 = ['STRAIGHT','TURN_L','TURN_R','ASCEND','DESC','HOVER']
INTENT_4 = ['STRAIGHT','TURN_L','TURN_R','DESCEND']
DT_LOW = 0.2   # 5Hz
DT_HIGH = 1.0  # 1Hz


def reconstruct_history(all_positions, start_idx, n_frames, dt):
    """
    Build a (n_frames, 6) history tensor from absolute positions.
    all_positions: (T, 3) numpy array of absolute positions
    start_idx: first frame to include
    n_frames: number of frames (20)
    dt: time step in seconds
    Returns: (20, 6) tensor [pos, vel]
    """
    pos = all_positions[start_idx:start_idx + n_frames].copy()
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / dt
    vel[0] = vel[1] if n_frames > 1 else 0
    return torch.from_numpy(np.concatenate([pos, vel], axis=1)).float()


def autoregressive_rollout(model, hist, n_rollouts, dt, device):
    """
    Perform autoregressive rollout prediction.

    Args:
        model: TrajectoryPredictor
        hist: (20, 6) initial history tensor
        n_rollouts: number of additional rollouts (0 = just the first 20-step pred)
        dt: time step
        device: torch device

    Returns:
        all_preds: list of (20, 3) relative predictions per rollout
        all_abs_pos: (T_total, 3) all absolute positions (history + accumulated preds)
    """
    hist = hist.to(device)
    n_frames = 20

    # Get initial absolute positions
    abs_positions = hist[:, :3].cpu().numpy().copy()  # (20, 3)

    all_preds = []

    for rollout_idx in range(n_rollouts + 1):
        # Build history from accumulated positions
        start_idx = len(abs_positions) - n_frames
        h = reconstruct_history(abs_positions, start_idx, n_frames, dt).to(device)

        with torch.no_grad():
            out = model(h.unsqueeze(0), force_predict=True)
        pred = out['predictions'][0].cpu().numpy()  # (20, 3) relative displacement

        all_preds.append(torch.from_numpy(pred))

        # Accumulate: pred[t] is TOTAL displacement from reference, NOT per-step delta
        base_pos = abs_positions[-1].copy()
        for step in range(n_frames):
            abs_pos = base_pos + pred[step]
            abs_positions = np.concatenate([abs_positions, abs_pos.reshape(1, 3)], axis=0)

    return all_preds, abs_positions


def draw_rollout(ax, abs_positions, rollout_boundaries, title, dt):
    """Draw 2D trajectory with rollout segments in different colors."""
    colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0', '#009688']
    n_frames = 20

    for i, boundary in enumerate(rollout_boundaries):
        start, end = boundary
        color = colors[i % len(colors)]
        label = f'R{i} ({start*dt:.0f}-{end*dt:.0f}s)' if i > 0 else f'History (0-{20*dt:.0f}s)'
        seg = abs_positions[start:end+1]
        lw = 2.5 if i == 0 else 1.5
        ls = '-' if i == 0 else '--'
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, ls=ls, label=label, alpha=0.85)
        if i > 0:
            # Mark rollout start
            ax.scatter(seg[0, 0], seg[0, 1], c=color, s=50, marker='D', zorder=10, ec='k', lw=0.5)

    ax.scatter(abs_positions[0, 0], abs_positions[0, 1], c='green', s=80, marker='o', zorder=15, ec='k', lw=1, label='Start')
    ax.scatter(abs_positions[-1, 0], abs_positions[-1, 1], c='red', s=80, marker='*', zorder=15, ec='k', lw=1, label='End')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_box_aspect(1); ax.grid(True, alpha=0.3); ax.legend(fontsize=5, loc='best')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    p = DronePredictor(device=device)

    # === LOW Model Rollout ===
    print('='*60)
    print('LOW MODEL: Autoregressive Rollout (4s → 8s → 12s)')
    print('='*60)

    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    np.random.seed(42)
    indices = np.random.choice(len(ds_low), 6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('LOW Model: Autoregressive Rollout (4s per rollout, 3 rollouts = 12s total)\n'
                 'Blue=History | Orange=R1 | Red=R2 | Purple=R3',
                 fontsize=12, fontweight='bold')
    axes = axes.flatten()

    for ax_i, idx in enumerate(indices):
        hist, target, intent = ds_low[idx]
        intent_name = INTENT_6[int(intent)] if int(intent) < 6 else '?'
        speed = DronePredictor.compute_speed(hist.unsqueeze(0)).item()

        n_rollouts = 2  # 3 total passes = 12s prediction
        all_preds, abs_positions = autoregressive_rollout(
            p.low, hist, n_rollouts, DT_LOW, device
        )

        # Build rollout boundaries
        boundaries = [(0, 19)]  # history
        for r in range(n_rollouts + 1):
            start = 20 + r * 20
            end = start + 19
            boundaries.append((start, end))

        draw_rollout(axes[ax_i], abs_positions, boundaries,
                     f'#{idx} {intent_name} {speed:.1f}m/s\n12s rollout (3x4s passes)',
                     DT_LOW)

    out_path = out_dir / 'rollout_low.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    # === HIGH Model Rollout ===
    print('\n' + '='*60)
    print('HIGH MODEL: Autoregressive Rollout (20s → 40s)')
    print('='*60)

    ds_high = FastWindowDataset('../SimCruise', split='test', label_remap={4: 3})
    np.random.seed(123)
    indices_h = np.random.choice(len(ds_high), 6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('HIGH Model: Autoregressive Rollout (20s per rollout, 2 rollouts = 40s total)\n'
                 'Blue=History | Orange=R1 (20-40s)',
                 fontsize=12, fontweight='bold')
    axes = axes.flatten()

    for ax_i, idx in enumerate(indices_h):
        hist, target, intent = ds_high[idx]
        intent_name = INTENT_4[int(intent)] if int(intent) < 4 else '?'
        speed = DronePredictor.compute_speed(hist.unsqueeze(0)).item()

        n_rollouts = 1  # 2 total passes = 40s prediction
        all_preds, abs_positions = autoregressive_rollout(
            p.high, hist, n_rollouts, DT_HIGH, device
        )

        boundaries = [(0, 19), (20, 39)]
        draw_rollout(axes[ax_i], abs_positions, boundaries,
                     f'#{idx} {intent_name} {speed:.1f}m/s\n40s rollout (2x20s passes)',
                     DT_HIGH)

    out_path = out_dir / 'rollout_high.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    # === Quantitative: Error accumulation over rollouts ===
    print('\n' + '='*60)
    print('ERROR ACCUMULATION OVER ROLLOUTS')
    print('='*60)

    for model_label, ds, model, dt, intent_names in [
        ('LOW', ds_low, p.low, DT_LOW, INTENT_6),
        ('HIGH', ds_high, p.high, DT_HIGH, INTENT_4),
    ]:
        np.random.seed(99)
        test_idx = np.random.choice(min(500, len(ds)), 50, replace=False)
        errors_by_rollout = {0: [], 1: [], 2: []}

        for idx in test_idx:
            hist, target_raw, intent = ds[idx]
            # target_raw = absolute future positions (20, 3)
            true_end = target_raw[-1, :3].numpy()  # last ground truth position

            n_rollouts = 2 if model_label == 'LOW' else 1
            all_preds, abs_positions = autoregressive_rollout(
                model, hist, n_rollouts, dt, device
            )

            # Validate first rollout against ground truth
            pred_end = abs_positions[39]  # last frame of first rollout (20 hist + 20 pred - 1)
            err = np.linalg.norm(pred_end - true_end)
            errors_by_rollout[0].append(err)

        print(f'\n{model_label}:')
        for r, errs in errors_by_rollout.items():
            if errs:
                errs = np.array(errs)
                print(f'  Rollout {r} ({r*20*dt:.0f}-{(r+1)*20*dt:.0f}s): '
                      f'FDE mean={errs.mean():.2f}m median={np.median(errs):.2f}m '
                      f'P95={np.percentile(errs, 95):.2f}m (n={len(errs)})')

    print('\nDone!')


if __name__ == '__main__':
    main()
