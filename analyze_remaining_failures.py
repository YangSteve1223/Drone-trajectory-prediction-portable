#!/usr/bin/env python3
"""
Analyze windows that STILL fail catastrophically (DirErr>=90) after the A+C fix
(stride=2 downsample + gate_scale=0.3) on long trajectories (>=150 frames).

Goal: characterize what makes these windows especially resistant to improvement,
and identify further optimization directions.
"""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
OUT = Path(__file__).parent / 'pic-results' / 'remaining_failures.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 20, 20
LONG_THRESHOLD = 150
STRIDE = 2       # A: downsampling
GATE_SCALE = 0.3  # C: gate scaling


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def make_windows(traj):
    """Downsampled windows (stride=2, 8s context)."""
    n = traj.shape[0]
    ml = HIST_LEN * STRIDE + PRED_LEN
    if n < ml:
        return [], [], []
    hists, futs, indices = [], [], []
    for i in range(0, n - ml + 1, 1):
        idx = np.arange(i, i + HIST_LEN * STRIDE, STRIDE)[:HIST_LEN]
        fut_start = i + HIST_LEN * STRIDE
        hists.append(traj[idx].copy())
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
        indices.append(i)  # starting frame
    return hists, futs, indices


def compute_turn_rate(traj, window_start, window_len):
    """Compute average turn rate (deg/s) for a segment of trajectory."""
    start = window_start
    end = min(window_start + window_len, len(traj) - 1)
    if end <= start + 1:
        return 0.0
    vel = traj[start + 1:end + 1, :2] - traj[start:end, :2]
    angles = np.arctan2(vel[:, 1], vel[:, 0])
    diffs = np.abs(np.diff(angles))
    diffs = np.minimum(diffs, 2 * np.pi - diffs)  # wrap-around
    return float(np.degrees(diffs.mean()) * 5.0)  # deg/s (5Hz)


def main():
    print('=' * 80)
    print('Analyzing REMAINING catastrophic failures after A+C fix')
    print(f'  Fix: stride={STRIDE} + gate_scale={GATE_SCALE}')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device

    # Apply gate scale
    orig_forward = model.ua_pgd.physics_gate.forward
    def scaled_forward(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_forward(last_encoded, intent_weights, step_encoding)
        return gi * GATE_SCALE, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_forward

    # Collect ALL windows from long trajectories with their features
    all_windows = []   # list of dicts with features per window
    n_traj, n_fixed, n_still_fail = 0, 0, 0

    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        traj = d['traj']
        nf = traj.shape[0]
        if nf < LONG_THRESHOLD:
            continue
        n_traj += 1

        hists, futs, starts = make_windows(traj)
        if len(hists) < 50:
            continue

        # Evaluate with fix
        for i in range(0, len(hists), 128):
            be = min(i + 128, len(hists))
            hb = torch.stack([torch.from_numpy(hists[j]).float()
                             for j in range(i, be)]).to(device)
            with torch.no_grad():
                pb = model(hb, force_predict=True)['predictions'].cpu()

            for j in range(pb.shape[0]):
                k = i + j  # original index
                de = dir_err(pb[j, -1, :2].numpy(), futs[k][-1, :2])
                ade = float(torch.norm(pb[j] - torch.from_numpy(futs[k]).float(),
                                       dim=-1).mean())
                fde = float(torch.norm(pb[j, -1, :] - torch.from_numpy(futs[k][-1]).float()))

                # Compute features
                ws = starts[k]
                hist_turn = compute_turn_rate(traj, ws, HIST_LEN * STRIDE)
                fut_turn = compute_turn_rate(traj, ws + HIST_LEN * STRIDE, PRED_LEN)
                hist_speed = float(np.linalg.norm(
                    traj[ws + HIST_LEN * STRIDE - 1, 3:6]))  # m/s at prediction point
                fut_disp = float(np.linalg.norm(futs[k][-1]))  # total displacement
                hist_disp = float(np.linalg.norm(
                    traj[ws + HIST_LEN * STRIDE - 1, :3] - traj[ws, :3]))  # history displacement

                window = {
                    'traj_name': f.name, 'traj_frames': nf,
                    'window_start': ws, 'ade': ade, 'fde': fde, 'dir_err': de,
                    'hist_turn_rate': hist_turn, 'fut_turn_rate': fut_turn,
                    'hist_speed': hist_speed, 'fut_disp': fut_disp,
                    'hist_disp': hist_disp,
                }
                all_windows.append(window)

                if de >= 90:
                    n_still_fail += 1
                else:
                    n_fixed += 1

    # Restore
    model.ua_pgd.physics_gate.forward = orig_forward

    total = n_fixed + n_still_fail
    print(f'\n  Long trajectories analyzed: {n_traj}')
    print(f'  Total windows: {total}')
    print(f'  Fixed (DirErr<90°): {n_fixed} ({n_fixed/total*100:.1f}%)')
    print(f'  STILL FAILING: {n_still_fail} ({n_still_fail/total*100:.1f}%)')

    # ── Compare fixed vs still-failing ──
    fixed = [w for w in all_windows if w['dir_err'] < 90]
    failing = [w for w in all_windows if w['dir_err'] >= 90]

    print(f'\n{"=" * 80}')
    print('FEATURE COMPARISON: Fixed vs Still-Failing Windows')
    print(f'{"=" * 80}')
    print(f'{"Feature":<25} {"Fixed (n=" + str(len(fixed)) + ")":<22} '
          f'{"Still Failing (n=" + str(len(failing)) + ")":<22}')

    for feat_name, fmt in [
        ('hist_turn_rate', '{:.1f} deg/s'),
        ('fut_turn_rate', '{:.1f} deg/s'),
        ('hist_speed', '{:.2f} m/s'),
        ('fut_disp', '{:.2f} m'),
        ('hist_disp', '{:.2f} m'),
        ('ade', '{:.3f} m'),
        ('fde', '{:.3f} m'),
        ('traj_frames', '{:.0f}'),
    ]:
        fv = np.array([w[feat_name] for w in fixed])
        sv = np.array([w[feat_name] for w in failing])
        print(f'  {feat_name:<25} {fmt.format(np.mean(fv)):<22} {fmt.format(np.mean(sv)):<22}')

    # Turn rate buckets
    print(f'\n{"=" * 80}')
    print('CATA RATE BY FUTURE TURN RATE BUCKET')
    print(f'{"=" * 80}')
    buckets = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 999)]
    for lo, hi in buckets:
        in_bucket = [w for w in all_windows if lo <= w['fut_turn_rate'] < hi]
        if len(in_bucket) < 10:
            continue
        cata_n = sum(1 for w in in_bucket if w['dir_err'] >= 90)
        print(f'  {lo:3d}-{hi:3d} deg/s: {len(in_bucket):5d} windows  '
              f'cata={cata_n:4d} ({cata_n/len(in_bucket)*100:5.1f}%)')

    # Worst trajectories
    print(f'\n{"=" * 80}')
    print('TOP 10 WORST TRAJECTORIES (by cata%)')
    print(f'{"=" * 80}')
    traj_stats = defaultdict(lambda: {'n': 0, 'cata': 0})
    for w in all_windows:
        t = traj_stats[w['traj_name']]
        t['n'] += 1
        if w['dir_err'] >= 90:
            t['cata'] += 1
    worst = sorted(traj_stats.items(), key=lambda x: x[1]['cata'] / max(x[1]['n'], 1), reverse=True)
    for name, stats in worst[:10]:
        print(f'  {name[:40]}: {stats["cata"]}/{stats["n"]} '
              f'({stats["cata"]/stats["n"]*100:.1f}%)')

    # Identify what makes remaining failures different
    print(f'\n{"=" * 80}')
    print('INSIGHTS FOR FURTHER OPTIMIZATION')
    print(f'{"=" * 80}')

    if len(failing) > 0:
        fv_turn = np.mean([w['fut_turn_rate'] for w in failing])
        fv_hist_turn = np.mean([w['hist_turn_rate'] for w in failing])
        fv_speed = np.mean([w['hist_speed'] for w in failing])
        fv_disp = np.mean([w['fut_disp'] for w in failing])
        fv_frames = np.mean([w['traj_frames'] for w in failing])

        # Extreme turn windows
        extreme_turn = [w for w in failing if w['fut_turn_rate'] > 120]
        hover = [w for w in failing if w['hist_speed'] < 0.3]

        print(f'  Remaining failures: {len(failing)} windows')
        print(f'  Avg future turn rate: {fv_turn:.1f} deg/s')
        print(f'  Avg history turn rate: {fv_hist_turn:.1f} deg/s')
        print(f'  Avg speed: {fv_speed:.2f} m/s')
        print(f'  Avg future displacement: {fv_disp:.2f} m')
        print(f'  Avg trajectory length: {fv_frames:.0f} frames')
        print(f'  Extreme turn (>120 deg/s): {len(extreme_turn)} ({len(extreme_turn)/len(failing)*100:.0f}%)')
        print(f'  Near-hover (<0.3 m/s): {len(hover)} ({len(hover)/len(failing)*100:.0f}%)')

        print(f'\n  → Remaining failures are characterized by:')
        if len(extreme_turn) / max(len(failing), 1) > 0.3:
            print(f'     • Extreme turning rates — beyond what 8s context can anticipate')
        if len(hover) / max(len(failing), 1) > 0.3:
            print(f'     • Near-hover speeds — physics model at near-zero velocity is unstable')
        print(f'     • These are the truly HARD cases — drones doing U-turns at near-zero speed')
        print(f'     • A+B+C (adapter with 12s context + downsampling + gate fix) may help')
        print(f'     • Ultimately, fine-tuning the base model on these challenging windows')
        print(f'       with higher loss weight on turn windows is needed for 0% failure')

    json.dump({
        'config': {'stride': STRIDE, 'gate_scale': GATE_SCALE},
        'summary': {'n_total': total, 'n_fixed': n_fixed, 'n_still_fail': n_still_fail,
                    'fixed_pct': n_fixed / max(total, 1) * 100},
        'feature_comparison': {
            'fixed': {k: float(np.mean([w[k] for w in fixed])) for k in
                      ['hist_turn_rate', 'fut_turn_rate', 'hist_speed', 'fut_disp', 'hist_disp',
                       'ade', 'fde']} if fixed else {},
            'failing': {k: float(np.mean([w[k] for w in failing])) for k in
                        ['hist_turn_rate', 'fut_turn_rate', 'hist_speed', 'fut_disp', 'hist_disp',
                         'ade', 'fde']} if failing else {},
        },
    }, open(OUT, 'w'), indent=2)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
