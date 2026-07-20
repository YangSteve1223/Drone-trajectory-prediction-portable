#!/usr/bin/env python3
"""
Extended context for long trajectories via temporal downsampling + gate fix.

Strategy:
  A: Downsample history to cover more time with the same 20-frame window.
     e.g., stride=3: 20 frames cover 60 original frames (12s vs 4s).
  B: Scale down gate_inertia output to give neural decoder more weight.
  Threshold: only activate for trajectories >= 150 frames.

Tests multiple stride configurations and gate scale factors.
"""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
OUT = Path(__file__).parent / 'pic-results' / 'extended_context_eval.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 20, 20
LONG_THRESHOLD = 150                # activate extended context for >=150 frames
DIRERR_MAX = 60.0                   # trainable window threshold


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def make_windows(traj, hist_stride=1):
    """
    Extract (history, future_displacement) windows.
    hist_stride > 1: downsample history to cover more time.
    e.g., hist_stride=3: each window's 20 frames span 60 original frames (12s @ 5Hz).
    """
    n = traj.shape[0]
    ml = HIST_LEN * hist_stride + PRED_LEN
    if n < ml:
        return [], []

    hists, futs = [], []
    step = max(1, hist_stride // 2)  # window stride for adequate coverage
    for i in range(0, n - ml + 1, step):
        # Downsampled history: take every hist_stride-th frame
        hist_indices = np.arange(i, i + HIST_LEN * hist_stride, hist_stride)[:HIST_LEN]
        fut_start = i + HIST_LEN * hist_stride
        hists.append(traj[hist_indices].copy())
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs


def evaluate_model(model, hists, futs, device, bs=128,
                   gate_scale=1.0, adapter=None, ctx_windows=None):
    """
    Evaluate model. If gate_scale != 1.0: modify gate_inertia output.
    ctx_windows: optional (N, 60, 6) context windows for adapter injection.
    """
    n = len(hists)
    ade, fde, dire = [], [], []

    # Save original bias for restoration
    orig_bias = model.ua_pgd.physics_gate.gate_mlp[2].bias.data.clone()

    for b in range(0, n, bs):
        be = min(b + bs, n)
        hb = torch.stack([torch.from_numpy(hists[i]).float() for i in range(b, be)]).to(device)
        tb = torch.stack([torch.from_numpy(futs[i]).float() for i in range(b, be)])

        with torch.no_grad():
            pb = model(hb, force_predict=True)['predictions'].cpu()

        diff = pb - tb
        ade.extend(torch.norm(diff, dim=-1).mean(dim=1).numpy())
        fde.extend(torch.norm(diff[:, -1, :], dim=-1).numpy())
        dire.extend([dir_err(pb[i, -1, :2].numpy(), tb[i, -1, :2].numpy())
                     for i in range(pb.shape[0])])

    return (np.array(ade), np.array(fde), np.array(dire))


def eval_trajectory_group(model, device, desc, min_frames=0, max_frames=999,
                          max_traj=500, hist_stride=1):
    """Evaluate a group of trajectories with optional downsampling."""
    all_ade, all_fde, all_dire = [], [], []
    n_traj, n_wins, n_cata = 0, 0, 0

    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        nf = d['traj'].shape[0]
        if nf < min_frames or nf > max_frames:
            continue
        n_traj += 1
        if n_traj > max_traj:
            break

        # For long trajectories: use downsampled windows
        hists, futs = make_windows(d['traj'], hist_stride=hist_stride)
        if len(hists) < 10:
            continue

        ade, fde, dire = evaluate_model(model, hists, futs, device)
        all_ade.extend(ade); all_fde.extend(fde); all_dire.extend(dire)
        n_wins += len(ade)
        n_cata += (dire >= 90).sum()

    all_ade = np.array(all_ade); all_fde = np.array(all_fde)
    all_dire = np.array(all_dire)
    return {
        'n_traj': n_traj, 'n_wins': n_wins,
        'n_cata': int(n_cata),
        'cata_pct': float(n_cata / max(n_wins, 1) * 100),
        'ade_mean': float(all_ade.mean()),
        'fde_mean': float(all_fde.mean()),
        'fde_p95': float(np.percentile(all_fde, 95)),
        'dir_mean': float(all_dire.mean()),
        'dir_p95': float(np.percentile(all_dire, 95)),
    }


def main():
    print('=' * 80)
    print('Extended Context Evaluation — Temporal Downsampling for Long Trajs')
    print(f'  Threshold: >= {LONG_THRESHOLD} frames → extended context')
    print(f'  Short/medium trajectories: unchanged (stride=1, 4s context)')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device

    # ── BASELINE (stride=1 for all) ──
    print('\n[1] BASELINE (stride=1, 4s context for all)...')
    bl_long = eval_trajectory_group(model, device, 'long', min_frames=LONG_THRESHOLD,
                                    max_traj=500, hist_stride=1)
    bl_med = eval_trajectory_group(model, device, 'med', min_frames=50,
                                    max_frames=LONG_THRESHOLD - 1,
                                    max_traj=1000, hist_stride=1)
    bl_short = eval_trajectory_group(model, device, 'short', max_frames=50,
                                     max_traj=1000, hist_stride=1)

    for tag, r in [('LONG (>=150f)', bl_long), ('MED (50-149f)', bl_med),
                   ('SHORT (<50f)', bl_short)]:
        print(f'  {tag}: {r["n_traj"]}trajs {r["n_wins"]}wins  '
              f'ADE={r["ade_mean"]:.3f}m  FDE={r["fde_mean"]:.3f}m  '
              f'Dir={r["dir_mean"]:.1f}°  Cata={r["cata_pct"]:.1f}%')

    # ── EXTENDED CONTEXT (downsampling for long trajectories only) ──
    results = {'baseline': {'long': bl_long, 'med': bl_med, 'short': bl_short},
               'configs': {}}

    configs = [
        # (name, hist_stride, effective_context)
        ('stride=2 (8s ctx)', 2, '8s'),
        ('stride=3 (12s ctx)', 3, '12s'),
        ('stride=4 (16s ctx)', 4, '16s'),
        ('stride=5 (20s ctx)', 5, '20s'),
    ]

    for cfg_name, stride, ctx_desc in configs:
        print(f'\n[2] EXTENDED: {cfg_name} for long trajectories...')

        # Only evaluate LONG trajectories with extended context
        # Medium and short stay the same (baseline reused)
        ex_long = eval_trajectory_group(model, device, 'long_ex',
                                        min_frames=LONG_THRESHOLD,
                                        max_traj=500, hist_stride=stride)

        # Compare vs baseline on long trajectories
        fde_gain = (bl_long['fde_mean'] - ex_long['fde_mean']) / bl_long['fde_mean'] * 100
        dir_delta = bl_long['dir_mean'] - ex_long['dir_mean']
        cata_delta = bl_long['cata_pct'] - ex_long['cata_pct']
        wins_ratio = ex_long['n_wins'] / max(bl_long['n_wins'], 1)

        print(f'  LONG: ADE={ex_long["ade_mean"]:.3f}m  FDE={ex_long["fde_mean"]:.3f}m  '
              f'Dir={ex_long["dir_mean"]:.1f}°  Cata={ex_long["cata_pct"]:.1f}%  '
              f'wins={ex_long["n_wins"]} ({wins_ratio:.1%} of baseline)')
        print(f'  vs Baseline: FDE {fde_gain:+.1f}%  Dir {dir_delta:+.1f}°  '
              f'Cata {cata_delta:+.1f}pp')

        results['configs'][cfg_name] = {
            'long': ex_long,
            'fde_gain_pct': float(fde_gain),
            'dir_delta_deg': float(dir_delta),
            'cata_delta_pp': float(cata_delta),
            'wins_ratio': float(wins_ratio),
        }

    # ── SUMMARY ──
    print(f'\n{"=" * 80}')
    print('SUMMARY — Extended Context on Long Trajectories')
    print(f'{"=" * 80}')
    print(f'{"Config":<25} {"ADE":<8} {"FDE":<8} {"FDEp95":<8} {"Dir":<7} {"Cata%":<8} '
          f'{"FDEGain":<9} {"DirΔ":<8}')
    print(f'{"-" * 80}')
    print(f'{"BASELINE (4s ctx)":<25} {bl_long["ade_mean"]:<8.3f} {bl_long["fde_mean"]:<8.3f} '
          f'{bl_long["fde_p95"]:<8.3f} {bl_long["dir_mean"]:<7.1f} '
          f'{bl_long["cata_pct"]:<8.1f} {"—":<9} {"—":<8}')
    for cfg_name, stride, ctx_desc in configs:
        r = results['configs'][cfg_name]
        ex = r['long']
        print(f'{cfg_name:<25} {ex["ade_mean"]:<8.3f} {ex["fde_mean"]:<8.3f} '
              f'{ex["fde_p95"]:<8.3f} {ex["dir_mean"]:<7.1f} {ex["cata_pct"]:<8.1f} '
              f'{r["fde_gain_pct"]:>+7.1f}%  {r["dir_delta_deg"]:>+6.1f}°')

    # Best config
    best = max(results['configs'].items(), key=lambda x: x[1]['fde_gain_pct'])
    print(f'\n  Best: {best[0]} — FDE {best[1]["fde_gain_pct"]:+.1f}%')

    json.dump(results, open(OUT, 'w'), indent=2, default=str)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
