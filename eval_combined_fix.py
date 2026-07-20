#!/usr/bin/env python3
"""
Combined fix evaluation for long trajectory failures:
  A: Temporal downsampling (stride=2, 8s context)
  C: Gate inertia scaling (scale=0.3 → neural gets 70% weight)
  A+C: Both combined

Tests all configurations on long trajectories (>=150 frames).
"""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
OUT = Path(__file__).parent / 'pic-results' / 'combined_fix_eval.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 20, 20
LONG_THRESHOLD = 150


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def make_windows(traj, hist_stride=1):
    n = traj.shape[0]
    if hist_stride == 1:
        ml = HIST_LEN + PRED_LEN
        if n < ml:
            return [], []
        hists, futs = [], []
        for i in range(0, n - ml + 1, 2):
            hists.append(traj[i:i + HIST_LEN])
            fut_abs = traj[i + HIST_LEN:i + HIST_LEN + PRED_LEN, :3]
            futs.append(fut_abs - traj[i + HIST_LEN - 1, :3])
    else:
        ml = HIST_LEN * hist_stride + PRED_LEN
        if n < ml:
            return [], []
        hists, futs = [], []
        step = max(1, hist_stride // 2)
        for i in range(0, n - ml + 1, step):
            indices = np.arange(i, i + HIST_LEN * hist_stride, hist_stride)[:HIST_LEN]
            fut_start = i + HIST_LEN * hist_stride
            hists.append(traj[indices].copy())
            fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
            futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs


def evaluate_with_gate_scale(model, hists, futs, device, gate_scale=1.0):
    """
    Evaluate model with gate_inertia output scaled by `gate_scale`.
    gate_scale=0.3: effective inertia drops from ~0.71 to ~0.21 → neural gets ~79%.
    Applied by temporarily replacing gate output in forward pass.
    """
    n = len(hists)
    ade, fde, dire = [], [], []

    # Intercept: wrap the PhysicsInertiaGate.forward to scale gate_inertia
    orig_forward = model.ua_pgd.physics_gate.forward

    def scaled_forward(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_forward(last_encoded, intent_weights, step_encoding)
        return gi * gate_scale, ga, gc, gm, gme

    model.ua_pgd.physics_gate.forward = scaled_forward

    try:
        for b in range(0, n, 128):
            be = min(b + 128, n)
            hb = torch.stack([torch.from_numpy(hists[i]).float()
                             for i in range(b, be)]).to(device)
            tb = torch.stack([torch.from_numpy(futs[i]).float()
                             for i in range(b, be)])
            with torch.no_grad():
                pb = model(hb, force_predict=True)['predictions'].cpu()
            diff = pb - tb
            ade.extend(torch.norm(diff, dim=-1).mean(dim=1).numpy())
            fde.extend(torch.norm(diff[:, -1, :], dim=-1).numpy())
            dire.extend([dir_err(pb[i, -1, :2].numpy(), tb[i, -1, :2].numpy())
                         for i in range(pb.shape[0])])
    finally:
        model.ua_pgd.physics_gate.forward = orig_forward

    return np.array(ade), np.array(fde), np.array(dire)


def eval_group(model, device, min_frames, max_frames, max_traj, hist_stride, gate_scale):
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
        hists, futs = make_windows(d['traj'], hist_stride=hist_stride)
        if len(hists) < 10:
            continue
        ade, fde, dire = evaluate_with_gate_scale(model, hists, futs, device, gate_scale)
        all_ade.extend(ade); all_fde.extend(fde); all_dire.extend(dire)
        n_wins += len(ade)
        n_cata += (dire >= 90).sum()
    all_ade = np.array(all_ade); all_fde = np.array(all_fde)
    all_dire = np.array(all_dire)
    return {'n_traj': n_traj, 'n_wins': n_wins,
            'n_cata': int(n_cata),
            'cata_pct': float(n_cata / max(n_wins, 1) * 100),
            'ade_mean': float(all_ade.mean()),
            'fde_mean': float(all_fde.mean()),
            'fde_p95': float(np.percentile(all_fde, 95)),
            'dir_mean': float(all_dire.mean()),
            'dir_p95': float(np.percentile(all_dire, 95))}


def main():
    print('=' * 80)
    print('COMBINED FIX — Downsampling + Gate Scale on Long Trajectories')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device

    # ── BASELINE ──
    print('\n[1] BASELINE (stride=1, gate_scale=1.0)...')
    bl = eval_group(model, device, LONG_THRESHOLD, 999, 500, 1, 1.0)
    print(f'  {bl["n_traj"]}trajs {bl["n_wins"]}wins  '
          f'ADE={bl["ade_mean"]:.3f}m  FDE={bl["fde_mean"]:.3f}m  '
          f'Dir={bl["dir_mean"]:.1f}°  Cata={bl["cata_pct"]:.1f}%')

    # ── CONFIGS ──
    configs = [
        ('A: stride=2 only', 2, 1.0),
        ('C: gate=0.5 only', 1, 0.5),
        ('C: gate=0.3 only', 1, 0.3),
        ('C: gate=0.2 only', 1, 0.2),
        ('A+C: stride=2 gate=0.5', 2, 0.5),
        ('A+C: stride=2 gate=0.3', 2, 0.3),
        ('A+C: stride=2 gate=0.2', 2, 0.2),
        ('A+C: stride=3 gate=0.3', 3, 0.3),
    ]

    results = {}
    for name, stride, gs in configs:
        print(f'\n  {name}...')
        r = eval_group(model, device, LONG_THRESHOLD, 999, 500, stride, gs)
        fde_g = (bl['fde_mean'] - r['fde_mean']) / bl['fde_mean'] * 100
        dir_d = bl['dir_mean'] - r['dir_mean']
        cata_d = bl['cata_pct'] - r['cata_pct']
        print(f'    ADE={r["ade_mean"]:.3f}m  FDE={r["fde_mean"]:.3f}m  '
              f'Dir={r["dir_mean"]:.1f}°  Cata={r["cata_pct"]:.1f}%  '
              f'wins={r["n_wins"]}')
        print(f'    vs BL: FDE {fde_g:+.1f}%  Dir {dir_d:+.1f}°  Cata {cata_d:+.1f}pp')
        results[name] = {**r, 'fde_gain_pct': fde_g, 'dir_delta': dir_d,
                         'cata_delta': cata_d}

    # ── Also evaluate on MED to check regression ──
    print(f'\n[REG CHECK] Medium trajectories (should not degrade)...')
    bl_med = eval_group(model, device, 50, LONG_THRESHOLD - 1, 1000, 1, 1.0)
    print(f'  Baseline MED: ADE={bl_med["ade_mean"]:.3f}m  FDE={bl_med["fde_mean"]:.3f}m  '
          f'Dir={bl_med["dir_mean"]:.1f}°  Cata={bl_med["cata_pct"]:.1f}%')

    # ── SUMMARY ──
    print(f'\n{"=" * 90}')
    print(f'{"Config":<30} {"ADE":<8} {"FDE":<8} {"Dir":<7} {"Cata%":<7} '
          f'{"FDEGain":<8} {"DirΔ":<7} {"CataΔ":<7}')
    print(f'{"-" * 90}')
    print(f'{"BASELINE":<30} {bl["ade_mean"]:<8.3f} {bl["fde_mean"]:<8.3f} '
          f'{bl["dir_mean"]:<7.1f} {bl["cata_pct"]:<7.1f} {"—":<8} {"—":<7} {"—":<7}')
    for name, stride, gs in configs:
        r = results[name]
        print(f'{name:<30} {r["ade_mean"]:<8.3f} {r["fde_mean"]:<8.3f} '
              f'{r["dir_mean"]:<7.1f} {r["cata_pct"]:<7.1f} '
              f'{r["fde_gain_pct"]:>+6.1f}%  {r["dir_delta"]:>+5.1f}°  '
              f'{r["cata_delta"]:>+5.1f}pp')

    json.dump({'baseline': bl, 'configs': results, 'med_baseline': bl_med},
              open(OUT, 'w'), indent=2, default=str)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
