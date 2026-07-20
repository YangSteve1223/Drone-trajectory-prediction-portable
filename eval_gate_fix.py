#!/usr/bin/env python3
"""Evaluate physics gate bias fix: before (bias=0.89, inertia≈0.71) vs after (bias=0.0, inertia=0.50)."""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
OUT = Path(__file__).parent / 'pic-results' / 'gate_fix_eval.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 20, 20


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def sliding_windows(traj):
    n = traj.shape[0]
    ml = HIST_LEN + PRED_LEN
    if n < ml:
        return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, 2):
        hists.append(traj[i:i + HIST_LEN])
        futs.append(traj[i + HIST_LEN:i + HIST_LEN + PRED_LEN, :3]
                    - traj[i + HIST_LEN - 1, :3])
    return hists, futs


def evaluate_model(model, hists, futs, device, bs=128):
    n = len(hists)
    ade, fde, dire = [], [], []
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
    return np.array(ade), np.array(fde), np.array(dire)


def eval_group(model, device, desc, min_frames=0, max_frames=999, max_traj=2000):
    """Evaluate on trajectories in a frame range. Limits to max_traj for speed."""
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
        hists, futs = sliding_windows(d['traj'])
        if len(hists) == 0:
            continue
        ade, fde, dire = evaluate_model(model, hists, futs, device)
        all_ade.extend(ade); all_fde.extend(fde); all_dire.extend(dire)
        n_wins += len(ade)
        n_cata += (dire >= 90).sum()
    all_ade = np.array(all_ade); all_fde = np.array(all_fde)
    all_dire = np.array(all_dire)
    return {'n_traj': n_traj, 'n_wins': n_wins,
            'n_cata': int(n_cata), 'cata_pct': float(n_cata / max(n_wins, 1) * 100),
            'ade_mean': float(all_ade.mean()), 'ade_median': float(np.median(all_ade)),
            'fde_mean': float(all_fde.mean()), 'fde_median': float(np.median(all_fde)),
            'fde_p95': float(np.percentile(all_fde, 95)),
            'dir_mean': float(all_dire.mean()), 'dir_median': float(np.median(all_dire)),
            'dir_p95': float(np.percentile(all_dire, 95))}


def main():
    print('=' * 80)
    print('Physics Gate Bias Fix — Before vs After')
    print('  Before: gate_inertia bias ≈ 0.89 → sigmoid ≈ 0.71 (71% physics)')
    print('  After:  gate_inertia bias = 0.00 → sigmoid = 0.50 (50% physics)')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device
    bias_before = model.ua_pgd.physics_gate.gate_mlp[2].bias.data.clone()
    print(f'\nLoaded gate_mlp[2] bias: [{bias_before[0]:.4f}, {bias_before[1]:.4f}]')

    # BEFORE
    print('\n[1/3] Evaluating BEFORE (original)...')
    before_long = eval_group(model, device, 'long', min_frames=150, max_traj=500)
    before_med = eval_group(model, device, 'med', min_frames=50, max_frames=149, max_traj=1000)
    before_short = eval_group(model, device, 'short', max_frames=50, max_traj=1000)

    # Fix bias
    with torch.no_grad():
        model.ua_pgd.physics_gate.gate_mlp[2].bias[0] = 0.0
    bias_after = model.ua_pgd.physics_gate.gate_mlp[2].bias.data.clone()
    print(f'\n  Modified bias: [{bias_after[0]:.4f}, {bias_after[1]:.4f}]')
    print(f'  gate_inertia sigmoid: {torch.sigmoid(bias_after[0]).item():.4f}')

    # AFTER
    print('\n[2/3] Evaluating AFTER (bias=0.0)...')
    after_long = eval_group(model, device, 'long', min_frames=150, max_traj=500)
    after_med = eval_group(model, device, 'med', min_frames=50, max_frames=149, max_traj=1000)
    after_short = eval_group(model, device, 'short', max_frames=50, max_traj=1000)

    # Restore
    with torch.no_grad():
        model.ua_pgd.physics_gate.gate_mlp[2].bias.copy_(bias_before)

    # COMPARISON
    print(f'\n[3/3] COMPARISON')
    print(f'{"=" * 80}')
    print(f'{"Group":<20} {"Metric":<10} {"Before":<10} {"After":<10} {"Delta":<10}')
    print(f'{"-" * 80}')

    results = {'before': {}, 'after': {}, 'delta': {}}
    for tag, before, after in [
        ('LONG (>=150f)', before_long, after_long),
        ('MED (50-149f)', before_med, after_med),
        ('SHORT (<50f)', before_short, after_short),
    ]:
        for metric, key, unit, better in [
            ('ADE', 'ade_mean', 'm', 'lower'),
            ('FDE', 'fde_mean', 'm', 'lower'),
            ('FDE p95', 'fde_p95', 'm', 'lower'),
            ('Dir', 'dir_mean', '°', 'lower'),
            ('Cata%', 'cata_pct', '%', 'lower'),
        ]:
            bv, av = before[key], after[key]
            delta = bv - av
            pct = delta / max(bv, 0.001) * 100 if bv > 0 else 0
            arrow = '▲' if (better == 'lower' and delta > 0) or (better == 'higher' and delta < 0) else '▽'
            print(f'  {tag:<20} {metric:<10} {bv:<10.3f} {av:<10.3f} '
                  f'{delta:+.3f} ({pct:+.1f}%) {arrow}')

        results['before'][tag] = before
        results['after'][tag] = after
        results['delta'][tag] = {
            'ade_gain_pct': float((before['ade_mean'] - after['ade_mean']) / max(before['ade_mean'], 0.001) * 100),
            'fde_gain_pct': float((before['fde_mean'] - after['fde_mean']) / max(before['fde_mean'], 0.001) * 100),
            'dir_delta_deg': float(before['dir_mean'] - after['dir_mean']),
            'cata_delta_pp': float(before['cata_pct'] - after['cata_pct']),
        }
        print()

    json.dump(results, open(OUT, 'w'), indent=2)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
