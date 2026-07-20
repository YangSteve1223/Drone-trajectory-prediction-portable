#!/usr/bin/env python3
"""Comprehensive regression test: fine-tuned vs original on all trajectory groups."""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from emam_model import TrajectoryPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
ORIG_PATH = Path(__file__).parent / 'weights' / 'low_speed_6class.pth'
FINE_PATH = Path(__file__).parent / 'weights' / 'low_speed_6class_finetuned.pth'
OUT = Path(__file__).parent / 'pic-results' / 'finetuned_regression_test.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 20, 20


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def sliding_windows(traj, stride=None):
    """stride=None: auto-choose based on trajectory length."""
    n = traj.shape[0]
    if stride is None:
        stride = 2 if n >= 50 else 1
    ml = HIST_LEN + PRED_LEN
    if n < ml:
        return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, stride):
        hists.append(traj[i:i + HIST_LEN])
        fut_abs = traj[i + HIST_LEN:i + HIST_LEN + PRED_LEN, :3]
        futs.append(fut_abs - traj[i + HIST_LEN - 1, :3])
    return hists, futs


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def eval_model(model, hists, futs, device, bs=256):
    n = len(hists)
    ade, fde, dire = [], [], []
    for b in range(0, n, bs):
        be = min(b + bs, n)
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
    a = np.array(ade); f = np.array(fde); d = np.array(dire)
    return {
        'ade_mean': float(a.mean()), 'ade_median': float(np.median(a)),
        'ade_p95': float(np.percentile(a, 95)),
        'fde_mean': float(f.mean()), 'fde_median': float(np.median(f)),
        'fde_p95': float(np.percentile(f, 95)),
        'dir_mean': float(d.mean()), 'dir_median': float(np.median(d)),
        'dir_p95': float(np.percentile(d, 95)),
        'cata_pct': float((d >= 90).sum() / max(len(d), 1) * 100),
        'n_wins': n, 'n_traj': 0,
    }


def main():
    print('=' * 80)
    print('Fine-tuned Model Regression Test')
    print('=' * 80)

    print('\nLoading models...')
    orig = load_model(ORIG_PATH)
    fine = load_model(FINE_PATH)
    device = DEVICE

    # Collect windows
    print('Collecting trajectories...')
    groups = {'SHORT (<50f)': [], 'MED (50-149f)': [], 'LONG (>=150f)': []}
    n_traj = {'SHORT (<50f)': 0, 'MED (50-149f)': 0, 'LONG (>=150f)': 0}

    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        traj = d['traj']; nf = traj.shape[0]
        hists, futs = sliding_windows(traj)
        if len(hists) < 3:
            continue
        if nf >= 150:
            g = 'LONG (>=150f)'
        elif nf >= 50:
            g = 'MED (50-149f)'
        else:
            g = 'SHORT (<50f)'
        groups[g].extend(zip(hists, futs))
        n_traj[g] += 1

    # Evaluate
    print('\nEvaluating...')
    results = {}
    for gname, windows in groups.items():
        if len(windows) == 0:
            continue
        print(f'  {gname}: {len(windows)} windows from {n_traj[gname]} trajs...')
        hists = [w[0] for w in windows]
        futs = [w[1] for w in windows]

        ro = eval_model(orig, hists, futs, device)
        rf = eval_model(fine, hists, futs, device)
        ro['n_traj'] = n_traj[gname]
        rf['n_traj'] = n_traj[gname]

        # Deltas
        deltas = {}
        for k in ['ade_mean', 'fde_mean', 'dir_mean', 'cata_pct',
                   'ade_p95', 'fde_p95', 'dir_p95']:
            deltas[k] = ro[k] - rf[k]
        deltas['fde_gain'] = (ro['fde_mean'] - rf['fde_mean']) / max(ro['fde_mean'], 0.001) * 100
        deltas['ade_gain'] = (ro['ade_mean'] - rf['ade_mean']) / max(ro['ade_mean'], 0.001) * 100

        results[gname] = {'orig': ro, 'fine': rf, 'delta': deltas}

    # Print
    print(f'\n{"=" * 90}')
    print(f'{"Group":<18} {"Metric":<10} {"Original":<10} {"Fine-tuned":<10} '
          f'{"Delta":<12} {"Verdict":<8}')
    print(f'{"-" * 90}')

    for gname in ['LONG (>=150f)', 'MED (50-149f)', 'SHORT (<50f)']:
        if gname not in results:
            continue
        r = results[gname]
        for metric, key, unit in [
            ('ADE', 'ade_mean', 'm'), ('FDE', 'fde_mean', 'm'),
            ('FDE p95', 'fde_p95', 'm'), ('Dir', 'dir_mean', '°'),
            ('Cata%', 'cata_pct', '%'),
        ]:
            ov = r['orig'][key]; fv = r['fine'][key]
            d = r['delta'][key]
            if key == 'cata_pct':
                verdict = 'OK' if d >= -1.0 else ('WARN' if d >= -3.0 else 'DEGRADE')
            elif key in ('ade_mean', 'fde_mean', 'fde_p95', 'dir_mean'):
                verdict = 'OK' if d >= -0.03 * max(ov, 0.001) else ('WARN' if d >= -0.1 * max(ov, 0.001) else 'DEGRADE')
            else:
                verdict = '—'
            print(f'  {gname:<18} {metric:<10} {ov:<10.3f} {fv:<10.3f} '
                  f'{d:>+8.3f}{unit:<4} {verdict:<8}')

    # Summary
    print(f'\n{"=" * 90}')
    print('SUMMARY')
    print(f'{"=" * 90}')
    for gname in ['LONG (>=150f)', 'MED (50-149f)', 'SHORT (<50f)']:
        if gname not in results:
            continue
        r = results[gname]
        d = r['delta']
        print(f'\n  {gname} ({r["orig"]["n_wins"]} windows, {r["orig"]["n_traj"]} trajs):')
        print(f'    FDE: {r["orig"]["fde_mean"]:.3f} → {r["fine"]["fde_mean"]:.3f}m  '
              f'({d["fde_gain"]:+.1f}%)')
        print(f'    Dir: {r["orig"]["dir_mean"]:.1f} → {r["fine"]["dir_mean"]:.1f}°  '
              f'({d["dir_mean"]:+.1f}°)')
        print(f'    Cata: {r["orig"]["cata_pct"]:.1f} → {r["fine"]["cata_pct"]:.1f}%  '
              f'({d["cata_pct"]:+.1f}pp)')

    json.dump(results, open(OUT, 'w'), indent=2, default=str)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
