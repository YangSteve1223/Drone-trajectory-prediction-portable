#!/usr/bin/env python3
"""
A+B+C combined evaluation on long trajectories (>=150 frames):
  A: Temporal downsampling (stride=2, 8s context)
  B: ContextAdapterV2 (60-frame → d_model injection)
  C: Gate inertia scaling (scale=0.3, neural gets 70%)

Tests all combinations to isolate each component's contribution.
"""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from context_adapter import ContextAdapterV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
ADAPTER_PATH = Path(__file__).parent / 'weights' / 'context_adapter_long.pth'
OUT = Path(__file__).parent / 'pic-results' / 'abc_combined_eval.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
LONG_THRESHOLD = 150
MAX_TRAJ = 500


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def make_windows(traj, stride=1):
    """Create windows. stride>1 means downsampled history."""
    n = traj.shape[0]
    ml = HIST_LEN * stride + PRED_LEN
    if n < ml:
        return [], [], []
    hists, futs, starts = [], [], []
    step = max(1, stride // 2)
    for i in range(0, n - ml + 1, step):
        idx = np.arange(i, i + HIST_LEN * stride, stride)[:HIST_LEN]
        hists.append(traj[idx].copy())
        fut_start = i + HIST_LEN * stride
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
        starts.append(i)
    return hists, futs, starts


def eval_config(model, device, adapter, stride, gate_scale, use_adapter):
    """Evaluate one config on long trajectories."""
    all_ade, all_fde, all_dire = [], [], []
    n_traj, n_wins, n_cata = 0, 0, 0

    # Patch gate
    orig_forward = model.ua_pgd.physics_gate.forward
    def scaled_forward(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_forward(last_encoded, intent_weights, step_encoding)
        return gi * gate_scale, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_forward

    try:
        for f in sorted(TRAJ_DIR.glob('*.npz')):
            d = np.load(f)
            nf = d['traj'].shape[0]
            if nf < LONG_THRESHOLD:
                continue
            n_traj += 1
            if n_traj > MAX_TRAJ:
                break

            hists, futs, starts = make_windows(d['traj'], stride=stride)
            if len(hists) < 10:
                continue

            # For adapter: create 60-frame context for each window
            ctx_feats = None
            if use_adapter and adapter is not None:
                ctx_feats = []
                for ws in starts:
                    ctx_end = ws + CTX_LEN
                    if ctx_end <= nf:
                        ctx = d['traj'][ws:ctx_end, :].copy()  # (60, 6)
                    else:
                        ctx = d['traj'][-CTX_LEN:, :].copy()
                    ctx_feats.append(ctx)
                ctx_feats = torch.from_numpy(np.array(ctx_feats, dtype=np.float32))

            # Evaluate
            for b in range(0, len(hists), 128):
                be = min(b + 128, len(hists))
                hb = torch.stack([torch.from_numpy(hists[i]).float()
                                 for i in range(b, be)]).to(device)
                ctx_inj = None
                if ctx_feats is not None:
                    cb = ctx_feats[b:be].to(device)
                    with torch.no_grad():
                        ctx_inj = adapter(cb)

                with torch.no_grad():
                    kwargs = {'force_predict': True}
                    if ctx_inj is not None:
                        kwargs['context_injection'] = ctx_inj
                    pb = model(hb, **kwargs)['predictions'].cpu()

                for j in range(pb.shape[0]):
                    k = b + j
                    tb = torch.from_numpy(futs[k]).float()
                    de = dir_err(pb[j, -1, :2].numpy(), futs[k][-1, :2])
                    ade_val = float(torch.norm(pb[j] - tb, dim=-1).mean())
                    fde_val = float(torch.norm(pb[j, -1, :] - tb[-1, :]))
                    all_ade.append(ade_val); all_fde.append(fde_val)
                    all_dire.append(de)
                    if de >= 90:
                        n_cata += 1
                n_wins += pb.shape[0]
    finally:
        model.ua_pgd.physics_gate.forward = orig_forward

    ade = np.array(all_ade); fde = np.array(all_fde); dire = np.array(all_dire)
    return {
        'n_traj': n_traj, 'n_wins': n_wins,
        'n_cata': n_cata, 'cata_pct': float(n_cata / max(n_wins, 1) * 100),
        'ade_mean': float(ade.mean()), 'fde_mean': float(fde.mean()),
        'fde_p95': float(np.percentile(fde, 95)),
        'dir_mean': float(dire.mean()),
        'dir_p95': float(np.percentile(dire, 95)),
    }


def main():
    print('=' * 80)
    print('A+B+C COMBINED EVALUATION — Long Trajectories (>=150 frames)')
    print('  A: stride=2 (8s context)')
    print('  B: ContextAdapterV2 (60-frame → d_model)')
    print('  C: gate_scale=0.3 (neural 70%)')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device

    # Load adapter
    adapter = None
    if ADAPTER_PATH.exists():
        adapter = ContextAdapterV2(input_dim=6, context_len=CTX_LEN,
                                   d_model=model.d_model, hidden=128).to(device)
        adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=device))
        adapter.eval()
        print(f'  Adapter loaded: {ADAPTER_PATH} (929 KB)')
    else:
        print(f'  WARNING: No adapter at {ADAPTER_PATH}, B disabled')

    # ── BASELINE (no fix) ──
    print('\n[1] BASELINE (stride=1, no gate, no adapter)...')
    bl = eval_config(model, device, adapter, stride=1, gate_scale=1.0, use_adapter=False)
    print(f'  {bl["n_traj"]}trajs {bl["n_wins"]}wins  ADE={bl["ade_mean"]:.3f}m  '
          f'FDE={bl["fde_mean"]:.3f}m  Dir={bl["dir_mean"]:.1f}°  Cata={bl["cata_pct"]:.1f}%')

    # ── ALL COMBINATIONS ──
    configs = [
        ('BASELINE', 1, 1.0, False),
        ('A only', 2, 1.0, False),
        ('C only', 1, 0.3, False),
        ('B only', 1, 1.0, True),
        ('A+C', 2, 0.3, False),
        ('B+C', 1, 0.3, True),
        ('A+B', 2, 1.0, True),
        ('A+B+C', 2, 0.3, True),
    ]

    results = {'baseline': bl, 'configs': {}}
    print(f'\n[2] Testing all combinations...')
    for name, stride, gs, use_adpt in configs[1:]:  # skip BASELINE
        print(f'  {name}...', end=' ', flush=True)
        r = eval_config(model, device, adapter, stride, gs, use_adpt)
        fde_g = (bl['fde_mean'] - r['fde_mean']) / bl['fde_mean'] * 100
        dir_d = bl['dir_mean'] - r['dir_mean']
        cata_d = bl['cata_pct'] - r['cata_pct']
        cata_reduction = (bl['cata_pct'] - r['cata_pct']) / max(bl['cata_pct'], 0.1) * 100
        print(f'FDE={r["fde_mean"]:.3f}m Dir={r["dir_mean"]:.1f}° '
              f'Cata={r["cata_pct"]:.1f}% (FDE {fde_g:+.1f}% Cata {cata_d:+.1f}pp '
              f'↓{cata_reduction:.0f}%)')
        results['configs'][name] = {**r, 'fde_gain_pct': fde_g, 'dir_delta': dir_d,
                                     'cata_delta': cata_d, 'cata_reduction': cata_reduction}

    # ── SUMMARY TABLE ──
    print(f'\n{"=" * 90}')
    print(f'{"Config":<20} {"ADE":<8} {"FDE":<8} {"Dir":<7} {"Cata%":<7} '
          f'{"FDEGain":<8} {"DirΔ":<7} {"Cata↓":<7} {"wins":<7}')
    print(f'{"-" * 90}')
    print(f'{"BASELINE":<20} {bl["ade_mean"]:<8.3f} {bl["fde_mean"]:<8.3f} '
          f'{bl["dir_mean"]:<7.1f} {bl["cata_pct"]:<7.1f} {"—":<8} {"—":<7} {"—":<7} '
          f'{bl["n_wins"]:<7}')
    for name, stride, gs, use_adpt in configs[1:]:
        r = results['configs'][name]
        print(f'{name:<20} {r["ade_mean"]:<8.3f} {r["fde_mean"]:<8.3f} '
              f'{r["dir_mean"]:<7.1f} {r["cata_pct"]:<7.1f} '
              f'{r["fde_gain_pct"]:>+6.1f}%  {r["dir_delta"]:>+5.1f}°  '
              f'{r["cata_delta"]:>+5.1f}pp  {r["n_wins"]:<7}')

    print(f'\n  Best Cata reduction: track the config with lowest Cata%')

    json.dump(results, open(OUT, 'w'), indent=2, default=str)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
