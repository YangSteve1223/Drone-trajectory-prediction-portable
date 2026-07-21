#!/usr/bin/env python3
"""
Ablation diagnostic: Why does base model deviate so much with ContextAdapter+A+C?

Tests 4 configs × 3 trajectory lengths to isolate the root cause.

Configs:
  1. RAW:        stride=1, gate=1.0, no adapter     (original model)
  2. +A:          stride=2, gate=1.0, no adapter     (downsample only)
  3. +C:          stride=1, gate=0.3, no adapter     (gate fix only)
  4. +AC:         stride=2, gate=0.3, no adapter     (A+C, no context)
  5. +Adapter:    stride=1, gate=1.0, adapter        (context only)
  6. +AC+Adapter: stride=2, gate=0.3, adapter        (current config)

Lengths: short (<80), medium (80-150), long (>=150)
"""

import torch, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from context_adapter import ContextAdapterV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
ADAPTER_PATH = Path(__file__).parent / 'weights' / 'context_adapter_ac.pth'
OUT_DIR = Path(__file__).parent / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
MAX_TRAJ_PER_GROUP = 50  # sample up to 50 per length group


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def make_windows(traj, stride=1):
    n = traj.shape[0]
    ml = HIST_LEN * stride + PRED_LEN
    if n < ml: return [], [], []
    hists, futs, starts = [], [], []
    step = max(1, stride // 2)
    for i in range(0, n - ml + 1, step):
        indices = np.arange(i, i + HIST_LEN * stride, stride)[:HIST_LEN]
        hists.append(traj[indices].copy())
        fut_start = i + HIST_LEN * stride
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
        starts.append(i)
    return hists, futs, starts


def eval_config(model, device, adapter, traj, stride, gate_scale, use_adapter):
    """Evaluate one config on a single trajectory. Returns per-window metrics."""
    # Patch gate
    orig_gate = model.ua_pgd.physics_gate.forward
    def scaled_gate(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_gate(
            last_encoded=last_encoded, intent_weights=intent_weights, step_encoding=step_encoding)
        return gi * gate_scale, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_gate

    try:
        hists, futs, starts = make_windows(traj, stride=stride)
        if len(hists) < 5: return None

        all_hist = np.array(hists, dtype=np.float32)
        n_total = len(hists)

        # Context adapter features
        ctx_feats = None
        if use_adapter and adapter is not None:
            n = traj.shape[0]
            ctx_windows = []
            for ws in starts:
                end = ws + CTX_LEN
                ctx_windows.append(traj[ws:end, :].copy() if end <= n else traj[-CTX_LEN:, :].copy())
            ctx_feats = np.array(ctx_windows, dtype=np.float32)

        bpred_list = []
        for b in range(0, n_total, 128):
            be = min(b + 128, n_total)
            hb = torch.from_numpy(all_hist[b:be]).to(device)
            ctx_inj = None
            if ctx_feats is not None:
                cb = torch.from_numpy(ctx_feats[b:be]).to(device)
                with torch.no_grad():
                    ctx_inj = adapter(cb)
            kwargs = {'force_predict': True}
            if ctx_inj is not None:
                kwargs['context_injection'] = ctx_inj
            with torch.no_grad():
                bpred_list.append(model(hb, **kwargs)['predictions'].cpu())
        bpred = torch.cat(bpred_list, dim=0)

        futs_t = [torch.from_numpy(t).float() for t in futs]
        futs_stack = torch.stack(futs_t)

        ade_vals = torch.norm(bpred - futs_stack, dim=-1).mean(dim=1).numpy()
        fde_vals = torch.norm(bpred[:, -1, :] - futs_stack[:, -1, :], dim=-1).numpy()
        dir_vals = np.array([dir_err(bpred[i, -1, :2].numpy(), futs[i][-1, :2])
                            for i in range(n_total)])

        # First-step boundary gap
        hist_last_pos = all_hist[:, -1, :3]
        gap_vals = np.linalg.norm(bpred[:, 0, :].numpy(), axis=1)  # first displacement magnitude

        return {
            'ade_mean': float(np.mean(ade_vals)), 'fde_mean': float(np.mean(fde_vals)),
            'fde_median': float(np.median(fde_vals)), 'fde_p95': float(np.percentile(fde_vals, 95)),
            'dir_mean': float(np.mean(dir_vals)),
            'gap_mean': float(np.mean(gap_vals)), 'gap_max': float(np.max(gap_vals)),
            'cata_pct': float(np.sum(dir_vals >= 90) / len(dir_vals) * 100),
            'n_wins': n_total,
        }
    finally:
        model.ua_pgd.physics_gate.forward = orig_gate


def main():
    print('=' * 80)
    print('BASE MODEL ABLATION — Why does it deviate so much?')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device

    adapter = ContextAdapterV2(input_dim=6, context_len=CTX_LEN,
                                d_model=model.d_model, hidden=128).to(device)
    adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=device))
    adapter.eval()

    # Collect trajectories by length
    short, medium, long_t = [], [], []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        n = d['traj'].shape[0]
        if n < 80: short.append((f.name, d['traj'], n))
        elif n < 150: medium.append((f.name, d['traj'], n))
        else: long_t.append((f.name, d['traj'], n))

    groups = [
        ('SHORT (<80fr)', short[:MAX_TRAJ_PER_GROUP]),
        ('MEDIUM (80-150fr)', medium[:MAX_TRAJ_PER_GROUP]),
        ('LONG (>=150fr)', long_t[:MAX_TRAJ_PER_GROUP]),
    ]

    configs = [
        ('1.RAW (no fix)', 1, 1.0, False),
        ('2.+A (stride=2)', 2, 1.0, False),
        ('3.+C (gate=0.3)', 1, 0.3, False),
        ('4.+AC (A+C)', 2, 0.3, False),
        ('5.+Adapter', 1, 1.0, True),
        ('6.+AC+Adapter (current)', 2, 0.3, True),
    ]

    all_results = {}

    for group_name, trajectories in groups:
        print(f'\n{"=" * 80}')
        print(f'  {group_name} — {len(trajectories)} trajectories')
        print(f'{"=" * 80}')

        group_results = {}
        for cfg_name, stride, gs, use_adpt in configs:
            all_ade, all_fde, all_dir, all_gap, all_cata = [], [], [], [], []
            n_traj, n_wins = 0, 0

            for name, traj, nf in trajectories:
                r = eval_config(model, device, adapter, traj, stride, gs, use_adpt)
                if r is None: continue
                n_traj += 1; n_wins += r['n_wins']
                all_ade.append(r['ade_mean'] * r['n_wins'])
                all_fde.append(r['fde_mean'] * r['n_wins'])
                all_dir.append(r['dir_mean'] * r['n_wins'])
                all_gap.append(r['gap_mean'] * r['n_wins'])
                all_cata.append(r['cata_pct'] * r['n_wins'])

            if n_wins == 0: continue
            w_ade = sum(all_ade) / n_wins
            w_fde = sum(all_fde) / n_wins
            w_dir = sum(all_dir) / n_wins
            w_gap = sum(all_gap) / n_wins
            w_cata = sum(all_cata) / n_wins

            group_results[cfg_name] = {
                'ade': w_ade, 'fde': w_fde, 'dir': w_dir,
                'gap': w_gap, 'cata': w_cata, 'n_traj': n_traj, 'n_wins': n_wins,
            }

        # Print table for this group
        print(f'\n  {"Config":<25} {"ADE":<8} {"FDE":<8} {"Dir":<7} {"Gap":<8} {"Cata%":<7} {"Wins":<6}')
        print(f'  {"-" * 70}')
        for cfg_name, _, _, _ in configs:
            if cfg_name not in group_results: continue
            r = group_results[cfg_name]
            print(f'  {cfg_name:<25} {r["ade"]:<8.3f} {r["fde"]:<8.3f} {r["dir"]:<7.1f} '
                  f'{r["gap"]:<8.3f} {r["cata"]:<7.1f} {r["n_wins"]:<6}')

        # Highlight the best and worst for FDE
        best_fde = min(group_results.values(), key=lambda x: x['fde'])
        worst_fde = max(group_results.values(), key=lambda x: x['fde'])
        best_cfg = [k for k, v in group_results.items() if v['fde'] == best_fde['fde']][0]
        worst_cfg = [k for k, v in group_results.items() if v['fde'] == worst_fde['fde']][0]
        print(f'\n  Best FDE: {best_cfg} ({best_fde["fde"]:.3f}m)')
        print(f'  Worst FDE: {worst_cfg} ({worst_fde["fde"]:.3f}m)')

        all_results[group_name] = group_results

    # ── Cross-group comparison table ──
    print(f'\n{"=" * 80}')
    print(f'CROSS-GROUP COMPARISON — FDE (m)')
    print(f'{"=" * 80}')
    print(f'  {"Config":<25} {"SHORT":<10} {"MEDIUM":<10} {"LONG":<10}')
    print(f'  {"-" * 55}')
    for cfg_name, _, _, _ in configs:
        vals = []
        for gn in [g[0] for g in groups]:
            if gn in all_results and cfg_name in all_results[gn]:
                vals.append(f'{all_results[gn][cfg_name]["fde"]:.3f}')
            else:
                vals.append('—')
        print(f'  {cfg_name:<25} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10}')

    # ── Gap comparison ──
    print(f'\n{"=" * 80}')
    print(f'CROSS-GROUP COMPARISON — Boundary Gap (m)')
    print(f'{"=" * 80}')
    print(f'  {"Config":<25} {"SHORT":<10} {"MEDIUM":<10} {"LONG":<10}')
    print(f'  {"-" * 55}')
    for cfg_name, _, _, _ in configs:
        vals = []
        for gn in [g[0] for g in groups]:
            if gn in all_results and cfg_name in all_results[gn]:
                vals.append(f'{all_results[gn][cfg_name]["gap"]:.3f}')
            else:
                vals.append('—')
        print(f'  {cfg_name:<25} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10}')

    # ── Conclusions ──
    print(f'\n{"=" * 80}')
    print('DIAGNOSIS:')
    if 'LONG (>=150fr)' in all_results:
        long_data = all_results['LONG (>=150fr)']
        raw = long_data.get('1.RAW (no fix)', {})
        ac = long_data.get('6.+AC+Adapter (current)', {})
        if raw and ac:
            print(f'  RAW FDE:     {raw.get("fde", 0):.3f}m  Gap: {raw.get("gap", 0):.3f}m')
            print(f'  Current FDE: {ac.get("fde", 0):.3f}m  Gap: {ac.get("gap", 0):.3f}m')
            fde_delta = (ac.get("fde", 0) - raw.get("fde", 0)) / max(raw.get("fde", 0.001), 1e-6) * 100
            print(f'  FDE change:  {fde_delta:+.1f}%')

    print(f'\n  All results saved for analysis.')
    print('=' * 80)

    # Save JSON
    json.dump({gn: {cn: r for cn, r in gr.items()} for gn, gr in all_results.items()},
              open(OUT_DIR / 'base_ablation.json', 'w'), indent=2, default=str)
    print(f'  Saved: pic-results/base_ablation.json')


if __name__ == '__main__':
    main()
