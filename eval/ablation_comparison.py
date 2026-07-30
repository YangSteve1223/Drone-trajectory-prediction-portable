#!/usr/bin/env python3
"""
Comprehensive ablation: compare 5 configurations for LOW (40-frame, UAV-Flow).

  A) Single-head 40f base (frozen)
  B) Multi-head K=5 (frozen)
  C) Multi-head K=5 + direction fallback (frozen)
  D) Single-head + online LoRA (current: accum=10, lr=3e-5)
  E) Single-head + online LoRA (improved: accum=5, lr=5e-5)
  F) Multi-head K=5 + direction fallback + online LoRA (improved config)

All use NO global LoRA. Causal protocol: warmup 60%, eval on held-out 40%.
Reports FDE, ADE, Dir, catastrophic rate (<90 deg), and per-drone breakdown.

Output: pic-results/ablation_comparison.json
"""

import torch, numpy as np, sys, json, shutil, math
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import online_config as OC
from adapter_manager import DroneAdapterManager
from online_learner import OnlineLearner, OnlineLearnerConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_FRAC = 0.6
MIN_LEN = 200
N_DRONES = 30
HIST_LEN = OC.HIST_LEN
PRED_LEN = OC.PRED_LEN
DT = 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv)/(pn*tn), -1.0, 1.0))))

def make_adaptive_windows(traj, hist_len=40):
    n = traj.shape[0]
    stride = max(1, min(4, n // 60))
    ml = hist_len * stride + PRED_LEN
    if n < ml: return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + hist_len*stride, stride)[:hist_len]
        if len(indices) < hist_len: continue
        hists.append(traj[indices].copy())
        fut_start = i + hist_len*stride
        fut_end = fut_start + PRED_LEN
        if fut_end > n: continue
        fut_abs = traj[fut_start:fut_end, :3]
        hists[-1][:, :3] -= hists[-1][-1, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs

def compute_errors_single(model, H, T, batch=64):
    """Forward single-head model, return FDE/ADE/dir per window."""
    preds = []
    for b in range(0, len(H), batch):
        be = min(b + batch, len(H))
        with torch.no_grad():
            preds.append(model(H[b:be].to(DEVICE), force_predict=True)['predictions'].cpu())
    P = torch.cat(preds, dim=0)
    return _compute_metrics(P, T)

def compute_errors_multihead(model, H, T, batch=64):
    """Forward multi-head model via forward_multi_head, return metrics."""
    P_list = []
    for b in range(0, len(H), batch):
        be = min(b + batch, len(H))
        hb = H[b:be].to(DEVICE)
        scale = hb.new_tensor([100., 100., 100., 10., 10., 10.])
        h_norm = hb / scale.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            encoded = model.emam_se(h_norm)
            dtp_out = model.ia_dtp(encoded, historical_trajectory=h_norm)
            out = model.ua_pgd.forward_multi_head(
                encoded_feat=encoded,
                global_anchor=dtp_out['global_anchor'],
                historical_trajectory=h_norm,
                intent_weights=dtp_out.get('intent_weights'),
            )
        P_list.append(out['predictions'].cpu())
    P = torch.cat(P_list, dim=0)
    return _compute_metrics(P, T)

def compute_errors_multihead_fallback(model, H, T, batch=64):
    """Multi-head + direction fallback."""
    P_list = []
    fell_back_count = 0
    for b in range(0, len(H), batch):
        be = min(b + batch, len(H))
        hb = H[b:be].to(DEVICE)
        scale = hb.new_tensor([100., 100., 100., 10., 10., 10.])
        h_norm = hb / scale.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            encoded = model.emam_se(h_norm)
            dtp_out = model.ia_dtp(encoded, historical_trajectory=h_norm)
            out = model.ua_pgd.forward_multi_head(
                encoded_feat=encoded,
                global_anchor=dtp_out['global_anchor'],
                historical_trajectory=h_norm,
                intent_weights=dtp_out.get('intent_weights'),
            )
        preds = out['predictions']

        # Direction fallback
        last_vel = hb[:, -1, 3:6]
        vel_recent = hb[:, -3:, 3:6]
        w = torch.tensor([0.2, 0.3, 0.5], device=hb.device)
        last_vel_smooth = (vel_recent * w.view(1, 3, 1)).sum(dim=1)
        steps = torch.arange(1, PRED_LEN + 1, device=hb.device).float()
        cv_pred = last_vel_smooth.unsqueeze(1) * (steps.view(1, -1, 1) * DT)

        pred_dir = preds[:, -1, :2]
        cv_dir = cv_pred[:, -1, :2]
        cos_sim = F.cosine_similarity(pred_dir + 1e-8, cv_dir + 1e-8, dim=-1)
        angle = torch.acos(cos_sim.clamp(-0.999, 0.999)) * (180.0 / math.pi)
        mask = (angle > 90.0).float().unsqueeze(-1).unsqueeze(-1)
        fell_back_count += int(mask.any(dim=-1).any(dim=-1).sum().item())
        preds = preds * (1 - mask) + cv_pred * mask

        P_list.append(preds.cpu())
    P = torch.cat(P_list, dim=0)
    metrics = _compute_metrics(P, T)
    metrics['fell_back'] = fell_back_count
    return metrics

def _compute_metrics(P, T):
    """Return dict of per-window and aggregate metrics."""
    fde = torch.norm(P[:, -1, :] - T[:, -1, :], dim=-1)
    ade = torch.norm(P - T, dim=-1).mean(dim=1)
    dirs = np.array([dir_err(P[i, -1, :2].numpy(), T[i, -1, :2].numpy())
                     for i in range(len(P))])
    n_cata = int((dirs >= 90).sum())
    return {
        'fde_mean': float(fde.mean()), 'fde_median': float(fde.median()),
        'ade_mean': float(ade.mean()),
        'dir_mean': float(dirs.mean()), 'dir_median': float(np.median(dirs)),
        'n_windows': len(P), 'n_cata': n_cata,
        'cata_rate': n_cata / len(P) * 100,
        'fde_per': fde.numpy().tolist(),
        'dir_per': dirs.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('COMPREHENSIVE ABLATION — LOW (40-frame, UAV-Flow, 30 drones)')
    print('  Comparing 6 configurations. No global LoRA. Causal protocol.')
    print('=' * 80)

    # ── Load trajectories ──
    TRAJ_DIR = ROOT / 'UAV-Flow-trajs'
    CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / '_ablation_cmp'
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

    trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); t = d['traj']
        if t.shape[0] >= MIN_LEN:
            trajs.append((f.stem, t))
    np.random.seed(11); np.random.shuffle(trajs)
    trajs = trajs[:N_DRONES]
    print(f'  {len(trajs)} drones (>= {MIN_LEN} frames)\n')

    # Pre-compute windows for all drones
    all_windows = {}
    for name, traj in trajs:
        hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
        if len(hists) < 30: continue
        H = torch.from_numpy(np.array(hists, dtype=np.float32))
        T = torch.from_numpy(np.array(futs, dtype=np.float32))
        n_warm = int(len(hists) * WARMUP_FRAC)
        if n_warm < 20 or len(hists) - n_warm < 5: continue
        all_windows[name] = (H, T, n_warm, len(hists))
    print(f'  {len(all_windows)} valid drones\n')

    # ── Build models ──
    print('Building models...')
    single_base = OC.build_base_model(device=DEVICE)
    multi_base = OC.build_multihead_base(device=DEVICE, K=5)
    print('  Single-head + Multi-head ready.\n')

    # ── Configs to evaluate ──
    configs = {
        'A_single_frozen': {
            'desc': 'Single-head (frozen)',
            'model': single_base,
            'eval_fn': lambda m, H, T: compute_errors_single(m, H, T),
        },
        'B_multihead_frozen': {
            'desc': 'Multi-head K=5 (frozen)',
            'model': multi_base,
            'eval_fn': lambda m, H, T: compute_errors_multihead(m, H, T),
        },
        'C_multihead_fallback_frozen': {
            'desc': 'Multi-head K=5 + dir fallback (frozen)',
            'model': multi_base,
            'eval_fn': lambda m, H, T: compute_errors_multihead_fallback(m, H, T),
        },
    }

    # D/E/F require online learning — evaluated separately per drone
    online_configs = {
        'D_single_online_current': {
            'desc': 'Single-head + online LoRA (accum=10, lr=3e-5)',
            'accum': 10, 'lr': 3e-5,
        },
        'E_single_online_improved': {
            'desc': 'Single-head + online LoRA (accum=5, lr=5e-5)',
            'accum': 5, 'lr': 5e-5,
        },
        'F_multihead_fallback_online': {
            'desc': 'Multi-head + fallback + online LoRA (accum=5, lr=5e-5)',
            'accum': 5, 'lr': 5e-5, 'multihead': True, 'fallback': True,
        },
    }

    all_results = {}

    # ── A, B, C: frozen evaluation ──
    for cfg_key in ['A_single_frozen', 'B_multihead_frozen', 'C_multihead_fallback_frozen']:
        cfg = configs[cfg_key]
        print(f'\n[{cfg_key}] {cfg["desc"]}')
        model = cfg['model']
        eval_fn = cfg['eval_fn']
        per_drone = []
        all_fde, all_dir = [], []
        total_cata = 0; total_win = 0

        for name, (H, T, n_warm, n_tot) in all_windows.items():
            Hh, Th = H[n_warm:], T[n_warm:]
            metrics = eval_fn(model, Hh, Th)
            per_drone.append({'drone': name, **{k: v for k, v in metrics.items()
                             if k not in ('fde_per', 'dir_per')}})
            all_fde.append(metrics['fde_mean'])
            all_dir.append(metrics['dir_mean'])
            total_cata += metrics['n_cata']
            total_win += metrics['n_windows']

        fr_fde = np.array(all_fde)
        fr_dir = np.array(all_dir)
        all_results[cfg_key] = {
            'desc': cfg['desc'],
            'fde_mean': float(fr_fde.mean()), 'fde_median': float(np.median(fr_fde)),
            'dir_mean': float(fr_dir.mean()),
            'total_windows': total_win, 'total_cata': total_cata,
            'cata_rate': total_cata / total_win * 100,
            'per_drone': per_drone,
        }
        print(f'  FDE: {fr_fde.mean():.4f}  Dir: {fr_dir.mean():.2f}deg  '
              f'Cata: {total_cata}/{total_win} ({total_cata/total_win*100:.1f}%)')

    # ── D, E, F: online learning evaluation ──
    for cfg_key, cfg in online_configs.items():
        print(f'\n[{cfg_key}] {cfg["desc"]}')
        is_mh = cfg.get('multihead', False)
        has_fb = cfg.get('fallback', False)

        online_base = OC.build_multihead_base(device=DEVICE, K=5) if is_mh else OC.build_base_model(device=DEVICE)
        mgr = DroneAdapterManager(
            OC.build_multihead_base(device=DEVICE, K=5) if is_mh else OC.build_base_model(device=DEVICE),
            checkpoint_dir=str(CKPT_DIR))
        ocfg = OnlineLearnerConfig(accumulation_steps=cfg['accum'], lr=cfg['lr'],
                                   device=str(DEVICE), conf_threshold=0.0)
        learner = OnlineLearner(mgr, ocfg)

        per_drone = []
        all_fde, all_dir = [], []
        total_cata = 0; total_win = 0; total_fb = 0

        for name, (H, T, n_warm, n_tot) in all_windows.items():
            Hw, Tw = H[:n_warm], T[:n_warm]
            Hh, Th = H[n_warm:], T[n_warm:]

            learner.reset_drone(name)
            for j in range(n_warm):
                learner.observe(name, Hw[j], Tw[j], confidence=1.0, timestep=j)

            if mgr.adapter is not None and mgr.active_drone == name:
                online_model = mgr.adapter.model
            else:
                mgr.activate(name); online_model = mgr.adapter.model

            if has_fb:
                metrics = compute_errors_multihead_fallback(online_model, Hh, Th)
                total_fb += metrics.get('fell_back', 0)
            elif is_mh:
                metrics = compute_errors_multihead(online_model, Hh, Th)
            else:
                metrics = compute_errors_single(online_model, Hh, Th)
            mgr.deactivate()

            per_drone.append({'drone': name, **{k: v for k, v in metrics.items()
                             if k not in ('fde_per', 'dir_per')}})
            all_fde.append(metrics['fde_mean'])
            all_dir.append(metrics['dir_mean'])
            total_cata += metrics['n_cata']
            total_win += metrics['n_windows']

        fr_fde = np.array(all_fde)
        fr_dir = np.array(all_dir)
        n_better = int((fr_fde < np.array([r['fde_mean'] for r in all_results['A_single_frozen']['per_drone']])).sum())
        result = {
            'desc': cfg['desc'],
            'fde_mean': float(fr_fde.mean()), 'fde_median': float(np.median(fr_fde)),
            'dir_mean': float(fr_dir.mean()),
            'total_windows': total_win, 'total_cata': total_cata,
            'cata_rate': total_cata / total_win * 100,
            'n_better_vs_A': n_better,
            'per_drone': per_drone,
        }
        if has_fb:
            result['direction_fallbacks'] = total_fb
        all_results[cfg_key] = result
        fb_str = f'  Fallbacks: {total_fb}' if has_fb else ''
        print(f'  FDE: {fr_fde.mean():.4f}  Dir: {fr_dir.mean():.2f}deg  '
              f'Cata: {total_cata}/{total_win} ({total_cata/total_win*100:.1f}%)  '
              f'Better vs A: {n_better}/{len(all_fde)}{fb_str}')

    # ── Final summary table ──
    print(f'\n{"=" * 100}')
    print('FINAL COMPARISON')
    print(f'{"=" * 100}')
    print(f'  {"Config":<44} {"FDE":>8} {"Dir":>7} {"Cata":>8} {"vs_A":>8}')
    print(f'  {"-" * 44} {"-" * 8} {"-" * 7} {"-" * 8} {"-" * 8}')

    a_fde = all_results['A_single_frozen']['fde_mean']
    for cfg_key in ['A_single_frozen', 'B_multihead_frozen', 'C_multihead_fallback_frozen',
                     'D_single_online_current', 'E_single_online_improved',
                     'F_multihead_fallback_online']:
        r = all_results[cfg_key]
        fde = r['fde_mean']
        gain = (a_fde - fde) / a_fde * 100
        vs_a = f'{gain:+.1f}%'
        if 'n_better_vs_A' in r:
            vs_a += f' ({r["n_better_vs_A"]}/{r["total_windows"]}d)'
        print(f'  {r["desc"]:<44} {fde:>7.4f}m {r["dir_mean"]:>6.1f}deg '
              f'{r["cata_rate"]:>7.1f}% {vs_a:>8}')

    # Extra row for fallback count if available
    for cfg_key in ['C_multihead_fallback_frozen', 'F_multihead_fallback_online']:
        if 'direction_fallbacks' in all_results.get(cfg_key, {}):
            r = all_results[cfg_key]
            print(f'    -> direction fallbacks triggered: {r["direction_fallbacks"]}')

    print(f'{"=" * 100}')

    json.dump(all_results, open(OUT_DIR / 'ablation_comparison.json', 'w'), indent=2)
    print(f'\nSaved: pic-results/ablation_comparison.json')
    shutil.rmtree(CKPT_DIR, ignore_errors=True)


if __name__ == '__main__':
    main()
