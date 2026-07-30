#!/usr/bin/env python3
"""
Final ablation: Base model vs Base + Online LoRA + Direction Fallback (bundled).

Direction fallback is ON by default — it's a safety net, not a separate config.
Covers both LOW (40-frame, UAV-Flow) and HIGH (20-frame, SimCruise).

Generates: 3D + XY top-down + per-second error LINE chart + table.
  LOW:  markers at 0,1,2,3,4s  (5 Hz × 4s = 20 steps)
  HIGH: markers at 0,5,10,15,20s (1 Hz × 20s = 20 steps)

Output: pic-results/summarize/
"""

import torch, numpy as np, sys, warnings, json, shutil, math
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import online_config as OC
from adapter_manager import DroneAdapterManager
from online_learner import OnlineLearner, OnlineLearnerConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results' / 'summarize'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LOW config
LOW_WARMUP_FRAC = 0.6
LOW_ACCUM = 5
LOW_LR = 5e-5
LOW_N_DRONES = 30
LOW_HIST_LEN = 40
LOW_PRED_LEN = 20
LOW_DT = 0.2
LOW_SEC_MARKS = {0: 0, 1: 5, 2: 10, 3: 15, 4: 19}   # step index for each second

# HIGH config
HIGH_WARMUP_FRAC = 0.6
HIGH_ACCUM = 10
HIGH_LR = 3e-5
HIGH_N_DRONES = 25
HIGH_HIST_LEN = 20
HIGH_PRED_LEN = 20
HIGH_DT = 1.0
HIGH_SEC_MARKS = {0: 0, 5: 5, 10: 10, 15: 15, 20: 19}

N_DISPLAY = 14
DIR_THRESHOLD_DEG = 90.0

# Colours
CH = {'hist': '#1565C0', 'base': '#D32F2F', 'lora': '#FF6D00', 'truth': '#2E7D32'}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 7.5,
                     'font.family': 'sans-serif', 'figure.dpi': 150})


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


# ═══════════════════════════════════════════════════════════════════════════════
# Window builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_windows_low(traj):
    n = traj.shape[0]
    stride = max(1, min(4, n // 60))
    ml = LOW_HIST_LEN * stride + LOW_PRED_LEN
    if n < ml: return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + LOW_HIST_LEN * stride, stride)[:LOW_HIST_LEN]
        if len(indices) < LOW_HIST_LEN: continue
        hists.append(traj[indices].copy())
        fut_start = i + LOW_HIST_LEN * stride
        fut_end = fut_start + LOW_PRED_LEN
        if fut_end > n: continue
        fut_abs = traj[fut_start:fut_end, :3]
        hists[-1][:, :3] -= hists[-1][-1, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs


def make_windows_high(traj):
    n = traj.shape[0]
    H, T = [], []
    for j in range(HIGH_HIST_LEN - 1, n - HIGH_PRED_LEN):
        h = traj[j - HIGH_HIST_LEN + 1:j + 1].copy()
        fut_abs = traj[j + 1:j + 1 + HIGH_PRED_LEN, :3]
        if fut_abs.shape[0] < HIGH_PRED_LEN: continue
        fut = fut_abs - traj[j, :3]
        h[:, :3] -= h[0, :3]
        H.append(h); T.append(fut)
    return H, T


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback helpers
# ═══════════════════════════════════════════════════════════════════════════════

def const_vel_prediction(hist, pred_len, dt):
    last_vel = hist[:, -1, 3:6]
    vel_recent = hist[:, -3:, 3:6]
    w = torch.tensor([0.2, 0.3, 0.5], device=hist.device)
    last_vel_smooth = (vel_recent * w.view(1, 3, 1)).sum(dim=1)
    steps = torch.arange(1, pred_len + 1, device=hist.device).float()
    return last_vel_smooth.unsqueeze(1) * (steps.view(1, -1, 1) * dt)


def apply_direction_fallback(predictions, hist, pred_len, dt):
    cv_pred = const_vel_prediction(hist, pred_len, dt)
    pred_dir = predictions[:, -1, :2]
    cv_dir = cv_pred[:, -1, :2]
    cos_sim = F.cosine_similarity(pred_dir + 1e-8, cv_dir + 1e-8, dim=-1)
    angle = torch.acos(cos_sim.clamp(-0.999, 0.999)) * (180.0 / math.pi)
    mask = (angle > DIR_THRESHOLD_DEG).float().unsqueeze(-1).unsqueeze(-1)
    safe_pred = predictions * (1 - mask) + cv_pred * mask
    return safe_pred, mask.bool(), angle


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_base_errors(model, H, T, batch=64):
    preds = []
    for b in range(0, len(H), batch):
        be = min(b + batch, len(H))
        with torch.no_grad():
            preds.append(model(H[b:be].to(DEVICE), force_predict=True)['predictions'].cpu())
    P = torch.cat(preds, dim=0)
    fde = torch.norm(P[:, -1, :] - T[:, -1, :], dim=-1)
    ade = torch.norm(P - T, dim=-1).mean(dim=1)
    dirs = np.array([dir_err(P[i, -1, :2].numpy(), T[i, -1, :2].numpy()) for i in range(len(P))])
    return fde, ade, dirs, P


def compute_adapted_errors(model, H, T, pred_len, dt, batch=64):
    """Model forward + direction fallback (bundled)."""
    preds_safe, preds_raw = [], []
    fell_back_count = 0
    angles_all = []
    for b in range(0, len(H), batch):
        be = min(b + batch, len(H))
        hb = H[b:be].to(DEVICE)
        with torch.no_grad():
            raw = model(hb, force_predict=True)['predictions']
        safe, fb_mask, angles = apply_direction_fallback(raw, hb, pred_len, dt)
        preds_raw.append(raw.cpu()); preds_safe.append(safe.cpu())
        fell_back_count += int(fb_mask.any(dim=-1).any(dim=-1).sum().item())
        angles_all.append(angles.cpu())
    P_safe = torch.cat(preds_safe, dim=0)
    angles_all = torch.cat(angles_all, dim=0)
    fde = torch.norm(P_safe[:, -1, :] - T[:, -1, :], dim=-1)
    ade = torch.norm(P_safe - T, dim=-1).mean(dim=1)
    dirs = np.array([dir_err(P_safe[i, -1, :2].numpy(), T[i, -1, :2].numpy()) for i in range(len(P_safe))])
    n_cata = int((dirs >= 90).sum())
    return fde, ade, dirs, P_safe, fell_back_count, n_cata, angles_all


def per_second_errors(P, T, sec_marks):
    """Per-second position error.  P: (pred_len, 3) or (N, pred_len, 3)."""
    errs = {}
    for sec, idx in sec_marks.items():
        if idx < P.shape[0]:   # FIXED: was .shape[1] (coordinate dim) — now .shape[0] (time dim)
            errs[sec] = float(torch.norm(P[idx] - T[idx], dim=-1).item())
    return errs


# ═══════════════════════════════════════════════════════════════════════════════
# LOW ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_low():
    print('=' * 80)
    print('LOW (40-frame, UAV-Flow, 5 Hz) — Base vs +LoRA+Fallback')
    print('=' * 80)

    TRAJ_DIR = ROOT / 'UAV-Flow-trajs'
    CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / '_abl_final_low'
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

    trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); t = d['traj']
        if t.shape[0] >= 200: trajs.append((f.stem, t))
    np.random.seed(11); np.random.shuffle(trajs)
    trajs = trajs[:LOW_N_DRONES]
    print(f'  {len(trajs)} drones\n')

    base = OC.build_online_base(device=DEVICE, with_global=False)
    mgr = DroneAdapterManager(OC.build_online_base(device=DEVICE, with_global=False),
                              checkpoint_dir=str(CKPT_DIR))
    ocfg = OnlineLearnerConfig(accumulation_steps=LOW_ACCUM, lr=LOW_LR, device=str(DEVICE),
                               conf_threshold=0.0)
    learner = OnlineLearner(mgr, ocfg)

    results, display_data = [], []
    for i, (drone_id, traj) in enumerate(trajs):
        hists, futs = make_windows_low(traj)
        if len(hists) < 30: continue
        H = torch.from_numpy(np.array(hists, dtype=np.float32))
        T = torch.from_numpy(np.array(futs, dtype=np.float32))
        n = len(hists); n_warm = int(n * LOW_WARMUP_FRAC)
        if n_warm < LOW_ACCUM * 2 or (n - n_warm) < 5: continue

        Hw, Tw = H[:n_warm], T[:n_warm]
        Hh, Th = H[n_warm:], T[n_warm:]

        fde_fr, ade_fr, dir_fr, P_fr = compute_base_errors(base, Hh, Th)

        learner.reset_drone(drone_id)
        n_updates = 0
        for j in range(n_warm):
            if learner.observe(drone_id, Hw[j], Tw[j], confidence=1.0, timestep=j):
                n_updates += 1

        if mgr.adapter is not None and mgr.active_drone == drone_id:
            ad_model = mgr.adapter.model
        else:
            mgr.activate(drone_id); ad_model = mgr.adapter.model

        fde_ad, ade_ad, dir_ad, P_ad, n_fb, n_cata_ad, angles_ad = \
            compute_adapted_errors(ad_model, Hh, Th, LOW_PRED_LEN, LOW_DT)
        mgr.deactivate()

        n_cata_fr = int((dir_fr >= 90).sum())
        fde_fr_m = float(fde_fr.mean()); fde_ad_m = float(fde_ad.mean())
        gain = (fde_fr_m - fde_ad_m) / max(fde_fr_m, 1e-6) * 100

        results.append(dict(drone=drone_id, n_warm=n_warm, n_heldout=int(n-n_warm),
                            n_updates=n_updates, base_fde=fde_fr_m, lora_fde=fde_ad_m,
                            base_dir=float(dir_fr.mean()), lora_dir=float(dir_ad.mean()),
                            base_cata=n_cata_fr, lora_cata=n_cata_ad,
                            n_fell_back=n_fb, fde_gain=gain))
        tag = 'GAIN' if gain > 0 else 'LOSS'
        print(f'  [{i+1:2d}] {drone_id[:28]:28s} BASE={fde_fr_m:.3f}m  LORA={fde_ad_m:.3f}m  '
              f'({gain:+.1f}%) [{tag}]  cata:{n_cata_fr}→{n_cata_ad}  FB:{n_fb}')

        display_data.append(dict(drone=drone_id, traj=traj, n_warm=n_warm,
                                 fde_fr_per=fde_fr.numpy(), fde_ad_per=fde_ad.numpy(),
                                 dir_fr=dir_fr, dir_ad=dir_ad, gain=gain, n_fell_back=n_fb))

    if not results: print('No valid drones.'); return None
    return _summarize(results, display_data, "LOW", LOW_SEC_MARKS, LOW_PRED_LEN, LOW_DT,
                      LOW_HIST_LEN, LOW_ACCUM, make_windows_low, base, mgr, learner, CKPT_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_high():
    print('\n' + '=' * 80)
    print('HIGH (20-frame, SimCruise, 1 Hz) — Base vs +LoRA+Fallback')
    print('=' * 80)

    SIM_DIR = ROOT / 'SimCruise'
    CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / '_abl_final_high'
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

    merged = sorted(SIM_DIR.rglob('trajectories_merged.npz'))
    if not merged: print('  No trajectories_merged.npz — skipping HIGH.'); return None
    d = np.load(merged[0])
    pos, vel, mask = d['positions'], d['velocities'], d['masks']
    lengths = mask.sum(axis=1)
    idx = np.where(lengths >= 120)[0]
    rng = np.random.RandomState(11); rng.shuffle(idx)
    idx = idx[:HIGH_N_DRONES]
    trajs = []
    for i in idx:
        L = int(lengths[i])
        p = pos[i, :L].astype(np.float32); v = vel[i, :L].astype(np.float32)
        trajs.append((f'high_{i}', np.concatenate([p, v], axis=1)))
    print(f'  {len(trajs)} HIGH drones\n')

    base = OC.build_high_base(device=DEVICE)
    mgr = DroneAdapterManager(OC.build_high_base(device=DEVICE), checkpoint_dir=str(CKPT_DIR))
    ocfg = OnlineLearnerConfig(accumulation_steps=HIGH_ACCUM, lr=HIGH_LR, device=str(DEVICE),
                               conf_threshold=0.0, dt=HIGH_DT)
    learner = OnlineLearner(mgr, ocfg)

    results, display_data = [], []
    for i, (drone_id, traj) in enumerate(trajs):
        H_list, T_list = make_windows_high(traj)
        if len(H_list) < 30: continue
        H = torch.from_numpy(np.array(H_list, dtype=np.float32))
        T = torch.from_numpy(np.array(T_list, dtype=np.float32))
        n = len(H_list); n_warm = int(n * HIGH_WARMUP_FRAC)
        if n_warm < HIGH_ACCUM * 2 or (n - n_warm) < 5: continue

        Hw, Tw = H[:n_warm], T[:n_warm]
        Hh, Th = H[n_warm:], T[n_warm:]

        fde_fr, ade_fr, dir_fr, P_fr = compute_base_errors(base, Hh, Th)

        learner.reset_drone(drone_id)
        n_updates = 0
        for j in range(n_warm):
            if learner.observe(drone_id, Hw[j], Tw[j], confidence=1.0, timestep=j):
                n_updates += 1

        if mgr.adapter is not None and mgr.active_drone == drone_id:
            ad_model = mgr.adapter.model
        else:
            mgr.activate(drone_id); ad_model = mgr.adapter.model

        fde_ad, ade_ad, dir_ad, P_ad, n_fb, n_cata_ad, angles_ad = \
            compute_adapted_errors(ad_model, Hh, Th, HIGH_PRED_LEN, HIGH_DT)
        mgr.deactivate()

        n_cata_fr = int((dir_fr >= 90).sum())
        fde_fr_m = float(fde_fr.mean()); fde_ad_m = float(fde_ad.mean())
        gain = (fde_fr_m - fde_ad_m) / max(fde_fr_m, 1e-6) * 100

        results.append(dict(drone=drone_id, n_warm=n_warm, n_heldout=int(n-n_warm),
                            n_updates=n_updates, base_fde=fde_fr_m, lora_fde=fde_ad_m,
                            base_dir=float(dir_fr.mean()), lora_dir=float(dir_ad.mean()),
                            base_cata=n_cata_fr, lora_cata=n_cata_ad,
                            n_fell_back=n_fb, fde_gain=gain))
        tag = 'GAIN' if gain > 0 else 'LOSS'
        print(f'  [{i+1:2d}] {drone_id:12s} BASE={fde_fr_m:.3f}m  LORA={fde_ad_m:.3f}m  '
              f'({gain:+.1f}%) [{tag}]  FB:{n_fb}')

        display_data.append(dict(drone=drone_id, traj=traj, n_warm=n_warm,
                                 fde_fr_per=fde_fr.numpy(), fde_ad_per=fde_ad.numpy(),
                                 dir_fr=dir_fr, dir_ad=dir_ad, gain=gain, n_fell_back=n_fb))

    if not results: print('No valid HIGH drones.'); return None
    return _summarize(results, display_data, "HIGH", HIGH_SEC_MARKS, HIGH_PRED_LEN, HIGH_DT,
                      HIGH_HIST_LEN, HIGH_ACCUM, make_windows_high, base, mgr, learner, CKPT_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared summary + visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _summarize(results, display_data, tag, sec_marks, pred_len, dt, hist_len,
               accum, window_fn, base, mgr, learner, ckpt_dir):
    fr_fde = np.array([r['base_fde'] for r in results])
    ad_fde = np.array([r['lora_fde'] for r in results])
    fr_dir = np.array([r['base_dir'] for r in results])
    ad_dir = np.array([r['lora_dir'] for r in results])
    total_cata_fr = sum(r['base_cata'] for r in results)
    total_cata_ad = sum(r['lora_cata'] for r in results)
    total_fb = sum(r['n_fell_back'] for r in results)
    total_win = sum(r['n_heldout'] for r in results)
    n_better = int((ad_fde < fr_fde).sum())
    gain_pct = float((fr_fde.mean() - ad_fde.mean()) / fr_fde.mean() * 100)

    print(f'\n{"=" * 70}')
    print(f'{tag} FINAL RESULT ({len(results)} drones, {total_win} held-out windows)')
    print(f'{"=" * 70}')
    print(f'  {"Config":<30} {"FDE":>10} {"Dir":>10} {"Cata":>8}')
    print(f'  {"-"*30} {"-"*10} {"-"*10} {"-"*8}')
    print(f'  {"Base":<30} {fr_fde.mean():>10.4f} {fr_dir.mean():>10.2f} {total_cata_fr:>4}/{total_win}')
    print(f'  {"+LoRA":<30} {ad_fde.mean():>10.4f} {ad_dir.mean():>10.2f} {total_cata_ad:>4}/{total_win}')
    print(f'  {"="*70}')
    print(f'  FDE gain:  {gain_pct:+.2f}%  ({n_better}/{len(results)} drones better)')
    print(f'  Dir gain:  {fr_dir.mean()-ad_dir.mean():+.2f}°')
    print(f'  Fallbacks: {total_fb}')
    print(f'{"=" * 70}')

    json.dump({
        'tag': tag, 'n_drones': len(results), 'n_windows': total_win, 'n_better': n_better,
        'base': {'fde_mean': float(fr_fde.mean()), 'dir_mean': float(fr_dir.mean()), 'cata': total_cata_fr},
        'lora': {'fde_mean': float(ad_fde.mean()), 'dir_mean': float(ad_dir.mean()),
                 'cata': total_cata_ad, 'fallbacks': total_fb},
        'fde_gain_pct': gain_pct,
        'per_drone': sorted(results, key=lambda r: r['fde_gain'], reverse=True),
    }, open(OUT_DIR / f'ablation_{tag.lower()}_final.json', 'w'), indent=2)
    print(f'  Saved: pic-results/summarize/ablation_{tag.lower()}_final.json')

    _generate_plots(results, display_data, tag, sec_marks, pred_len, dt, hist_len,
                    accum, window_fn, base, mgr, learner)

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {'tag': tag, 'gain_pct': gain_pct, 'fde_base': float(fr_fde.mean()),
            'fde_lora': float(ad_fde.mean()), 'cata_base': total_cata_fr, 'cata_lora': total_cata_ad,
            'n_better': n_better, 'n_drones': len(results)}


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_plots(results, display_data, tag, sec_marks, pred_len, dt, hist_len,
                    accum, window_fn, base, mgr, learner):
    sorted_r = sorted(results, key=lambda r: r['fde_gain'], reverse=True)
    n = len(sorted_r)

    # Pick N_DISPLAY trajectories across the full gain spectrum
    picks = []
    # Best gain
    if n >= 1: picks.append(0)
    # Upper quartile
    if n >= 4: picks.append(max(1, n // 4))
    # Near median
    if n >= 6: picks.append(n // 2)
    # Lower quartile
    if n >= 8: picks.append(max(n // 2 + 1, 3 * n // 4))
    # Worst loss
    if n >= 3: picks.append(n - 1)
    # Second-worst loss
    if n >= 5: picks.append(n - 2)
    # Extra spread: ~15%, ~40%, ~60%, ~85% positions
    for frac in [0.15, 0.40, 0.60, 0.85]:
        cand = int(n * frac)
        if 0 < cand < n - 1 and cand not in picks:
            picks.append(cand)
    # Ensure worst-cata drone is picked
    by_cata = sorted(results, key=lambda r: r['base_cata'], reverse=True)
    worst = by_cata[0]
    if worst['base_cata'] > 0 and worst not in [sorted_r[p] for p in picks]:
        picks.append(next(i for i, r in enumerate(sorted_r) if r['drone'] == worst['drone']))
    picks = sorted(set(picks))[:N_DISPLAY]
    print(f'  Picks (by gain rank): {[f"#{p+1}" for p in picks]}')

    traj_map = {d['drone']: d['traj'] for d in display_data}
    data_map = {d['drone']: d for d in display_data}

    display = []
    for p in picks:
        r = sorted_r[p]
        name = r['drone']
        if name not in traj_map: continue
        traj = traj_map[name]
        dd = data_map[name]

        H_list, T_list = window_fn(traj)
        if len(H_list) < 20: continue
        H = torch.from_numpy(np.array(H_list, dtype=np.float32))
        T = torch.from_numpy(np.array(T_list, dtype=np.float32))
        n_warm = dd['n_warm']; n_tot = len(H_list)
        Hh, Th = H[n_warm:], T[n_warm:]

        learner.reset_drone(name)
        for j in range(n_warm):
            learner.observe(name, H[j], T[j], confidence=1.0, timestep=j)

        if mgr.adapter is not None and mgr.active_drone == name:
            ad_model = mgr.adapter.model
        else:
            mgr.activate(name); ad_model = mgr.adapter.model

        _, _, _, P_fr = compute_base_errors(base, Hh, Th)
        _, _, _, P_ad, n_fb, _, angles = compute_adapted_errors(ad_model, Hh, Th, pred_len, dt)
        mgr.deactivate()

        fde_fr_per = torch.norm(P_fr[:, -1, :] - Th[:, -1, :], dim=-1)
        fde_ad_per = torch.norm(P_ad[:, -1, :] - Th[:, -1, :], dim=-1)
        improvements = fde_fr_per - fde_ad_per
        best_idx = int(torch.argmax(improvements))

        # Fallback-triggered window (for LOW cata drone)
        fb_idx = None
        if n_fb > 0:
            diff_raw = torch.norm(P_ad[:, -1, :] - Th[:, -1, :], dim=-1)
            diff_safe = torch.norm(P_ad[:, -1, :] - Th[:, -1, :], dim=-1)
            # Actually we need raw vs safe separately... just find windows where fallback changed things
            try:
                fb_candidates = torch.where((fde_fr_per - fde_ad_per) > 0.5)[0]
                if len(fb_candidates) > 0:
                    fb_idx = int(fb_candidates[0])
                    if fb_idx == best_idx and len(fb_candidates) > 1:
                        fb_idx = int(fb_candidates[1])
            except: pass

        for idx, suffix in [(best_idx, ''), (fb_idx, '_fallback')]:
            if idx is None: continue
            if idx >= len(H_list) - n_warm: continue

            per_sec_fr = per_second_errors(P_fr[idx], Th[idx], sec_marks)
            per_sec_ad = per_second_errors(P_ad[idx], Th[idx], sec_marks)

            hist_last = H_list[n_warm + idx][-1, :3]
            hp_abs = H_list[n_warm + idx][:, :3]
            fp_abs = P_fr[idx].numpy() + hist_last
            ap_abs = P_ad[idx].numpy() + hist_last
            tp_abs = T_list[n_warm + idx] + hist_last

            dir_fr_val = dir_err(P_fr[idx, -1, :2].numpy(), Th[idx, -1, :2].numpy())
            dir_ad_val = dir_err(P_ad[idx, -1, :2].numpy(), Th[idx, -1, :2].numpy())
            fb_triggered = bool(idx < len(angles) and float(angles[idx]) > DIR_THRESHOLD_DEG)

            display.append(dict(
                name=name, rank=p + 1, gain=float(dd['gain']),
                base_fde=float(fde_fr_per[idx]), lora_fde=float(fde_ad_per[idx]),
                base_dir=dir_fr_val, lora_dir=dir_ad_val,
                fb_triggered=fb_triggered,
                hp=hp_abs, fp=fp_abs, ap=ap_abs, tp=tp_abs,
                per_sec_fr=per_sec_fr, per_sec_ad=per_sec_ad,
                suffix=suffix, sec_labels=list(sec_marks.keys()),
            ))

        learner.reset_drone(name)

    print(f'  {len(display)} panels for plotting.')

    for d in display:
        _plot_single(d, tag, sec_marks)

    overview = [d for d in display if d['suffix'] == '']
    _plot_overview(overview[:N_DISPLAY], tag)


def _plot_single(d, tag, sec_marks):
    suffix = d.get('suffix', '')
    fb_note = ' [FALLBACK]' if d.get('fb_triggered') else ''
    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        f'{tag} — {d["name"][:28]}{fb_note}\n'
        f'FDE: {d["base_fde"]:.3f} → {d["lora_fde"]:.3f}m  '
        f'({d["gain"]:+.1f}%)  |  Dir: {d["base_dir"]:.1f}° → {d["lora_dir"]:.1f}°',
        fontsize=11, fontweight='bold')

    ax3 = fig.add_subplot(1, 3, 1, projection='3d')
    _draw_3d(ax3, d, sec_marks)
    ax_xy = fig.add_subplot(1, 3, 2)
    _draw_xy(ax_xy, d, sec_marks)
    ax_err = fig.add_subplot(1, 3, 3)
    ax_err.axis('off')
    _draw_error_panel(ax_err, d, sec_marks)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    tag_lower = tag.lower()
    fname = f'ablation_{tag_lower}_{d["name"][:30].replace("/","_").replace(" ","_")}{suffix}.png'
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {fname}')


def _plot_overview(display, tag):
    if not display: return
    n = len(display)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.5 * cols, 5 * rows))
    fig.suptitle(f'{tag} — Overview (Blue=History  Red=Base  Orange=LoRA  Green=Truth)',
                 fontsize=12, fontweight='bold')
    for i, d in enumerate(display):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        _draw_3d(ax, d, {})  # no markers on overview
        ax.set_title(f'#{d["rank"]} {d["name"][:20]}\nFDE: {d["base_fde"]:.2f}→{d["lora_fde"]:.2f}m ({d["gain"]:+.0f}%)',
                     fontsize=7.5)
    plt.tight_layout(pad=1.5)
    tag_lower = tag.lower()
    fig.savefig(OUT_DIR / f'ablation_{tag_lower}_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: ablation_{tag_lower}_overview.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_3d(ax, d, sec_marks):
    hp, fp, ap, tp = d['hp'], d['fp'], d['ap'], d['tp']
    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=CH['hist'], lw=2.5, label='History')
    ax.plot(fp[:, 0], fp[:, 1], fp[:, 2], color=CH['base'], lw=1.6, ls='--', alpha=0.65, label='Base')
    ax.plot(ap[:, 0], ap[:, 1], ap[:, 2], color=CH['lora'], lw=2.2, label='LoRA')
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=CH['truth'], lw=2.5, label='Truth')
    ax.scatter(tp[0, 0], tp[0, 1], tp[0, 2], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], tp[-1, 2], c='black', s=70, marker='*', zorder=10)

    for sec, step in sec_marks.items():
        if step < len(fp):
            ax.scatter(fp[step, 0], fp[step, 1], fp[step, 2],
                      c=CH['base'], s=25, marker='x', alpha=0.7)
            ax.scatter(ap[step, 0], ap[step, 1], ap[step, 2],
                      c=CH['lora'], s=30, marker='d', alpha=0.8)
            ax.scatter(tp[step, 0], tp[step, 1], tp[step, 2],
                      c=CH['truth'], s=20, marker='o', alpha=0.6)

    all_pts = np.concatenate([hp, fp, ap, tp], axis=0)
    rng = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.55
    zm = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    ax.set_xlim(all_pts[:, 0].mean() - rng, all_pts[:, 0].mean() + rng)
    ax.set_ylim(all_pts[:, 1].mean() - rng, all_pts[:, 1].mean() + rng)
    ax.set_zlim(zm - rng, zm + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.view_init(elev=22, azim=-55)


def _draw_xy(ax, d, sec_marks):
    hp, fp, ap, tp = d['hp'], d['fp'], d['ap'], d['tp']
    ax.plot(hp[:, 0], hp[:, 1], color=CH['hist'], lw=2.5, label='History')
    ax.plot(fp[:, 0], fp[:, 1], color=CH['base'], lw=1.6, ls='--', alpha=0.65, label='Base')
    ax.plot(ap[:, 0], ap[:, 1], color=CH['lora'], lw=2.2, label='LoRA')
    ax.plot(tp[:, 0], tp[:, 1], color=CH['truth'], lw=2.5, label='Truth')
    ax.scatter(hp[-1, 0], hp[-1, 1], c=CH['hist'], s=80, marker='s',
              edgecolors='black', lw=0.8, zorder=5)
    ax.scatter(tp[0, 0], tp[0, 1], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], c='black', s=70, marker='*', zorder=10)

    for sec, step in sec_marks.items():
        if step < len(fp):
            ax.scatter(fp[step, 0], fp[step, 1], c=CH['base'], s=25, marker='x', alpha=0.7)
            ax.scatter(ap[step, 0], ap[step, 1], c=CH['lora'], s=30, marker='d', alpha=0.8)
            ax.scatter(tp[step, 0], tp[step, 1], c=CH['truth'], s=20, marker='o', alpha=0.6)
            ax.annotate(f'{sec}s', (tp[step, 0], tp[step, 1]),
                       textcoords="offset points", xytext=(5, 5), fontsize=6, color=CH['truth'])

    all_xy = np.concatenate([hp[:, :2], fp[:, :2], ap[:, :2], tp[:, :2]], axis=0)
    rng = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55
    xm, ym = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(xm - rng, xm + rng); ax.set_ylim(ym - rng, ym + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper left')


def _draw_error_panel(ax, d, sec_marks):
    """Line chart (top) + table (bottom) of per-second errors."""
    fr = d.get('per_sec_fr', {})
    ad = d.get('per_sec_ad', {})
    if not fr or not ad:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
        return

    seconds = sorted(fr.keys())
    fr_vals = [fr[s] for s in seconds]
    ad_vals = [ad[s] for s in seconds]

    # ── Line chart (top) ──
    ax_line = ax.inset_axes([0.08, 0.48, 0.88, 0.48])
    ax_line.plot(seconds, fr_vals, 'o-', color=CH['base'], lw=2, ms=6, label='Base')
    ax_line.plot(seconds, ad_vals, 's-', color=CH['lora'], lw=2, ms=6, label='LoRA')
    ax_line.set_xlabel('Time (s)', fontsize=8)
    ax_line.set_ylabel('Error (m)', fontsize=8)
    ax_line.set_title('Per-Second Position Error', fontsize=9, fontweight='bold')
    ax_line.legend(fontsize=7.5)
    ax_line.grid(True, alpha=0.3)
    ax_line.set_xticks(seconds)
    ax_line.set_xticklabels([f'{s}s' if s > 0 else 'Start' for s in seconds], fontsize=7)
    # Annotate points
    for s, v in zip(seconds, fr_vals):
        ax_line.annotate(f'{v:.3f}', (s, v), textcoords="offset points", xytext=(0, -12),
                        fontsize=6, color=CH['base'], ha='center')
    for s, v in zip(seconds, ad_vals):
        ax_line.annotate(f'{v:.3f}', (s, v), textcoords="offset points", xytext=(0, 8),
                        fontsize=6, color=CH['lora'], ha='center')

    # ── Table (bottom) ──
    table_data = [['Time', 'Base (m)', 'LoRA (m)', u'Δ (m)']]
    for s in seconds:
        label = f'{s}s' if s > 0 else 'Start'
        table_data.append([label, f'{fr[s]:.3f}', f'{ad[s]:.3f}', f'{ad[s]-fr[s]:+.3f}'])

    tbl = ax.table(cellText=table_data, cellLoc='center', loc='lower center',
                   bbox=[0.05, 0.02, 0.9, 0.42])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0: cell.set_facecolor('#E0E0E0'); cell.set_fontsize(8)
        cell.set_linewidth(0.5)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    low = run_low()
    high = run_high()

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    if low:
        print(f'  LOW:  FDE {low["fde_base"]:.4f} → {low["fde_lora"]:.4f}m  '
              f'({low["gain_pct"]:+.2f}%)  Cata: {low["cata_base"]}→{low["cata_lora"]}  '
              f'{low["n_better"]}/{low["n_drones"]} drones better')
    if high:
        print(f'  HIGH: FDE {high["fde_base"]:.4f} → {high["fde_lora"]:.4f}m  '
              f'({high["gain_pct"]:+.2f}%)  {high["n_better"]}/{high["n_drones"]} drones better')
    print(f'  All outputs: pic-results/summarize/')
    print('=' * 70)
