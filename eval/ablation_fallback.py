#!/usr/bin/env python3
"""
Ablation with direction fallback: validate the recommended deployment config.

Compares 3 configurations on LOW (40-frame, UAV-Flow):
  1) Frozen base (single-head)
  2) + Online LoRA (improved: accum=5, lr=5e-5)
  3) + Online LoRA + Direction Fallback (90° threshold)

All use NO global LoRA. Causal protocol: warmup 60%, eval on held-out 40%.
Reports FDE, ADE, Dir, catastrophic rate, and per-drone breakdown.

Generates trajectory plots: 3D + XY top-down + per-second error table.
Lines: History (blue), Frozen (red), +LoRA (orange), +LoRA+FB (purple), Truth (green)

Output: pic-results/ablation_fallback.json, ablation_fallback_*.png
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
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_FRAC = 0.6
ACCUM = 5          # improved config
LR = 5e-5          # improved config
N_DRONES = 30
N_DISPLAY = 6
HIST_LEN = OC.HIST_LEN
PRED_LEN = OC.PRED_LEN
DT = 0.2
DIR_THRESHOLD_DEG = 90.0

# Plot colours — 5 lines
C = {
    'hist':     '#1565C0',   # blue
    'frozen':   '#D32F2F',   # red
    'lora':     '#FF6D00',   # orange
    'lora_fb':  '#7B1FA2',   # purple
    'truth':    '#2E7D32',   # green
}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 7,
                     'font.family': 'sans-serif', 'figure.dpi': 150})


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def make_adaptive_windows(traj, hist_len=40):
    n = traj.shape[0]
    stride = max(1, min(4, n // 60))
    ml = hist_len * stride + PRED_LEN
    if n < ml:
        return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + hist_len * stride, stride)[:hist_len]
        if len(indices) < hist_len:
            continue
        hists.append(traj[indices].copy())
        fut_start = i + hist_len * stride
        fut_end = fut_start + PRED_LEN
        if fut_end > n:
            continue
        fut_abs = traj[fut_start:fut_end, :3]
        hists[-1][:, :3] -= hists[-1][-1, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs


def const_vel_prediction(hist, pred_len=20, dt=0.2):
    """Constant-velocity displacement prediction (meters)."""
    last_vel = hist[:, -1, 3:6]
    vel_recent = hist[:, -3:, 3:6]
    w = torch.tensor([0.2, 0.3, 0.5], device=hist.device)
    last_vel_smooth = (vel_recent * w.view(1, 3, 1)).sum(dim=1)
    steps = torch.arange(1, pred_len + 1, device=hist.device).float()
    return last_vel_smooth.unsqueeze(1) * (steps.view(1, -1, 1) * dt)


def apply_direction_fallback(predictions, hist, threshold_deg=90.0):
    """Replace predictions with const-vel when direction deviates >threshold."""
    cv_pred = const_vel_prediction(hist, pred_len=predictions.shape[1], dt=DT)
    pred_dir = predictions[:, -1, :2]
    cv_dir = cv_pred[:, -1, :2]
    cos_sim = F.cosine_similarity(pred_dir + 1e-8, cv_dir + 1e-8, dim=-1)
    angle = torch.acos(cos_sim.clamp(-0.999, 0.999)) * (180.0 / math.pi)
    mask = (angle > threshold_deg).float().unsqueeze(-1).unsqueeze(-1)
    safe_pred = predictions * (1 - mask) + cv_pred * mask
    return safe_pred, mask.bool(), angle


def compute_errors(model, H_tensor, T_tensor, batch=64):
    """Forward single-head model, return (fde, ade, dirs, predictions)."""
    preds = []
    for b in range(0, len(H_tensor), batch):
        be = min(b + batch, len(H_tensor))
        with torch.no_grad():
            preds.append(model(H_tensor[b:be].to(DEVICE), force_predict=True)['predictions'].cpu())
    P = torch.cat(preds, dim=0)
    fde = torch.norm(P[:, -1, :] - T_tensor[:, -1, :], dim=-1)
    ade = torch.norm(P - T_tensor, dim=-1).mean(dim=1)
    dirs = np.array([dir_err(P[i, -1, :2].numpy(), T_tensor[i, -1, :2].numpy())
                     for i in range(len(P))])
    return fde, ade, dirs, P


def compute_errors_fallback(model, H_tensor, T_tensor, batch=64):
    """Forward model + apply direction fallback. Returns extended metrics."""
    preds_raw, preds_safe = [], []
    fell_back_count = 0
    angles_all = []

    for b in range(0, len(H_tensor), batch):
        be = min(b + batch, len(H_tensor))
        hb = H_tensor[b:be].to(DEVICE)
        with torch.no_grad():
            raw = model(hb, force_predict=True)['predictions']
        safe, fb_mask, angles = apply_direction_fallback(raw, hb, DIR_THRESHOLD_DEG)
        preds_raw.append(raw.cpu())
        preds_safe.append(safe.cpu())
        fell_back_count += int(fb_mask.any(dim=-1).any(dim=-1).sum().item())
        angles_all.append(angles.cpu())

    P_raw = torch.cat(preds_raw, dim=0)
    P_safe = torch.cat(preds_safe, dim=0)
    angles_all = torch.cat(angles_all, dim=0)

    fde = torch.norm(P_safe[:, -1, :] - T_tensor[:, -1, :], dim=-1)
    ade = torch.norm(P_safe - T_tensor, dim=-1).mean(dim=1)
    dirs = np.array([dir_err(P_safe[i, -1, :2].numpy(), T_tensor[i, -1, :2].numpy())
                     for i in range(len(P_safe))])
    n_cata = int((dirs >= 90).sum())

    return fde, ade, dirs, P_safe, P_raw, fell_back_count, n_cata, angles_all


def per_second_errors(P, T, dt=0.2):
    """Per-second position error at t=0,1,2,3,4s."""
    steps = {0: 0, 1: 5, 2: 10, 3: 15, 4: 19}
    errs = {}
    for sec, idx in steps.items():
        if idx < P.shape[1]:
            errs[sec] = float(torch.norm(P[idx] - T[idx], dim=-1).item())
    return errs


def _compute_aggregate(fde_tensor, ade_tensor, dir_array):
    return {
        'fde_mean': float(fde_tensor.mean()),
        'fde_median': float(fde_tensor.median()),
        'ade_mean': float(ade_tensor.mean()),
        'dir_mean': float(dir_array.mean()),
        'dir_median': float(np.median(dir_array)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('ABLATION — Single-head + Online LoRA + Direction Fallback')
    print(f'  Config: accum={ACCUM}, lr={LR}, direction_threshold={DIR_THRESHOLD_DEG}°')
    print('  No global LoRA. Causal protocol: warmup 60%, eval on held-out 40%.')
    print('=' * 80)

    TRAJ_DIR = ROOT / 'UAV-Flow-trajs'
    CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / '_ablation_fb_ck'
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

    MIN_LEN = 200

    # Collect trajectories
    trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        t = d['traj']
        if t.shape[0] >= MIN_LEN:
            trajs.append((f.stem, t))
    np.random.seed(11)
    np.random.shuffle(trajs)
    trajs = trajs[:N_DRONES]
    print(f'  {len(trajs)} drones (>= {MIN_LEN} frames)\n')

    # Build base (NO global LoRA)
    base = OC.build_online_base(device=DEVICE, with_global=False)
    mgr = DroneAdapterManager(
        OC.build_online_base(device=DEVICE, with_global=False),
        checkpoint_dir=str(CKPT_DIR))
    ocfg = OnlineLearnerConfig(
        accumulation_steps=ACCUM, lr=LR, device=str(DEVICE), conf_threshold=0.0)
    learner = OnlineLearner(mgr, ocfg)

    # ── Per-drone evaluation ──
    results = []
    all_display_data = []   # for visualization

    for i, (drone_id, traj) in enumerate(trajs):
        hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
        if len(hists) < 30:
            continue
        H = torch.from_numpy(np.array(hists, dtype=np.float32))
        T = torch.from_numpy(np.array(futs, dtype=np.float32))
        n = len(hists)
        n_warm = int(n * WARMUP_FRAC)
        if n_warm < ACCUM * 2 or (n - n_warm) < 5:
            continue

        Hw, Tw = H[:n_warm], T[:n_warm]
        Hh, Th = H[n_warm:], T[n_warm:]

        # ── Config 1: FROZEN base ──
        fde_fr, ade_fr, dir_fr, P_fr = compute_errors(base, Hh, Th)

        # ── Online learning (shared between Config 2 & 3) ──
        learner.reset_drone(drone_id)
        n_updates = 0
        for j in range(n_warm):
            updated = learner.observe(drone_id, Hw[j], Tw[j], confidence=1.0, timestep=j)
            if updated:
                n_updates += 1

        # Get online model
        if mgr.adapter is not None and mgr.active_drone == drone_id:
            online_model = mgr.adapter.model
        else:
            mgr.activate(drone_id)
            online_model = mgr.adapter.model

        # ── Config 2: +Online LoRA (no fallback) ──
        fde_on, ade_on, dir_on, P_on = compute_errors(online_model, Hh, Th)

        # ── Config 3: +Online LoRA + Direction Fallback ──
        (fde_fb, ade_fb, dir_fb, P_fb, P_fb_raw,
         n_fell_back, n_cata_fb, angles_fb) = compute_errors_fallback(online_model, Hh, Th)

        mgr.deactivate()

        # ── Aggregate metrics per config ──
        n_cata_fr = int((dir_fr >= 90).sum())
        n_cata_on = int((dir_on >= 90).sum())

        m_fr = _compute_aggregate(fde_fr, ade_fr, dir_fr)
        m_on = _compute_aggregate(fde_on, ade_on, dir_on)
        m_fb = _compute_aggregate(fde_fb, ade_fb, dir_fb)

        gain_on = (m_fr['fde_mean'] - m_on['fde_mean']) / max(m_fr['fde_mean'], 1e-6) * 100
        gain_fb = (m_fr['fde_mean'] - m_fb['fde_mean']) / max(m_fr['fde_mean'], 1e-6) * 100
        gain_on_to_fb = (m_on['fde_mean'] - m_fb['fde_mean']) / max(m_on['fde_mean'], 1e-6) * 100

        results.append(dict(
            drone=drone_id,
            n_windows=n, n_warm=n_warm, n_heldout=int(n - n_warm),
            n_updates=n_updates,
            frozen=m_fr, online=m_on, online_fallback=m_fb,
            frozen_cata=n_cata_fr, online_cata=n_cata_on, fallback_cata=n_cata_fb,
            n_fell_back=n_fell_back,
            gain_online_vs_frozen=float(gain_on),
            gain_fallback_vs_frozen=float(gain_fb),
            gain_fallback_vs_online=float(gain_on_to_fb),
        ))

        tag_on = 'GAIN' if gain_on > 0 else 'LOSS'
        tag_fb = 'GAIN' if gain_fb > 0 else 'LOSS'
        print(f'  [{i+1:2d}/{len(trajs)}] {drone_id[:28]:28s} '
              f'FROZEN={m_fr["fde_mean"]:.3f}m  '
              f'+LoRA={m_on["fde_mean"]:.3f}m ({gain_on:+.1f}%) [{tag_on}]  '
              f'+LoRA+FB={m_fb["fde_mean"]:.3f}m ({gain_fb:+.1f}%) [{tag_fb}]  '
              f'cata: {n_cata_fr}→{n_cata_on}→{n_cata_fb}  '
              f'FB_trig={n_fell_back}')

        # Store window-level data for display selection
        all_display_data.append(dict(
            drone=drone_id, traj=traj,
            n_warm=n_warm,
            fde_fr_per=fde_fr.numpy(), fde_on_per=fde_on.numpy(), fde_fb_per=fde_fb.numpy(),
            dir_fr=dir_fr, dir_on=dir_on, dir_fb=dir_fb,
            fde_fr_mean=m_fr['fde_mean'], fde_on_mean=m_on['fde_mean'],
            fde_fb_mean=m_fb['fde_mean'],
            gain_on=gain_on, gain_fb=gain_fb,
            n_fell_back=n_fell_back,
        ))

    if not results:
        print('No valid drones.')
        return

    # ── Summary ──
    fr_fde = np.array([r['frozen']['fde_mean'] for r in results])
    on_fde = np.array([r['online']['fde_mean'] for r in results])
    fb_fde = np.array([r['online_fallback']['fde_mean'] for r in results])
    fr_dir = np.array([r['frozen']['dir_mean'] for r in results])
    on_dir = np.array([r['online']['dir_mean'] for r in results])
    fb_dir = np.array([r['online_fallback']['dir_mean'] for r in results])

    total_cata_fr = sum(r['frozen_cata'] for r in results)
    total_cata_on = sum(r['online_cata'] for r in results)
    total_cata_fb = sum(r['fallback_cata'] for r in results)
    total_fb = sum(r['n_fell_back'] for r in results)
    total_win = sum(r['n_heldout'] for r in results)

    n_better_on = int((on_fde < fr_fde).sum())
    n_better_fb = int((fb_fde < fr_fde).sum())

    print(f'\n{"=" * 90}')
    print(f'FALLBACK ABLATION RESULTS ({len(results)} drones, {total_win} held-out windows)')
    print(f'{"=" * 90}')
    print(f'  {"Config":<40} {"FDE mean":>10} {"FDE med":>10} {"Dir mean":>10} {"Cata":>8}')
    print(f'  {"-"*40} {"-"*10} {"-"*10} {"-"*10} {"-"*8}')
    print(f'  {"(1) Frozen base":<40} {fr_fde.mean():>10.4f} {np.median(fr_fde):>10.4f} '
          f'{fr_dir.mean():>10.2f} {total_cata_fr:>4}/{total_win}')
    print(f'  {"(2) +Online LoRA":<40} {on_fde.mean():>10.4f} {np.median(on_fde):>10.4f} '
          f'{on_dir.mean():>10.2f} {total_cata_on:>4}/{total_win}')
    print(f'  {"(3) +Online LoRA +Fallback":<40} {fb_fde.mean():>10.4f} {np.median(fb_fde):>10.4f} '
          f'{fb_dir.mean():>10.2f} {total_cata_fb:>4}/{total_win}')
    print(f'  {"="*90}')
    print(f'  Online vs Frozen gain:     {(fr_fde.mean()-on_fde.mean())/fr_fde.mean()*100:+.2f}% '
          f'({n_better_on}/{len(results)} drones better)')
    print(f'  Online+FB vs Frozen gain:   {(fr_fde.mean()-fb_fde.mean())/fr_fde.mean()*100:+.2f}% '
          f'({n_better_fb}/{len(results)} drones better)')
    print(f'  Online+FB vs Online delta:  {(on_fde.mean()-fb_fde.mean())/max(on_fde.mean(),1e-6)*100:+.2f}%')
    print(f'  Direction fallbacks triggered: {total_fb} / {total_win} windows ({total_fb/total_win*100:.1f}%)')
    print(f'{"=" * 90}')

    # Save JSON
    summary = {
        'model': 'LOW (40-frame, 5Hz) — Single-head + Online LoRA + Direction Fallback',
        'config': {'accum': ACCUM, 'lr': LR, 'direction_threshold_deg': DIR_THRESHOLD_DEG},
        'n_drones': len(results), 'n_windows': total_win,
        'n_better_online': n_better_on, 'n_better_fallback': n_better_fb,
        'total_fallbacks': total_fb,
        'frozen': {
            'fde_mean': float(fr_fde.mean()), 'fde_median': float(np.median(fr_fde)),
            'dir_mean': float(fr_dir.mean()), 'cata': total_cata_fr,
            'cata_rate': total_cata_fr / total_win * 100,
        },
        'online': {
            'fde_mean': float(on_fde.mean()), 'fde_median': float(np.median(on_fde)),
            'dir_mean': float(on_dir.mean()), 'cata': total_cata_on,
            'cata_rate': total_cata_on / total_win * 100,
        },
        'online_fallback': {
            'fde_mean': float(fb_fde.mean()), 'fde_median': float(np.median(fb_fde)),
            'dir_mean': float(fb_dir.mean()), 'cata': total_cata_fb,
            'cata_rate': total_cata_fb / total_win * 100,
        },
        'per_drone': sorted(results, key=lambda r: r['gain_fallback_vs_frozen'], reverse=True),
    }
    json.dump(summary, open(OUT_DIR / 'ablation_fallback.json', 'w'), indent=2)
    print(f'\n  Saved: pic-results/ablation_fallback.json')

    # ── Visualization ──
    _generate_plots(results, all_display_data, base, mgr, learner, CKPT_DIR)

    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_plots(results, all_display_data, base, mgr, learner, ckpt_dir):
    """Select representative trajectories and generate per-trajectory + overview plots."""

    # ── Select representative drones ──
    sorted_r = sorted(results, key=lambda r: r['gain_fallback_vs_frozen'], reverse=True)
    n = len(sorted_r)

    picks = []
    if n >= 1: picks.append(0)           # best overall
    if n >= 3: picks.append(n // 3)      # upper-mid
    if n >= 5: picks.append(n // 2)      # median
    if n >= 7: picks.append(2 * n // 3)  # lower-mid
    if n >= 4: picks.append(n - 1)       # worst

    # Also pick highest-fallback drone (most cata rescues)
    by_fb = sorted(results, key=lambda r: r['n_fell_back'], reverse=True)
    worst_fb = by_fb[0]
    if worst_fb not in [sorted_r[p] for p in picks]:
        picks.append(next(i for i, r in enumerate(sorted_r) if r['drone'] == worst_fb['drone']))

    picks = sorted(set(picks))
    print(f'\n  Representative picks (by gain rank): {[f"#{p+1}" for p in picks]}')

    traj_map = {d['drone']: d['traj'] for d in all_display_data}
    data_map = {d['drone']: d for d in all_display_data}

    display = []
    for p in picks:
        r = sorted_r[p]
        name = r['drone']
        if name not in traj_map:
            continue
        traj = traj_map[name]
        dd = data_map[name]

        H_list, T_list = make_adaptive_windows(traj, hist_len=HIST_LEN)
        if len(H_list) < 20:
            continue
        H = torch.from_numpy(np.array(H_list, dtype=np.float32))
        T = torch.from_numpy(np.array(T_list, dtype=np.float32))
        n_tot = len(H_list)
        n_warm = dd['n_warm']
        Hh, Th = H[n_warm:], T[n_warm:]

        # Re-run online learning fresh
        learner.reset_drone(name)
        for j in range(n_warm):
            learner.observe(name, H[j], T[j], confidence=1.0, timestep=j)

        if mgr.adapter is not None and mgr.active_drone == name:
            on_model = mgr.adapter.model
        else:
            mgr.activate(name)
            on_model = mgr.adapter.model

        # Get all 3 prediction sets
        _, _, _, P_fr = compute_errors(base, Hh, Th)
        _, _, _, P_on = compute_errors(on_model, Hh, Th)
        _, _, _, P_fb, P_fb_raw, fb_count, _, angles = compute_errors_fallback(on_model, Hh, Th)

        mgr.deactivate()

        # Pick the best single window: where LoRA+FB improves most over frozen
        fde_fr_per = torch.norm(P_fr[:, -1, :] - Th[:, -1, :], dim=-1)
        fde_fb_per = torch.norm(P_fb[:, -1, :] - Th[:, -1, :], dim=-1)
        improvements = fde_fr_per - fde_fb_per
        best_idx = int(torch.argmax(improvements))

        # Also find a window where fallback actually triggered (if any)
        fb_idx = None
        if fb_count > 0:
            diff_fb_raw = torch.norm(P_fb_raw[:, -1, :] - Th[:, -1, :], dim=-1)
            diff_fb_safe = torch.norm(P_fb[:, -1, :] - Th[:, -1, :], dim=-1)
            fb_candidates = (diff_fb_raw - diff_fb_safe).abs() > 0.001
            fb_indices = torch.where(fb_candidates)[0]
            if len(fb_indices) > 0:
                fb_idx = int(fb_indices[0])

        for idx, suffix in [(best_idx, ''), (fb_idx, '_fallback')]:
            if idx is None:
                continue

            # Per-second errors
            per_sec_fr = per_second_errors(P_fr[idx], Th[idx])
            per_sec_on = per_second_errors(P_on[idx], Th[idx])
            per_sec_fb = per_second_errors(P_fb[idx], Th[idx])

            # Absolute coordinates
            hist_last = H_list[n_warm + idx][-1, :3]
            hp_abs = H_list[n_warm + idx][:, :3]
            fp_abs = P_fr[idx].numpy() + hist_last
            op_abs = P_on[idx].numpy() + hist_last
            fbp_abs = P_fb[idx].numpy() + hist_last
            tp_abs = T_list[n_warm + idx] + hist_last

            # Direction error at final step
            dir_fr_val = dir_err(P_fr[idx, -1, :2].numpy(), Th[idx, -1, :2].numpy())
            dir_on_val = dir_err(P_on[idx, -1, :2].numpy(), Th[idx, -1, :2].numpy())
            dir_fb_val = dir_err(P_fb[idx, -1, :2].numpy(), Th[idx, -1, :2].numpy())

            # Did fallback trigger on this window?
            if idx < len(angles):
                angle_val = float(angles[idx])
                fb_triggered = angle_val > DIR_THRESHOLD_DEG
            else:
                fb_triggered = False

            display.append(dict(
                name=name, rank=p + 1,
                gain_on=float(dd['gain_on']),
                gain_fb=float(dd['gain_fb']),
                frozen_fde=float(fde_fr_per[idx]),
                online_fde=float(torch.norm(P_on[idx, -1, :] - Th[idx, -1, :]).item()),
                fallback_fde=float(fde_fb_per[idx]),
                frozen_dir=dir_fr_val,
                online_dir=dir_on_val,
                fallback_dir=dir_fb_val,
                fb_triggered=fb_triggered,
                hp=hp_abs, fp=fp_abs, op=op_abs, fbp=fbp_abs, tp=tp_abs,
                per_sec_fr=per_sec_fr, per_sec_on=per_sec_on, per_sec_fb=per_sec_fb,
                suffix=suffix,
            ))

        learner.reset_drone(name)

    print(f'  {len(display)} trajectories selected for plotting.')

    # ── Per-trajectory plots ──
    for d in display:
        _plot_single(d)

    # ── Overview grid ──
    # Only plot the "best" windows (suffix=''), not the fallback examples
    overview = [d for d in display if d['suffix'] == '']
    _plot_overview(overview[:N_DISPLAY])


def _plot_single(d):
    """Plot one trajectory: 3D + XY + per-second error table."""
    suffix = d.get('suffix', '')
    fig = plt.figure(figsize=(18, 7))
    fb_note = ' [FALLBACK TRIGGERED]' if d.get('fb_triggered') else ''
    fig.suptitle(
        f'Fallback Ablation — {d["name"][:28]}{fb_note}\n'
        f'FDE: Frozen={d["frozen_fde"]:.3f}  +LoRA={d["online_fde"]:.3f}  '
        f'+LoRA+FB={d["fallback_fde"]:.3f} m  '
        f'(LoRA {d["gain_on"]:+.1f}%, +FB {d["gain_fb"]:+.1f}%)',
        fontsize=11, fontweight='bold')

    # (1) 3D
    ax3 = fig.add_subplot(1, 3, 1, projection='3d')
    _draw_3d(ax3, d)

    # (2) XY top-down
    ax_xy = fig.add_subplot(1, 3, 2)
    _draw_xy(ax_xy, d)

    # (3) Per-second error table + bars
    ax_tbl = fig.add_subplot(1, 3, 3)
    ax_tbl.axis('off')
    _draw_per_second_table(ax_tbl, d)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f'ablation_fallback_{d["name"][:30].replace("/","_").replace(" ","_")}{suffix}.png'
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {fname}')


def _plot_overview(display):
    """Compact 3D overview grid."""
    if not display:
        return
    n = len(display)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5.2 * rows))
    fig.suptitle(
        'Fallback Ablation — Overview\n'
        '(Blue=History  Red=Frozen  Orange=+LoRA  Purple=+LoRA+FB  Green=Truth)',
        fontsize=12, fontweight='bold')
    for i, d in enumerate(display):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        _draw_3d(ax, d)
        ax.set_title(
            f'#{d["rank"]} {d["name"][:20]}\n'
            f'FDE: {d["frozen_fde"]:.2f}→{d["online_fde"]:.2f}→{d["fallback_fde"]:.2f}m\n'
            f'LoRA {d["gain_on"]:+.0f}%  +FB {d["gain_fb"]:+.0f}%',
            fontsize=7.5)
    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / 'ablation_fallback_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: ablation_fallback_overview.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_3d(ax, d):
    hp, fp, op, fbp, tp = d['hp'], d['fp'], d['op'], d['fbp'], d['tp']
    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=C['hist'], lw=2.5, label='History')
    ax.plot(fp[:, 0], fp[:, 1], fp[:, 2], color=C['frozen'], lw=1.6, ls='--', alpha=0.65,
            label='Frozen')
    ax.plot(op[:, 0], op[:, 1], op[:, 2], color=C['lora'], lw=2.0, label='+LoRA')
    ax.plot(fbp[:, 0], fbp[:, 1], fbp[:, 2], color=C['lora_fb'], lw=2.2, label='+LoRA+FB')
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=C['truth'], lw=2.5, label='Truth')

    # Start/end markers
    ax.scatter(tp[0, 0], tp[0, 1], tp[0, 2], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], tp[-1, 2], c='black', s=70, marker='*', zorder=10)

    # Per-second markers
    for sec, step in [(1, 5), (2, 10), (3, 15), (4, 19)]:
        if step < len(fp):
            ax.scatter(fp[step, 0], fp[step, 1], fp[step, 2],
                      c=C['frozen'], s=25, marker='x', alpha=0.7)
            ax.scatter(op[step, 0], op[step, 1], op[step, 2],
                      c=C['lora'], s=25, marker='+', alpha=0.8)
            ax.scatter(fbp[step, 0], fbp[step, 1], fbp[step, 2],
                      c=C['lora_fb'], s=30, marker='d', alpha=0.8)
            ax.scatter(tp[step, 0], tp[step, 1], tp[step, 2],
                      c=C['truth'], s=20, marker='o', alpha=0.6)

    all_pts = np.concatenate([hp, fp, op, fbp, tp], axis=0)
    rng = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.55
    zm = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    ax.set_xlim(all_pts[:, 0].mean() - rng, all_pts[:, 0].mean() + rng)
    ax.set_ylim(all_pts[:, 1].mean() - rng, all_pts[:, 1].mean() + rng)
    ax.set_zlim(zm - rng, zm + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(fontsize=6, loc='upper left')
    ax.view_init(elev=22, azim=-55)


def _draw_xy(ax, d):
    hp, fp, op, fbp, tp = d['hp'], d['fp'], d['op'], d['fbp'], d['tp']
    ax.plot(hp[:, 0], hp[:, 1], color=C['hist'], lw=2.5, label='History')
    ax.plot(fp[:, 0], fp[:, 1], color=C['frozen'], lw=1.6, ls='--', alpha=0.65,
            label='Frozen')
    ax.plot(op[:, 0], op[:, 1], color=C['lora'], lw=2.0, label='+LoRA')
    ax.plot(fbp[:, 0], fbp[:, 1], color=C['lora_fb'], lw=2.2, label='+LoRA+FB')
    ax.plot(tp[:, 0], tp[:, 1], color=C['truth'], lw=2.5, label='Truth')

    ax.scatter(hp[-1, 0], hp[-1, 1], c=C['hist'], s=80, marker='s',
              edgecolors='black', lw=0.8, zorder=5)
    ax.scatter(tp[0, 0], tp[0, 1], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], c='black', s=70, marker='*', zorder=10)

    for sec, step in [(1, 5), (2, 10), (3, 15), (4, 19)]:
        if step < len(fp):
            ax.scatter(fp[step, 0], fp[step, 1], c=C['frozen'], s=25, marker='x', alpha=0.7)
            ax.scatter(op[step, 0], op[step, 1], c=C['lora'], s=25, marker='+', alpha=0.8)
            ax.scatter(fbp[step, 0], fbp[step, 1], c=C['lora_fb'], s=30, marker='d', alpha=0.8)
            ax.scatter(tp[step, 0], tp[step, 1], c=C['truth'], s=20, marker='o', alpha=0.6)
            ax.annotate(f'{sec}s', (tp[step, 0], tp[step, 1]),
                       textcoords="offset points", xytext=(5, 5), fontsize=6, color=C['truth'])

    all_xy = np.concatenate([hp[:, :2], fp[:, :2], op[:, :2], fbp[:, :2], tp[:, :2]], axis=0)
    rng = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55
    xm, ym = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(xm - rng, xm + rng); ax.set_ylim(ym - rng, ym + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc='upper left')


def _draw_per_second_table(ax, d):
    """Per-second error table + bar chart for all 3 configs."""
    fr = d.get('per_sec_fr', {})
    on = d.get('per_sec_on', {})
    fb = d.get('per_sec_fb', {})
    if not fr or not on or not fb:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
        return

    seconds = sorted(fr.keys())
    fr_vals = [fr[s] for s in seconds]
    on_vals = [on[s] for s in seconds]
    fb_vals = [fb[s] for s in seconds]

    # Table
    table_data = [['Time', 'Frozen', '+LoRA', '+LoRA+FB', u'Δ(FB-Fr)']]
    for s in seconds:
        label = f'{s}s' if s > 0 else 'Start'
        table_data.append([
            label,
            f'{fr[s]:.3f}', f'{on[s]:.3f}', f'{fb[s]:.3f}',
            f'{fb[s]-fr[s]:+.3f}',
        ])

    tbl = ax.table(cellText=table_data, cellLoc='center', loc='upper center',
                   bbox=[0.03, 0.52, 0.94, 0.44])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_fontsize(7.5)
        cell.set_linewidth(0.5)

    # Bar chart
    axins = ax.inset_axes([0.10, 0.05, 0.85, 0.42])
    x = np.arange(len(seconds))
    w = 0.25
    axins.bar(x - w, fr_vals, w, color=C['frozen'], alpha=0.7, label='Frozen')
    axins.bar(x, on_vals, w, color=C['lora'], alpha=0.8, label='+LoRA')
    axins.bar(x + w, fb_vals, w, color=C['lora_fb'], alpha=0.8, label='+LoRA+FB')
    axins.set_xticks(x)
    axins.set_xticklabels([f'{s}s' if s > 0 else 'Start' for s in seconds], fontsize=7)
    axins.set_ylabel('Position Error (m)', fontsize=8)
    axins.set_title('Per-Second Position Error', fontsize=9, fontweight='bold')
    axins.legend(fontsize=6.5)
    axins.grid(True, alpha=0.3, axis='y')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
    print('\nDone.')
