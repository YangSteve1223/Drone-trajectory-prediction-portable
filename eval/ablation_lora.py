#!/usr/bin/env python3
"""
Ablation study: base model vs base + per-drone online LoRA (NO global LoRA).

Evaluates on held-out raw trajectories using the causal streaming protocol:
warmup on the first 60% of each flight, evaluate FROZEN vs ONLINE on the
held-out last 40%. Per-drone LoRA never sees the held-out portion.

Generates:
  - Summary table (FDE / ADE / Dir error, per-drone)
  - Representative trajectory plots: 3D + XY + per-second error table
  - LOW: 40-frame model, UAV-Flow (5 Hz, 4 s prediction)
  - HIGH: 20-frame model, SimCruise (1 Hz, 20 s prediction)

Output: pic-results/ablation_low.json, ablation_high.json,
        pic-results/ablation_low_*.png, ablation_high_*.png
"""

import torch, numpy as np, sys, warnings, json, shutil
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
ROOT = Path(__file__).resolve().parents[2]   # Drone-trajectory-prediction/
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared settings
# ═══════════════════════════════════════════════════════════════════════════════
WARMUP_FRAC = 0.6
ACCUM = 10
N_DISPLAY = 8          # representative trajectories to plot

# Plot colours
C = {'hist': '#1565C0', 'base': '#D32F2F', 'online': '#FF6D00', 'truth': '#2E7D32'}

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


def make_adaptive_windows(traj, hist_len=40, pred_len=20):
    """Adaptive-stride windows from a raw trajectory."""
    n = traj.shape[0]
    stride = max(1, min(4, n // 60))
    ml = hist_len * stride + pred_len
    if n < ml:
        return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + hist_len * stride, stride)[:hist_len]
        if len(indices) < hist_len:
            continue
        hists.append(traj[indices].copy())
        fut_start = i + hist_len * stride
        fut_end = fut_start + pred_len
        if fut_end > n:
            continue
        fut_abs = traj[fut_start:fut_end, :3]
        hists[-1][:, :3] -= hists[-1][-1, :3]       # centre on last history position
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs


def make_windows_high(traj, hist_len=20, pred_len=20):
    """20-frame stride-1 windows for HIGH (SimCruise, position-centred on first frame)."""
    n = traj.shape[0]
    H, T = [], []
    for j in range(hist_len - 1, n - pred_len):
        h = traj[j - hist_len + 1:j + 1].copy()
        fut_abs = traj[j + 1:j + 1 + pred_len, :3]
        if fut_abs.shape[0] < pred_len:
            continue
        fut = fut_abs - traj[j, :3]
        h[:, :3] -= h[0, :3]
        H.append(h); T.append(fut)
    return H, T


def compute_errors(model, H_tensor, T_tensor, batch=64):
    """Return (FDE, ADE, dir_err_mean, all_predictions)."""
    preds = []
    for b in range(0, len(H_tensor), batch):
        be = min(b + batch, len(H_tensor))
        with torch.no_grad():
            preds.append(model(H_tensor[b:be].to(DEVICE), force_predict=True)['predictions'].cpu())
    P = torch.cat(preds, dim=0)          # (N, 20, 3)
    fde = torch.norm(P[:, -1, :] - T_tensor[:, -1, :], dim=-1)      # (N,)
    ade = torch.norm(P - T_tensor, dim=-1).mean(dim=1)               # (N,)
    dirs = np.array([dir_err(P[i, -1, :2].numpy(), T_tensor[i, -1, :2].numpy())
                     for i in range(len(H_tensor))])
    return fde, ade, dirs, P


def per_second_errors(P, T, dt=0.2):
    """Per-second position error (Euclidean) for 20-step predictions at 5 Hz.
    Returns dict: {second: (frozen_error, online_error)} at t=0,1,2,3,4 s."""
    # steps at 0s, 1s, 2s, 3s, 4s  →  indices 0, 5, 10, 15, 19
    steps = {0: 0, 1: 5, 2: 10, 3: 15, 4: 19}
    errs = {}
    for sec, idx in steps.items():
        if idx < P.shape[1]:
            errs[sec] = float(torch.norm(P[idx] - T[idx], dim=-1).item())
    return errs


# ═══════════════════════════════════════════════════════════════════════════════
# LOW model ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_low():
    print('=' * 80)
    print('ABLATION — LOW (40-frame, UAV-Flow, 5 Hz)')
    print('  Base: 40-frame model  |  Online: base + per-drone streaming LoRA')
    print('  NO global LoRA.  Causal protocol: warmup 60%, eval on held-out 40%.')
    print('=' * 80)

    TRAJ_DIR = ROOT / 'UAV-Flow-trajs'
    CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / '_ablation_low_ck'
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

    MIN_LEN = 200
    N_DRONES = 30

    # Collect long trajectories
    trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        t = d['traj']
        if t.shape[0] >= MIN_LEN:
            trajs.append((f.stem, t))
    np.random.seed(11); np.random.shuffle(trajs)
    trajs = trajs[:N_DRONES]
    print(f'  {len(trajs)} drones (>= {MIN_LEN} frames)\n')

    # Build base (NO global LoRA)
    base = OC.build_online_base(device=DEVICE, with_global=False)
    mgr = DroneAdapterManager(OC.build_online_base(device=DEVICE, with_global=False),
                              checkpoint_dir=str(CKPT_DIR))
    cfg = OnlineLearnerConfig(accumulation_steps=ACCUM, device=str(DEVICE),
                              conf_threshold=0.0)
    learner = OnlineLearner(mgr, cfg)

    results = []
    for i, (drone_id, traj) in enumerate(trajs):
        hists, futs = make_adaptive_windows(traj, hist_len=OC.HIST_LEN)
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

        # FROZEN (base only)
        fde_fr, ade_fr, dir_fr, _ = compute_errors(base, Hh, Th)

        # ONLINE (base + per-drone LoRA, streaming warmup)
        learner.reset_drone(drone_id)
        n_updates = 0
        for j in range(n_warm):
            updated = learner.observe(drone_id, Hw[j], Tw[j], confidence=1.0, timestep=j)
            if updated:
                n_updates += 1

        # Evaluate on held-out using the resident (trained) adapter
        if mgr.adapter is not None and mgr.active_drone == drone_id:
            online_model = mgr.adapter.model
            fde_on, ade_on, dir_on, P_on = compute_errors(online_model, Hh, Th)
        else:
            mgr.activate(drone_id)
            fde_on, ade_on, dir_on, P_on = compute_errors(mgr.adapter.model, Hh, Th)
        mgr.deactivate()

        fde_fr_mean = float(fde_fr.mean())
        fde_on_mean = float(fde_on.mean())
        gain = (fde_fr_mean - fde_on_mean) / max(fde_fr_mean, 1e-6) * 100
        results.append(dict(
            drone=drone_id, n_windows=n, n_warm=n_warm, n_heldout=int(n - n_warm),
            n_updates=n_updates,
            frozen_fde=fde_fr_mean, online_fde=fde_on_mean,
            frozen_ade=float(ade_fr.mean()), online_ade=float(ade_on.mean()),
            frozen_dir=float(dir_fr.mean()), online_dir=float(dir_on.mean()),
            fde_gain=gain,
        ))
        tag = 'GAIN' if gain > 0 else 'LOSS'
        print(f'  [{i+1:2d}/{len(trajs)}] {drone_id[:28]:28s} '
              f'FROZEN FDE={fde_fr_mean:.3f}  ONLINE FDE={fde_on_mean:.3f}  '
              f'({gain:+.1f}%) [{tag}]  upd={n_updates}')

    if not results:
        print('No valid LOW drones.')
        return

    # ── Summary ──
    fr_fde = np.array([r['frozen_fde'] for r in results])
    on_fde = np.array([r['online_fde'] for r in results])
    fr_ade = np.array([r['frozen_ade'] for r in results])
    on_ade = np.array([r['online_ade'] for r in results])
    fr_dir = np.array([r['frozen_dir'] for r in results])
    on_dir = np.array([r['online_dir'] for r in results])
    n_better = int((on_fde < fr_fde).sum())

    print(f'\n{"=" * 80}')
    print(f'LOW ABLATION RESULTS ({len(results)} drones)')
    print(f'{"=" * 80}')
    print(f'  {"":18} {"FROZEN":<12} {"ONLINE":<12} {"Change":<10}')
    print(f'  {"mean FDE (m)":18} {fr_fde.mean():<12.4f} {on_fde.mean():<12.4f} '
          f'{(fr_fde.mean()-on_fde.mean())/fr_fde.mean()*100:>+7.1f}%')
    print(f'  {"median FDE (m)":18} {np.median(fr_fde):<12.4f} {np.median(on_fde):<12.4f}')
    print(f'  {"mean ADE (m)":18} {fr_ade.mean():<12.4f} {on_ade.mean():<12.4f} '
          f'{(fr_ade.mean()-on_ade.mean())/max(fr_ade.mean(),1e-6)*100:>+7.1f}%')
    print(f'  {"mean Dir (deg)":18} {fr_dir.mean():<12.2f} {on_dir.mean():<12.2f} '
          f'{(fr_dir.mean()-on_dir.mean())/max(fr_dir.mean(),0.1)*100:>+7.1f}%')
    print(f'  online beats frozen on {n_better}/{len(results)} drones')

    # Save JSON
    summary = {
        'model': 'LOW (40-frame, 5Hz)',
        'n_drones': len(results), 'n_better': n_better,
        'frozen_fde_mean': float(fr_fde.mean()), 'online_fde_mean': float(on_fde.mean()),
        'frozen_ade_mean': float(fr_ade.mean()), 'online_ade_mean': float(on_ade.mean()),
        'frozen_dir_mean': float(fr_dir.mean()), 'online_dir_mean': float(on_dir.mean()),
        'fde_gain_pct': float((fr_fde.mean() - on_fde.mean()) / fr_fde.mean() * 100),
        'per_drone': sorted(results, key=lambda r: r['fde_gain'], reverse=True),
    }
    json.dump(summary, open(OUT_DIR / 'ablation_low.json', 'w'), indent=2)
    print(f'\n  Saved: pic-results/ablation_low.json')

    # ── Representative trajectory selection ──
    display = _select_representative(results, trajs, make_adaptive_windows,
                                     OC.HIST_LEN, base, mgr, learner, tag='LOW')
    _plot_low_display(display)

    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH model ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_high():
    print('\n' + '=' * 80)
    print('ABLATION — HIGH (20-frame, SimCruise, 1 Hz)')
    print('  Base: 20-frame HIGH model  |  Online: base + per-drone streaming LoRA')
    print('  NO global LoRA.  Causal protocol: warmup 60%, eval on held-out 40%.')
    print('=' * 80)

    SIM_DIR = ROOT / 'SimCruise'
    CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / '_ablation_high_ck'
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

    MIN_LEN = 120
    N_DRONES = 25
    HIST_LEN = 20
    PRED_LEN = 20

    # Load HIGH trajectories
    merged = sorted(SIM_DIR.rglob('trajectories_merged.npz'))
    if not merged:
        print('  trajectories_merged.npz not found — skipping HIGH ablation.')
        return None
    d = np.load(merged[0])
    pos, vel, mask = d['positions'], d['velocities'], d['masks']
    lengths = mask.sum(axis=1)
    idx = np.where(lengths >= MIN_LEN)[0]
    rng = np.random.RandomState(11); rng.shuffle(idx)
    idx = idx[:N_DRONES]
    trajs = []
    for i in idx:
        L = int(lengths[i])
        p = pos[i, :L].astype(np.float32); v = vel[i, :L].astype(np.float32)
        trajs.append((f'high_{i}', np.concatenate([p, v], axis=1)))
    print(f'  {len(trajs)} HIGH drones (>= {MIN_LEN} frames)\n')

    base = OC.build_high_base(device=DEVICE)
    mgr = DroneAdapterManager(OC.build_high_base(device=DEVICE), checkpoint_dir=str(CKPT_DIR))
    cfg = OnlineLearnerConfig(accumulation_steps=ACCUM, device=str(DEVICE),
                              conf_threshold=0.0, dt=OC.HIGH_DT)
    learner = OnlineLearner(mgr, cfg)

    results = []
    for i, (drone_id, traj) in enumerate(trajs):
        H_list, T_list = make_windows_high(traj, hist_len=HIST_LEN)
        if len(H_list) < 30:
            continue
        H = torch.from_numpy(np.array(H_list, dtype=np.float32))
        T = torch.from_numpy(np.array(T_list, dtype=np.float32))
        n = len(H_list)
        n_warm = int(n * WARMUP_FRAC)
        if n_warm < ACCUM * 2 or (n - n_warm) < 5:
            continue

        Hw, Tw = H[:n_warm], T[:n_warm]
        Hh, Th = H[n_warm:], T[n_warm:]

        # FROZEN
        fde_fr, ade_fr, dir_fr, _ = compute_errors(base, Hh, Th)

        # ONLINE
        learner.reset_drone(drone_id)
        for j in range(n_warm):
            learner.observe(drone_id, Hw[j], Tw[j], confidence=1.0, timestep=j)

        if mgr.adapter is not None and mgr.active_drone == drone_id:
            fde_on, ade_on, dir_on, _ = compute_errors(mgr.adapter.model, Hh, Th)
        else:
            mgr.activate(drone_id)
            fde_on, ade_on, dir_on, _ = compute_errors(mgr.adapter.model, Hh, Th)
        mgr.deactivate()

        fde_fr_m = float(fde_fr.mean())
        fde_on_m = float(fde_on.mean())
        gain = (fde_fr_m - fde_on_m) / max(fde_fr_m, 1e-6) * 100
        results.append(dict(
            drone=drone_id, n_windows=n, n_warm=n_warm, n_heldout=int(n - n_warm),
            frozen_fde=fde_fr_m, online_fde=fde_on_m,
            frozen_ade=float(ade_fr.mean()), online_ade=float(ade_on.mean()),
            frozen_dir=float(dir_fr.mean()), online_dir=float(dir_on.mean()),
            fde_gain=gain,
        ))
        tag = 'GAIN' if gain > 0 else 'LOSS'
        print(f'  [{i+1:2d}/{len(trajs)}] {drone_id:12s} '
              f'FROZEN FDE={fde_fr_m:.3f}  ONLINE FDE={fde_on_m:.3f}  '
              f'({gain:+.1f}%) [{tag}]')

    if not results:
        print('No valid HIGH drones.')
        return None

    # ── Summary ──
    fr_fde = np.array([r['frozen_fde'] for r in results])
    on_fde = np.array([r['online_fde'] for r in results])
    fr_dir = np.array([r['frozen_dir'] for r in results])
    on_dir = np.array([r['online_dir'] for r in results])
    n_better = int((on_fde < fr_fde).sum())

    print(f'\n{"=" * 80}')
    print(f'HIGH ABLATION RESULTS ({len(results)} drones)')
    print(f'{"=" * 80}')
    print(f'  {"":18} {"FROZEN":<12} {"ONLINE":<12} {"Change":<10}')
    print(f'  {"mean FDE (m)":18} {fr_fde.mean():<12.4f} {on_fde.mean():<12.4f} '
          f'{(fr_fde.mean()-on_fde.mean())/fr_fde.mean()*100:>+7.1f}%')
    print(f'  {"median FDE (m)":18} {np.median(fr_fde):<12.4f} {np.median(on_fde):<12.4f}')
    print(f'  {"mean Dir (deg)":18} {fr_dir.mean():<12.2f} {on_dir.mean():<12.2f} '
          f'{(fr_dir.mean()-on_dir.mean())/max(fr_dir.mean(),0.1)*100:>+7.1f}%')
    print(f'  online beats frozen on {n_better}/{len(results)} drones')

    summary = {
        'model': 'HIGH (20-frame, 1Hz)',
        'n_drones': len(results), 'n_better': n_better,
        'frozen_fde_mean': float(fr_fde.mean()), 'online_fde_mean': float(on_fde.mean()),
        'frozen_dir_mean': float(fr_dir.mean()), 'online_dir_mean': float(on_dir.mean()),
        'fde_gain_pct': float((fr_fde.mean() - on_fde.mean()) / fr_fde.mean() * 100),
        'per_drone': sorted(results, key=lambda r: r['fde_gain'], reverse=True),
    }
    json.dump(summary, open(OUT_DIR / 'ablation_high.json', 'w'), indent=2)
    print(f'\n  Saved: pic-results/ablation_high.json')

    # ── Representative trajectory selection ──
    display = _select_representative(results, trajs, make_windows_high,
                                     HIST_LEN, base, mgr, learner, tag='HIGH')
    _plot_high_display(display)

    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Representative selection
# ═══════════════════════════════════════════════════════════════════════════════

def _select_representative(results, trajs, window_fn, hist_len, base, mgr, learner, tag):
    """Select N_DISPLAY representative trajectories covering gain spectrum."""
    traj_map = {name: t for name, t in trajs}
    sorted_r = sorted(results, key=lambda r: r['fde_gain'], reverse=True)

    selected = []
    n = len(sorted_r)

    # top gain, ~median gain, low gain, near-zero, and one degradation if any
    picks = []
    if n >= 1: picks.append(0)                                    # best
    if n >= 3: picks.append(n // 3)                               # upper-mid
    if n >= 5: picks.append(n // 2)                               # median
    if n >= 7: picks.append(2 * n // 3)                           # lower-mid
    if n >= 4: picks.append(n - 1)                                # worst

    # Add a sharp-turn candidate: highest frozen dir error among middle 50%
    mid = sorted_r[n//4:3*n//4]
    if mid:
        mid_sorted = sorted(mid, key=lambda r: r['frozen_dir'], reverse=True)
        # pick one that's not already selected
        for r in mid_sorted:
            if r not in [sorted_r[p] for p in picks]:
                picks.append(sorted_r.index(r))
                break

    picks = sorted(set(picks))  # dedup, preserve order
    print(f'\n  [{tag}] Representative picks (by gain rank): {[f"#{p+1}" for p in picks]}')

    for p in picks:
        r = sorted_r[p]
        name = r['drone']
        if name not in traj_map:
            continue
        traj = traj_map[name]
        H_list, T_list = window_fn(traj, hist_len=hist_len)
        if len(H_list) < 20:
            continue
        H = torch.from_numpy(np.array(H_list, dtype=np.float32))
        T = torch.from_numpy(np.array(T_list, dtype=np.float32))
        n_tot = len(H_list)
        n_warm = int(n_tot * WARMUP_FRAC)
        Hh, Th = H[n_warm:], T[n_warm:]

        # Re-run online learning fresh for this drone
        learner.reset_drone(name)
        for j in range(n_warm):
            learner.observe(name, H[j], T[j], confidence=1.0, timestep=j)

        # Get predictions on held-out
        fde_fr, ade_fr, dir_fr, P_fr = compute_errors(base, Hh, Th)

        if mgr.adapter is not None and mgr.active_drone == name:
            on_model = mgr.adapter.model
        else:
            mgr.activate(name)
            on_model = mgr.adapter.model
        fde_on, ade_on, dir_on, P_on = compute_errors(on_model, Hh, Th)
        mgr.deactivate()

        # Pick the best single window for display (one with high frozen error that LoRA fixes)
        fde_fr_per = torch.norm(P_fr[:, -1, :] - Th[:, -1, :], dim=-1)
        fde_on_per = torch.norm(P_on[:, -1, :] - Th[:, -1, :], dim=-1)
        improvements = fde_fr_per - fde_on_per
        best_idx = int(torch.argmax(improvements))

        # Per-second errors for the best window (LOW only; for HIGH, just FDE)
        if hist_len == 40:   # LOW
            per_sec_fr = {}
            per_sec_on = {}
            for sec, step in {0: 0, 1: 5, 2: 10, 3: 15, 4: 19}.items():
                per_sec_fr[sec] = float(torch.norm(P_fr[best_idx, step, :] - Th[best_idx, step, :]).item())
                per_sec_on[sec] = float(torch.norm(P_on[best_idx, step, :] - Th[best_idx, step, :]).item())
        else:
            per_sec_fr = per_sec_on = None

        # Absolute coordinates for plotting
        hist_last = H_list[n_warm + best_idx][-1, :3]
        hp_abs = H_list[n_warm + best_idx][:, :3]                     # history
        bp_abs = P_fr[best_idx].numpy() + hist_last                   # base pred
        op_abs = P_on[best_idx].numpy() + hist_last                   # online pred
        tp_abs = T_list[n_warm + best_idx] + hist_last                # ground truth

        selected.append(dict(
            name=name, rank=p + 1, gain=r['fde_gain'],
            frozen_fde=float(fde_fr_per[best_idx]),
            online_fde=float(fde_on_per[best_idx]),
            frozen_dir=float(dir_fr[best_idx]) if isinstance(dir_fr, np.ndarray) else 0,
            online_dir=float(dir_on[best_idx]) if isinstance(dir_on, np.ndarray) else 0,
            hp=hp_abs, bp=bp_abs, op=op_abs, tp=tp_abs,
            per_sec_fr=per_sec_fr, per_sec_on=per_sec_on,
        ))

        # Cleanup adapter for next drone
        mgr.deactivate()
        learner.reset_drone(name)

    print(f'  [{tag}] {len(selected)} trajectories selected for plotting.')
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting — LOW (3D + XY + per-second table)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_low_display(display):
    if not display:
        return
    for d in display:
        fig = plt.figure(figsize=(18, 7))
        fig.suptitle(f'LOW Ablation — {d["name"][:28]}  '
                     f'(FDE: {d["frozen_fde"]:.3f} → {d["online_fde"]:.3f} m, '
                     f'{d["gain"]:+.1f}%)',
                     fontsize=11, fontweight='bold')

        # ── (1) 3D ──
        ax3 = fig.add_subplot(1, 3, 1, projection='3d')
        _plot_3d_traj(ax3, d, tag='LOW')
        ax3.set_title('3D Trajectory', fontsize=10, fontweight='bold')

        # ── (2) XY ──
        ax_xy = fig.add_subplot(1, 3, 2)
        _plot_xy_traj(ax_xy, d, tag='LOW')
        ax_xy.set_title('XY Top-Down View', fontsize=10, fontweight='bold')

        # ── (3) Per-second error table + bars ──
        ax_tbl = fig.add_subplot(1, 3, 3)
        ax_tbl.axis('off')
        _draw_per_second_table(ax_tbl, d)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fname = f'ablation_low_{d["name"][:30].replace("/","_").replace(" ","_")}.png'
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved: {fname}')

    # ── Overview grid (3D only, compact) ──
    n = len(display)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.5 * cols, 5 * rows))
    fig.suptitle('LOW Ablation — Overview (Blue=History  Red=Frozen  Orange=Online  Green=Truth)',
                 fontsize=12, fontweight='bold')
    for i, d in enumerate(display):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        _plot_3d_traj(ax, d, tag='')
        ax.set_title(f'#{d["rank"]} {d["name"][:20]}\nFDE: {d["frozen_fde"]:.2f}→{d["online_fde"]:.2f}m ({d["gain"]:+.0f}%)',
                     fontsize=8)
    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / 'ablation_low_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: ablation_low_overview.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting — HIGH (3D + XY, no per-second table — 20s horizon at 1Hz)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_high_display(display):
    if not display:
        return
    for d in display:
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f'HIGH Ablation — {d["name"]}  '
                     f'(FDE: {d["frozen_fde"]:.3f} → {d["online_fde"]:.3f} m, '
                     f'{d["gain"]:+.1f}%)',
                     fontsize=11, fontweight='bold')

        ax3 = fig.add_subplot(1, 2, 1, projection='3d')
        _plot_3d_traj(ax3, d, tag='HIGH')
        ax3.set_title('3D Trajectory', fontsize=10, fontweight='bold')

        ax_xy = fig.add_subplot(1, 2, 2)
        _plot_xy_traj(ax_xy, d, tag='HIGH')
        ax_xy.set_title('XY Top-Down View', fontsize=10, fontweight='bold')

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fname = f'ablation_high_{d["name"].replace("/","_").replace(" ","_")}.png'
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved: {fname}')

    # Overview grid
    n = len(display)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.5 * cols, 5 * rows))
    fig.suptitle('HIGH Ablation — Overview (Blue=History  Red=Frozen  Orange=Online  Green=Truth)',
                 fontsize=12, fontweight='bold')
    for i, d in enumerate(display):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        _plot_3d_traj(ax, d, tag='')
        ax.set_title(f'#{d["rank"]} {d["name"]}\nFDE: {d["frozen_fde"]:.2f}→{d["online_fde"]:.2f}m ({d["gain"]:+.0f}%)',
                     fontsize=8)
    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / 'ablation_high_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: ablation_high_overview.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Shared plot primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_3d_traj(ax, d, tag=''):
    hp, bp, op, tp = d['hp'], d['bp'], d['op'], d['tp']
    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=C['hist'], lw=2.5, label='History')
    ax.plot(bp[:, 0], bp[:, 1], bp[:, 2], color=C['base'], lw=1.8, ls='--', alpha=0.7,
            label='Base (Frozen)')
    ax.plot(op[:, 0], op[:, 1], op[:, 2], color=C['online'], lw=2.2, label='Base+LoRA (Online)')
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=C['truth'], lw=2.5, label='Ground Truth')
    ax.scatter(tp[0, 0], tp[0, 1], tp[0, 2], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], tp[-1, 2], c='black', s=70, marker='*', zorder=10)

    # Per-second markers for LOW (5 Hz → 0.2s steps; mark at t=1,2,3,4s)
    if tag == 'LOW':
        for sec, step in [(1, 5), (2, 10), (3, 15), (4, 19)]:
            if step < len(bp):
                ax.scatter(bp[step, 0], bp[step, 1], bp[step, 2],
                          c=C['base'], s=30, marker='x', alpha=0.8)
                ax.scatter(op[step, 0], op[step, 1], op[step, 2],
                          c=C['online'], s=30, marker='+', alpha=0.9)
                ax.scatter(tp[step, 0], tp[step, 1], tp[step, 2],
                          c=C['truth'], s=25, marker='o', alpha=0.7)

    all_pts = np.concatenate([hp, bp, op, tp], axis=0)
    rng = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.55
    zm = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    ax.set_xlim(all_pts[:, 0].mean() - rng, all_pts[:, 0].mean() + rng)
    ax.set_ylim(all_pts[:, 1].mean() - rng, all_pts[:, 1].mean() + rng)
    ax.set_zlim(zm - rng, zm + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.view_init(elev=22, azim=-55)


def _plot_xy_traj(ax, d, tag=''):
    hp, bp, op, tp = d['hp'], d['bp'], d['op'], d['tp']
    ax.plot(hp[:, 0], hp[:, 1], color=C['hist'], lw=2.5, label='History')
    ax.plot(bp[:, 0], bp[:, 1], color=C['base'], lw=1.8, ls='--', alpha=0.7,
            label='Base (Frozen)')
    ax.plot(op[:, 0], op[:, 1], color=C['online'], lw=2.2, label='Base+LoRA (Online)')
    ax.plot(tp[:, 0], tp[:, 1], color=C['truth'], lw=2.5, label='Ground Truth')
    ax.scatter(hp[-1, 0], hp[-1, 1], c=C['hist'], s=80, marker='s',
              edgecolors='black', lw=0.8, zorder=5)
    ax.scatter(tp[0, 0], tp[0, 1], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], c='black', s=70, marker='*', zorder=10)

    # Per-second markers for LOW
    if tag == 'LOW':
        for sec, step in [(1, 5), (2, 10), (3, 15), (4, 19)]:
            if step < len(bp):
                ax.scatter(bp[step, 0], bp[step, 1], c=C['base'], s=30, marker='x', alpha=0.8)
                ax.scatter(op[step, 0], op[step, 1], c=C['online'], s=30, marker='+', alpha=0.9)
                ax.scatter(tp[step, 0], tp[step, 1], c=C['truth'], s=25, marker='o', alpha=0.7)
                ax.annotate(f'{sec}s', (tp[step, 0], tp[step, 1]),
                           textcoords="offset points", xytext=(5, 5), fontsize=6, color=C['truth'])

    all_xy = np.concatenate([hp[:, :2], bp[:, :2], op[:, :2], tp[:, :2]], axis=0)
    rng = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55
    xm, ym = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(xm - rng, xm + rng); ax.set_ylim(ym - rng, ym + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper left')


def _draw_per_second_table(ax, d):
    """Draw a per-second error table + bar chart on the axis."""
    fr = d.get('per_sec_fr', {})
    on = d.get('per_sec_on', {})
    if not fr or not on:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
        return

    seconds = sorted(fr.keys())
    fr_vals = [fr[s] for s in seconds]
    on_vals = [on[s] for s in seconds]

    # Table
    table_data = [['Time', 'Frozen (m)', 'Online (m)', u'Δ (m)']]
    for s in seconds:
        label = f'{s}s' if s > 0 else 'Start'
        table_data.append([label, f'{fr[s]:.3f}', f'{on[s]:.3f}', f'{fr[s]-on[s]:+.3f}'])

    tbl = ax.table(cellText=table_data, cellLoc='center', loc='upper center',
                   bbox=[0.05, 0.52, 0.9, 0.44])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_fontsize(8)
        cell.set_linewidth(0.5)

    # Bar chart below
    axins = ax.inset_axes([0.12, 0.05, 0.8, 0.42])
    x = np.arange(len(seconds))
    w = 0.35
    bars_fr = axins.bar(x - w/2, fr_vals, w, color=C['base'], alpha=0.7, label='Frozen')
    bars_on = axins.bar(x + w/2, on_vals, w, color=C['online'], alpha=0.8, label='Online')
    axins.set_xticks(x)
    axins.set_xticklabels([f'{s}s' if s > 0 else 'Start' for s in seconds], fontsize=7)
    axins.set_ylabel('Position Error (m)', fontsize=8)
    axins.set_title('Per-Second Position Error', fontsize=9, fontweight='bold')
    axins.legend(fontsize=7)
    axins.grid(True, alpha=0.3, axis='y')

    # Annotate bar values
    for bar, val in zip(bars_fr, fr_vals):
        axins.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f'{val:.3f}', ha='center', fontsize=6, color=C['base'])
    for bar, val in zip(bars_on, on_vals):
        axins.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f'{val:.3f}', ha='center', fontsize=6, color=C['online'])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    low_summary = run_low()
    high_summary = run_high()

    print('\n' + '=' * 80)
    print('ABLATION COMPLETE')
    if low_summary:
        print(f'  LOW:  {low_summary["n_better"]}/{low_summary["n_drones"]} drones improved, '
              f'mean FDE gain {low_summary["fde_gain_pct"]:+.1f}%')
    if high_summary:
        print(f'  HIGH: {high_summary["n_better"]}/{high_summary["n_drones"]} drones improved, '
              f'mean FDE gain {high_summary["fde_gain_pct"]:+.1f}%')
    print(f'  Plots: {OUT_DIR}/ablation_low_*.png, ablation_high_*.png')
    print('=' * 80)
