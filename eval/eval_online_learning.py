#!/usr/bin/env python3
"""
Phase 4 — End-to-end validation of per-drone ONLINE learning.

Question this answers: does streaming per-drone LoRA adaptation actually make
predictions better, and does the drone "get more accurate the longer it flies"?

Setup (single-flight, most realistic — each drone has one flight of data):
  - Each long trajectory (>=MIN_LEN frames) = one drone's flight.
  - Stream it frame by frame. At each step build a 40-frame adaptive-stride
    history window and its 20-frame future.
  - Split the flight's windows chronologically:
        WARMUP (first WARMUP_FRAC): online-learn the per-drone LoRA here.
        HELDOUT (rest): frozen — predict only, no learning — and score.
  - Compare on the SAME held-out windows:
        (A) FROZEN  = 40-frame base + merged global LoRA, no per-drone learning
        (B) ONLINE  = same base, with the per-drone LoRA learned during warmup
  - Also record FDE vs #warmup-windows-seen: re-evaluate
    the held-out set after every few online updates.

Outputs: pic-results/online_learning.json + online_learning.png
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
from eval_lora import make_adaptive_windows, dir_err, HIST_LEN, PRED_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
CKPT_DIR = Path(__file__).resolve().parents[1] / 'weights' / 'online_adapters_eval'

MIN_LEN = 200            # frames; long enough for warmup + heldout windows
N_DRONES = 30
WARMUP_FRAC = 0.6        # first 60% of windows for online learning
ACCUM = 10               # online update every 10 observations (gentler; see study)
CURVE_EVERY = 1          # re-eval heldout every N online updates for the curve


def fde_ade(model, H, T):
    preds = []
    for b in range(0, len(H), 64):
        with torch.no_grad():
            preds.append(model(H[b:b + 64].to(DEVICE), force_predict=True)['predictions'].cpu())
    P = torch.cat(preds, dim=0)
    fde = torch.norm(P[:, -1, :] - T[:, -1, :], dim=-1).mean().item()
    ade = torch.norm(P - T, dim=-1).mean(dim=1).mean().item()
    de = np.array([dir_err(P[i, -1, :2].numpy(), T[i, -1, :2].numpy()) for i in range(len(H))])
    return fde, ade, float(de.mean())


def run_drone(drone_id, traj, base_frozen, mgr, learner):
    hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
    if len(hists) < 20:
        return None
    H = torch.from_numpy(np.array(hists, dtype=np.float32))
    T = torch.from_numpy(np.array(futs, dtype=np.float32))
    n = len(hists); n_warm = int(n * WARMUP_FRAC)
    if n_warm < ACCUM * 2 or (n - n_warm) < 5:
        return None
    Hw, Tw = H[:n_warm], T[:n_warm]
    Hh, Th = H[n_warm:], T[n_warm:]

    # (A) FROZEN baseline on held-out
    frozen_fde, frozen_ade, frozen_dir = fde_ade(base_frozen, Hh, Th)

    # (B) ONLINE: stream warmup windows, learn per-drone LoRA, track heldout curve.
    # The learner keeps the drone's adapter RESIDENT (active) across updates, so
    # we evaluate on mgr.adapter.model directly — reactivating would reload stale
    # disk state and wipe the in-memory learned LoRA.
    learner.reset_drone(drone_id)
    curve = []  # (updates_seen, heldout_fde)
    n_updates = 0
    for i in range(n_warm):
        updated = learner.observe(
            drone_id, Hw[i], Tw[i], confidence=1.0, intent=0, timestep=i)
        if updated:
            n_updates += 1
            if n_updates % CURVE_EVERY == 0 and mgr.adapter is not None:
                f, _, _ = fde_ade(mgr.adapter.model, Hh, Th)
                curve.append((n_updates, f))

    # Final online eval on held-out — use the resident (trained) adapter.
    if mgr.adapter is not None and mgr.active_drone == drone_id:
        online_fde, online_ade, online_dir = fde_ade(mgr.adapter.model, Hh, Th)
    else:
        # fell back to disk (e.g. never updated) — activate to load whatever exists
        mgr.activate(drone_id)
        online_fde, online_ade, online_dir = fde_ade(mgr.adapter.model, Hh, Th)
    mgr.deactivate()

    return dict(
        drone=drone_id, n_windows=n, n_warm=n_warm, n_heldout=int(n - n_warm),
        n_updates=n_updates,
        frozen_fde=frozen_fde, frozen_ade=frozen_ade, frozen_dir=frozen_dir,
        online_fde=online_fde, online_ade=online_ade, online_dir=online_dir,
        curve=curve,
    )


def main():
    print('=' * 80)
    print('Phase 4 — Online per-drone learning: FROZEN vs ONLINE (single-flight)')
    print('=' * 80)
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR, ignore_errors=True)

    # Held-out long trajectories = drones
    trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); t = d['traj']
        if t.shape[0] >= MIN_LEN:
            trajs.append((f.stem, t))
    np.random.seed(11); np.random.shuffle(trajs)
    trajs = trajs[:N_DRONES]
    print(f'  {len(trajs)} drones (long flights >= {MIN_LEN} frames)\n')

    # Shared frozen base (40f + merged global LoRA) and the online base+manager
    base_frozen = OC.build_online_base(device=DEVICE, with_global=True)
    base_online = OC.build_online_base(device=DEVICE, with_global=True)
    mgr = DroneAdapterManager(base_online, checkpoint_dir=str(CKPT_DIR))
    cfg = OnlineLearnerConfig(accumulation_steps=ACCUM, device=str(DEVICE),
                              conf_threshold=0.0)  # accept all obs in this sim
    learner = OnlineLearner(mgr, cfg)

    results = []
    for i, (drone_id, traj) in enumerate(trajs):
        r = run_drone(drone_id, traj, base_frozen, mgr, learner)
        if r is None:
            continue
        results.append(r)
        gain = (r['frozen_fde'] - r['online_fde']) / max(r['frozen_fde'], 1e-6) * 100
        print(f'  [{i+1:2d}/{len(trajs)}] {drone_id[:22]:22s} '
              f'frozen={r["frozen_fde"]:.3f} online={r["online_fde"]:.3f} '
              f'({gain:+.1f}%) upd={r["n_updates"]}')

    if not results:
        print('No valid drones.'); return

    fr = np.array([r['frozen_fde'] for r in results])
    on = np.array([r['online_fde'] for r in results])
    frd = np.array([r['frozen_dir'] for r in results])
    ond = np.array([r['online_dir'] for r in results])
    n_better = int((on < fr).sum())

    print(f'\n{"=" * 80}')
    print(f'ONLINE LEARNING RESULTS ({len(results)} drones)')
    print(f'{"=" * 80}')
    print(f'  {"":16} {"FROZEN":<12} {"ONLINE":<12} {"Change":<10}')
    print(f'  {"mean FDE":16} {fr.mean():<12.4f} {on.mean():<12.4f} '
          f'{(fr.mean()-on.mean())/fr.mean()*100:>+7.1f}%')
    print(f'  {"median FDE":16} {np.median(fr):<12.4f} {np.median(on):<12.4f}')
    print(f'  {"mean Dir(deg)":16} {frd.mean():<12.2f} {ond.mean():<12.2f} '
          f'{(frd.mean()-ond.mean())/max(frd.mean(),0.1)*100:>+7.1f}%')
    print(f'  online beats frozen on {n_better}/{len(results)} drones')

    # ---- Plot ----
    fig = plt.figure(figsize=(14, 4.2))
    # (1) per-drone frozen vs online scatter
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(fr, on, c=['#2E7D32' if o < f else '#C62828' for f, o in zip(fr, on)], s=28)
    lim = max(fr.max(), on.max()) * 1.05
    ax1.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)
    ax1.set_xlabel('FROZEN FDE (m)'); ax1.set_ylabel('ONLINE FDE (m)')
    ax1.set_title(f'Per-drone held-out FDE\n(below diagonal = online better, {n_better}/{len(results)})',
                  fontsize=9, fontweight='bold')
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)

    # (2) mean improvement bar
    ax2 = fig.add_subplot(1, 3, 2)
    bars = ax2.bar(['FROZEN', 'ONLINE'], [fr.mean(), on.mean()],
                   color=['#90A4AE', '#FF6D00'], width=0.6)
    for b, v in zip(bars, [fr.mean(), on.mean()]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    ax2.set_ylabel('mean held-out FDE (m)')
    ax2.set_title('Frozen vs Online (mean)', fontsize=9, fontweight='bold')

    # (3) Normalized heldout FDE vs online updates seen (avg over drones)
    ax3 = fig.add_subplot(1, 3, 3)
    # align curves by update index, normalize each drone by its frozen FDE
    max_u = max((r['curve'][-1][0] for r in results if r['curve']), default=0)
    if max_u > 0:
        grid = np.arange(1, max_u + 1)
        acc = np.zeros(len(grid)); cnt = np.zeros(len(grid))
        for r in results:
            if not r['curve']:
                continue
            us = np.array([c[0] for c in r['curve']])
            fs = np.array([c[1] for c in r['curve']]) / max(r['frozen_fde'], 1e-6)
            for gi, g in enumerate(grid):
                # last curve value with update <= g
                m = us <= g
                if m.any():
                    acc[gi] += fs[m][-1]; cnt[gi] += 1
        valid = cnt > 0
        ax3.plot(grid[valid], acc[valid] / cnt[valid], color='#FF6D00', lw=2)
        ax3.axhline(1.0, color='#90A4AE', ls='--', lw=1, label='frozen baseline')
        ax3.set_xlabel('online updates seen'); ax3.set_ylabel('heldout FDE / frozen FDE')
        ax3.set_title('Does it get better the longer it flies?', fontsize=9, fontweight='bold')
        ax3.legend(fontsize=7)
    fig.suptitle('Per-drone Online Learning — 40-frame base + global LoRA, streaming per-drone LoRA',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / 'online_learning.png', bbox_inches='tight'); plt.close(fig)

    summary = {
        'n_drones': len(results), 'n_better': n_better,
        'frozen_mean_fde': float(fr.mean()), 'online_mean_fde': float(on.mean()),
        'fde_gain_pct': float((fr.mean() - on.mean()) / fr.mean() * 100),
        'frozen_mean_dir': float(frd.mean()), 'online_mean_dir': float(ond.mean()),
        'config': {'min_len': MIN_LEN, 'warmup_frac': WARMUP_FRAC, 'accum': ACCUM},
        'per_drone': [{k: r[k] for k in ('drone', 'n_windows', 'n_warm', 'n_heldout',
                       'n_updates', 'frozen_fde', 'online_fde', 'frozen_dir', 'online_dir')}
                      for r in results],
    }
    json.dump(summary, open(OUT_DIR / 'online_learning.json', 'w'), indent=2)
    print(f'\n  Saved: pic-results/online_learning.json, online_learning.png')
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print('=' * 80)


if __name__ == '__main__':
    main()
