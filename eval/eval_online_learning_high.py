#!/usr/bin/env python3
"""
Phase B — Online per-drone learning for the HIGH model (SimCruise, 1Hz, 4-class).

HIGH online learning uses ONLY: base HIGH model + per-drone LoRA (NO global layer).
Same causal protocol as the LOW study: each long SimCruise trajectory = one drone;
stream the first WARMUP_FRAC to learn a per-drone LoRA, evaluate on the held-out rest.

HIGH specifics:
  - 20-frame history (40-frame expansion was rejected), 1 Hz -> DT=1.0.
  - Windows are position-centered on the first frame (SimCruise convention).
  - No global LoRA to merge.

Output: pic-results/online_learning_high.json
"""

import torch, numpy as np, sys, warnings, json, shutil
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import online_config as OC
from adapter_manager import DroneAdapterManager
from online_learner import OnlineLearner, OnlineLearnerConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIM_DIR = Path(__file__).resolve().parents[2] / 'SimCruise'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
CKPT = Path(__file__).resolve().parents[1] / 'weights' / '_online_high_ck'

HIST_LEN = 20
PRED_LEN = 20
MIN_LEN = 120           # SimCruise frames (1Hz) to be a usable "drone flight"
N_DRONES = 25
WARMUP_FRAC = 0.6
ACCUM = 10


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def load_high_trajs(max_trajs):
    merged = sorted(SIM_DIR.rglob('trajectories_merged.npz'))
    if not merged:
        raise FileNotFoundError('trajectories_merged.npz not found')
    d = np.load(merged[0])
    pos, vel, mask = d['positions'], d['velocities'], d['masks']
    lengths = mask.sum(axis=1)
    idx = np.where(lengths >= MIN_LEN)[0]
    rng = np.random.RandomState(11); rng.shuffle(idx)
    idx = idx[:max_trajs]
    out = []
    for i in idx:
        L = int(lengths[i])
        p = pos[i, :L].astype(np.float32); v = vel[i, :L].astype(np.float32)
        out.append(np.concatenate([p, v], axis=1))
    return out


def make_windows_high(traj):
    """20-frame stride-1 windows, position-centered on each window's first frame."""
    n = traj.shape[0]
    H, T = [], []
    for j in range(HIST_LEN - 1, n - PRED_LEN):
        h = traj[j - HIST_LEN + 1:j + 1].copy()
        fut_abs = traj[j + 1:j + 1 + PRED_LEN, :3]
        if fut_abs.shape[0] < PRED_LEN:
            continue
        fut = fut_abs - traj[j, :3]
        h[:, :3] -= h[0, :3]            # HIGH centering
        H.append(h); T.append(fut)
    return H, T


def eval_fde(model, H, T):
    with torch.no_grad():
        P = torch.cat([model(H[b:b + 64].to(DEVICE), force_predict=True)['predictions'].cpu()
                       for b in range(0, len(H), 64)])
    fde = torch.norm(P[:, -1, :] - T[:, -1, :], dim=-1).mean().item()
    de = np.array([dir_err(P[i, -1, :2].numpy(), T[i, -1, :2].numpy()) for i in range(len(H))])
    return fde, float(de.mean())


def main():
    print('=' * 80)
    print('Phase B — HIGH online per-drone learning (base + per-drone LoRA, NO global)')
    print('=' * 80)
    shutil.rmtree(CKPT, ignore_errors=True)

    trajs = load_high_trajs(N_DRONES)
    print(f'  {len(trajs)} HIGH drones (SimCruise flights >= {MIN_LEN} frames)\n')

    base = OC.build_high_base(device=DEVICE)           # frozen reference
    mgr = DroneAdapterManager(OC.build_high_base(device=DEVICE), checkpoint_dir=str(CKPT))
    cfg = OnlineLearnerConfig(accumulation_steps=ACCUM, device=str(DEVICE),
                              conf_threshold=0.0, dt=OC.HIGH_DT)
    learner = OnlineLearner(mgr, cfg)

    rows = []
    for i, traj in enumerate(trajs):
        H_list, T_list = make_windows_high(traj)
        if len(H_list) < 30:
            continue
        H = torch.from_numpy(np.array(H_list, dtype=np.float32))
        T = torch.from_numpy(np.array(T_list, dtype=np.float32))
        nw = int(len(H_list) * WARMUP_FRAC)
        if nw < ACCUM * 2 or len(H_list) - nw < 5:
            continue
        Hw, Tw, Hh, Th = H[:nw], T[:nw], H[nw:], T[nw:]

        f_fr, d_fr = eval_fde(base, Hh, Th)
        did = f'high_{i}'
        learner.reset_drone(did)
        for j in range(nw):
            learner.observe(did, Hw[j], Tw[j], confidence=1.0, timestep=j)
        if mgr.adapter is not None and mgr.active_drone == did:
            f_on, d_on = eval_fde(mgr.adapter.model, Hh, Th)
        else:
            f_on, d_on = f_fr, d_fr
        mgr.deactivate()

        rows.append((did, f_fr, f_on, d_fr, d_on))
        gain = (f_fr - f_on) / max(f_fr, 1e-6) * 100
        print(f'  [{i+1:2d}/{len(trajs)}] frozen={f_fr:.3f} online={f_on:.3f} ({gain:+.1f}%)')

    if not rows:
        print('No valid HIGH drones.'); return
    fr = np.array([r[1] for r in rows]); on = np.array([r[2] for r in rows])
    dfr = np.array([r[3] for r in rows]); don = np.array([r[4] for r in rows])
    n_better = int((on < fr).sum())

    print(f'\n{"=" * 80}')
    print(f'HIGH ONLINE LEARNING RESULTS ({len(rows)} drones)')
    print(f'{"=" * 80}')
    print(f'  {"":14} {"FROZEN":<12} {"ONLINE":<12} {"Change":<10}')
    print(f'  {"mean FDE":14} {fr.mean():<12.4f} {on.mean():<12.4f} '
          f'{(fr.mean()-on.mean())/fr.mean()*100:>+7.1f}%')
    print(f'  {"median FDE":14} {np.median(fr):<12.4f} {np.median(on):<12.4f}')
    print(f'  {"mean Dir(deg)":14} {dfr.mean():<12.2f} {don.mean():<12.2f} '
          f'{(dfr.mean()-don.mean())/max(dfr.mean(),0.1)*100:>+7.1f}%')
    print(f'  online beats frozen on {n_better}/{len(rows)} drones')

    summary = {'n_drones': len(rows), 'n_better': n_better,
               'frozen_mean_fde': float(fr.mean()), 'online_mean_fde': float(on.mean()),
               'fde_gain_pct': float((fr.mean() - on.mean()) / fr.mean() * 100),
               'frozen_mean_dir': float(dfr.mean()), 'online_mean_dir': float(don.mean()),
               'per_drone': [{'drone': r[0], 'frozen_fde': r[1], 'online_fde': r[2]} for r in rows]}
    json.dump(summary, open(OUT_DIR / 'online_learning_high.json', 'w'), indent=2)
    print(f'\n  Saved: pic-results/online_learning_high.json')
    shutil.rmtree(CKPT, ignore_errors=True)
    print('=' * 80)


if __name__ == '__main__':
    main()
