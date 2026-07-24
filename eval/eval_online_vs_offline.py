#!/usr/bin/env python3
"""
Phase 5 helper — Online incremental vs Offline one-shot per-drone LoRA.

Same drones, same warmup/held-out split. Three arms scored on held-out:
  FROZEN  : 40-frame base + merged global LoRA (no per-drone)
  ONLINE  : streaming incremental LoRA (OnlineLearner, gentle config)
  OFFLINE : batch-train a per-drone LoRA on the warmup windows (full epochs +
            restarts, like eval_lora) then freeze and eval.

Answers: is edge streaming worth it, or is "land then batch-finetune" better?
"""

import torch, numpy as np, sys, warnings, json, shutil
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import online_config as OC
from adapter_manager import DroneAdapterManager
from online_learner import OnlineLearner, OnlineLearnerConfig
from lora import LoRAAdapter
from eval_lora import make_adaptive_windows, dir_err, HIST_LEN, PRED_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'

MIN_LEN = 200
N_DRONES = 20
WARMUP_FRAC = 0.6
# Offline batch config (on warmup windows only)
OFF_EPOCHS, OFF_LR, OFF_WD = 30, 1e-3, 1e-4
DT = 0.2


def eval_fde(model, H, T):
    with torch.no_grad():
        P = torch.cat([model(H[b:b+64].to(DEVICE), force_predict=True)['predictions'].cpu()
                       for b in range(0, len(H), 64)])
    return torch.norm(P[:, -1, :] - T[:, -1, :], dim=-1).mean().item()


def offline_loss(pred, tgt, hist):
    sp = 100.0
    pn, tn = pred / sp, tgt / sp
    lh = F.smooth_l1_loss(pn, tn, beta=0.2)
    pv, tv = pn[:, 1:] - pn[:, :-1], tn[:, 1:] - tn[:, :-1]
    ld = (1.0 - F.cosine_similarity(pv, tv, dim=-1)).mean()
    ec = (hist[:, -1, 3:6] * DT) / sp
    lb = ((pn[:, 0, :] - ec) ** 2).mean()
    return lh + 0.3 * ld + 0.4 * lb


def train_offline_lora(base, Hw, Tw):
    """Batch-train a per-drone LoRA on warmup windows; return a fresh eval model."""
    model = OC.build_online_base(device=DEVICE, with_global=True)  # same base as frozen
    ad = LoRAAdapter(model, lora_targets=OC.ONLINE_LORA_TARGETS, head_targets=OC.ONLINE_HEAD_TARGETS)
    ad.activate()
    params = ad.get_trainable_params()
    opt = torch.optim.AdamW(params, lr=OFF_LR, weight_decay=OFF_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OFF_EPOCHS, eta_min=1e-5)
    idx = np.arange(len(Hw)); bs = min(32, len(Hw))
    model.eval()
    for ep in range(OFF_EPOCHS):
        np.random.shuffle(idx)
        for b in range(0, len(idx), bs):
            bi = idx[b:b+bs]
            hb, tb = Hw[bi].to(DEVICE), Tw[bi].to(DEVICE)
            opt.zero_grad()
            pred = model(hb, force_predict=True)['predictions']
            loss = offline_loss(pred, tb, hb)
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
    return model  # LoRA active, trained


def main():
    print('=' * 80)
    print('Phase 5 — Online (streaming) vs Offline (batch) per-drone LoRA')
    print('=' * 80)
    ck = Path(__file__).resolve().parents[1] / 'weights' / '_ovo_ck'
    shutil.rmtree(ck, ignore_errors=True)

    trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        t = np.load(f)['traj']
        if t.shape[0] >= MIN_LEN:
            trajs.append((f.stem, t))
    np.random.seed(11); np.random.shuffle(trajs); trajs = trajs[:N_DRONES]
    print(f'  {len(trajs)} drones\n')

    base = OC.build_online_base(device=DEVICE, with_global=True)
    mgr = DroneAdapterManager(OC.build_online_base(device=DEVICE, with_global=True),
                              checkpoint_dir=str(ck))
    learner = OnlineLearner(mgr, OnlineLearnerConfig(device=str(DEVICE), conf_threshold=0.0))

    rows = []
    for i, (did, traj) in enumerate(trajs):
        h, fu = make_adaptive_windows(traj, hist_len=HIST_LEN)
        if len(h) < 20: continue
        H = torch.from_numpy(np.array(h, dtype=np.float32))
        T = torch.from_numpy(np.array(fu, dtype=np.float32))
        nw = int(len(h) * WARMUP_FRAC)
        if nw < 20 or len(h) - nw < 5: continue
        Hw, Tw, Hh, Th = H[:nw], T[:nw], H[nw:], T[nw:]

        f_frozen = eval_fde(base, Hh, Th)

        # online
        learner.reset_drone(did)
        for j in range(nw):
            learner.observe(did, Hw[j], Tw[j], confidence=1.0)
        f_online = (eval_fde(mgr.adapter.model, Hh, Th)
                    if mgr.adapter is not None and mgr.active_drone == did else f_frozen)
        mgr.deactivate()

        # offline
        off_model = train_offline_lora(base, Hw, Tw)
        f_offline = eval_fde(off_model, Hh, Th)
        del off_model; torch.cuda.empty_cache()

        rows.append((did, f_frozen, f_online, f_offline))
        print(f'  [{i+1:2d}/{len(trajs)}] {did[:20]:20s} frozen={f_frozen:.3f} '
              f'online={f_online:.3f} offline={f_offline:.3f}')

    fr = np.array([r[1] for r in rows]); on = np.array([r[2] for r in rows]); of = np.array([r[3] for r in rows])
    print(f'\n{"=" * 80}')
    print(f'ONLINE vs OFFLINE ({len(rows)} drones)')
    print(f'{"=" * 80}')
    print(f'  {"arm":<10} {"mean FDE":<12} {"vs frozen":<12} {"wins/frozen":<12}')
    print(f'  {"frozen":<10} {fr.mean():<12.4f} {"—":<12}')
    print(f'  {"online":<10} {on.mean():<12.4f} {(fr.mean()-on.mean())/fr.mean()*100:>+8.1f}%   {int((on<fr).sum())}/{len(rows)}')
    print(f'  {"offline":<10} {of.mean():<12.4f} {(fr.mean()-of.mean())/fr.mean()*100:>+8.1f}%   {int((of<fr).sum())}/{len(rows)}')
    print(f'\n  offline vs online: {(on.mean()-of.mean())/max(on.mean(),1e-6)*100:+.1f}% '
          f'(offline better on {int((of<on).sum())}/{len(rows)})')

    summary = {'n_drones': len(rows),
               'frozen_fde': float(fr.mean()), 'online_fde': float(on.mean()), 'offline_fde': float(of.mean()),
               'online_gain_pct': float((fr.mean()-on.mean())/fr.mean()*100),
               'offline_gain_pct': float((fr.mean()-of.mean())/fr.mean()*100),
               'per_drone': [{'drone': r[0], 'frozen': r[1], 'online': r[2], 'offline': r[3]} for r in rows]}
    json.dump(summary, open(OUT_DIR / 'online_vs_offline.json', 'w'), indent=2)
    print(f'\n  Saved: pic-results/online_vs_offline.json')
    shutil.rmtree(ck, ignore_errors=True)
    print('=' * 80)


if __name__ == '__main__':
    main()
