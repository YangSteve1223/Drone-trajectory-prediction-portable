#!/usr/bin/env python3
"""
C6 — Global LoRA + per-drone LoRA STACKING on the 40-frame base.

Three-way comparison per held-out long trajectory:
  (a) 40-frame base            — no adaptation
  (b) base + per-drone LoRA    — current approach
  (c) base + global(merged) + per-drone LoRA — the stacked approach

Mechanism (merge-then-train, since two live LoRALinear can't nest):
  load global_lora_40.pth -> inject structure -> load weights -> merge() folds
  the global delta into the base Linear weights -> unwrap to plain nn.Linear.
  Then per-drone LoRA is trained on this globally-enhanced base exactly like
  eval_lora.py does on the raw base.

PASS: (c) beats (b) on mean per-drone test FDE, or is more stable on unfamiliar
trajectories. Otherwise report "stacking adds no extra gain".
"""

import torch, numpy as np, sys, warnings, json, copy
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emam_model import TrajectoryPredictor
from lora import LoRALinear

# Reuse the exact config + helpers from the per-drone eval
import eval_lora as E40
from eval_lora import (
    LORA_TARGETS, HEAD_TARGETS, HIST_LEN, PRED_LEN, DT, LONG_THRESHOLD,
    EPOCHS, RESTARTS, LR_MAX, LR_MIN, WEIGHT_DECAY, GRAD_CLIP, BATCH_SIZE,
    TRAIN_SPLIT, MIN_TRAINABLE, DIRERR_MAX,
    dir_err, resolve_module, set_module, make_adaptive_windows,
    inject_lora, collect_trainable, save_lora_state, load_lora_state,
    restore_model, compute_loss,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'
WEIGHT_DIR = Path(__file__).resolve().parents[1] / 'weights'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAJECTORIES = 20        # held-out long trajectories to test on


def load_base(hist_len=HIST_LEN):
    m = TrajectoryPredictor(
        input_dim=6, history_len=hist_len, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load(WEIGHT_DIR / 'low_speed_6class_40frame.pth', map_location=DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    return m


def merge_global_lora(model, global_path):
    """Fold the saved global LoRA into the base weights (in place)."""
    g = torch.load(global_path, map_location=DEVICE)
    lls = {}
    for path, rank in LORA_TARGETS:
        orig = resolve_module(model, path)
        lora = LoRALinear(orig, r=rank, alpha=rank * 2.0)
        set_module(model, path, lora)
        lls[path] = lora
    for path, mats in g['lora_state'].items():
        if path in lls:
            lls[path].lora_A.data.copy_(mats['A'].to(DEVICE))
            lls[path].lora_B.data.copy_(mats['B'].to(DEVICE))
    for key, t in g['head_state'].items():
        pth, attr = key.rsplit('.', 1)
        lay = resolve_module(model, pth)
        if attr == 'weight':
            lay.weight.data.copy_(t.to(DEVICE))
    # merge each LoRA into its base Linear, then unwrap
    for path, lora in lls.items():
        lora.merge()
        set_module(model, path, lora.base_layer)
    return model


def eval_fde(model, hists, futs_t, idx):
    preds = []
    for b in range(0, len(idx), 64):
        bi = idx[b:b + 64]
        hb = torch.from_numpy(np.array([hists[i] for i in bi], dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            preds.append(model(hb, force_predict=True)['predictions'].cpu())
    preds = torch.cat(preds, dim=0)
    tt = torch.stack([futs_t[i] for i in idx])
    return torch.norm(preds[:, -1, :] - tt[:, -1, :], dim=-1).mean().item()


def train_perdrone_lora(model, hists, futs_t, bpred, tr_idx, val_idx):
    """Train per-drone LoRA on `model` (may be base or globally-merged base)."""
    tr_h = torch.from_numpy(np.array([hists[i] for i in tr_idx], dtype=np.float32))
    tr_t = torch.stack([futs_t[i] for i in tr_idx])
    tr_bp = torch.stack([bpred[i] for i in tr_idx])
    val_h = torch.from_numpy(np.array([hists[i] for i in val_idx], dtype=np.float32))
    val_t = torch.stack([futs_t[i] for i in val_idx])

    best_val = float('inf'); best_state = None
    for restart in range(RESTARTS):
        torch.manual_seed(42 + restart * 137); np.random.seed(42 + restart * 137)
        ll, hl, ol, ho = inject_lora(model)
        params = collect_trainable(ll, hl)
        opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
        bs = min(BATCH_SIZE, len(tr_idx))
        for ep in range(EPOCHS):
            model.eval(); perm = np.random.permutation(len(tr_idx))
            for b in range(0, len(tr_idx), bs):
                si = perm[b:b + bs]
                hb, tb = tr_h[si].to(DEVICE), tr_t[si].to(DEVICE)
                opt.zero_grad()
                pred = model(hb, force_predict=True)['predictions']
                loss = compute_loss(pred, tb, hb, tr_bp[si].to(DEVICE), ep)
                if torch.isfinite(loss):
                    loss.backward(); torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
            sched.step()
        model.eval(); vfd = []
        for b in range(0, len(val_idx), bs):
            be = min(b + bs, len(val_idx))
            hb = val_h[b:be].to(DEVICE)
            with torch.no_grad():
                pv = model(hb, force_predict=True)['predictions'].cpu()
            vfd.append(torch.norm(pv[:, -1, :] - val_t[b:be][:, -1, :], dim=-1))
        vf = torch.cat(vfd).mean().item()
        if vf < best_val: best_val = vf; best_state = save_lora_state(ll, hl)
        restore_model(model, ol, ho)
    return best_state

def prepare_traj(model_for_base, traj):
    """Build windows and base predictions + train/val/test split for one trajectory."""
    hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
    if len(hists) < MIN_TRAINABLE + 8:
        return None
    all_hist = np.array(hists, dtype=np.float32)
    futs_t = [torch.from_numpy(f).float() for f in futs]
    n_total = len(hists)
    # base predictions (for anchor loss + trainable selection)
    bpred = []
    for b in range(0, n_total, 64):
        hb = torch.from_numpy(all_hist[b:b + 64]).to(DEVICE)
        with torch.no_grad():
            bpred.append(model_for_base(hb, force_predict=True)['predictions'].cpu())
    bpred = torch.cat(bpred, dim=0)
    bdir = np.array([dir_err(bpred[i, -1, :2].numpy(), futs[i][-1, :2]) for i in range(n_total)])
    trainable = np.where(bdir < DIRERR_MAX)[0]
    if len(trainable) < MIN_TRAINABLE:
        return None
    np.random.seed(42); np.random.shuffle(trainable)
    # 70/15/15 train/val/test with guaranteed test samples
    n = len(trainable)
    n_te = max(3, int(n * 0.15))
    n_val = max(3, int(n * 0.15))
    te_idx = trainable[:n_te]
    val_idx = trainable[n_te:n_te + n_val]
    tr_idx = trainable[n_te + n_val:]
    if len(tr_idx) < 5 or len(te_idx) < 3:
        return None
    return dict(hists=hists, futs=futs, futs_t=futs_t, bpred=bpred,
                tr_idx=tr_idx, val_idx=val_idx, te_idx=te_idx)


def main():
    print('=' * 80)
    print('C6 — Global LoRA + per-drone LoRA STACKING (40-frame base)')
    print('=' * 80)

    # Load held-out long trajectories
    all_files = sorted(TRAJ_DIR.glob('*.npz'))
    trajs = []
    for f in all_files:
        d = np.load(f); n = d['traj'].shape[0]
        if n >= LONG_THRESHOLD:
            trajs.append((f.name, d['traj']))
    np.random.seed(7); np.random.shuffle(trajs)
    trajs = trajs[:N_TRAJECTORIES]
    print(f'  Testing on {len(trajs)} held-out long trajectories (>={LONG_THRESHOLD} frames)\n')

    results = []  # per-traj (name, base_fde, b_fde, c_fde)
    for ti, (name, traj) in enumerate(trajs):
        # ---- (a) plain base: build windows + split from the plain base ----
        base_plain = load_base()
        prep = prepare_traj(base_plain, traj)
        if prep is None:
            del base_plain; continue
        te_idx = prep['te_idx']
        base_fde = eval_fde(base_plain, prep['hists'], prep['futs_t'], te_idx)

        # ---- (b) base + per-drone LoRA ----
        st_b = train_perdrone_lora(base_plain, prep['hists'], prep['futs_t'],
                                   prep['bpred'], prep['tr_idx'], prep['val_idx'])
        b_fde = base_fde
        if st_b is not None:
            ll, hl, ol, ho = inject_lora(base_plain)
            load_lora_state(ll, hl, st_b)
            b_fde = eval_fde(base_plain, prep['hists'], prep['futs_t'], te_idx)
            restore_model(base_plain, ol, ho)
        del base_plain

        # ---- (c) base + global(merged) + per-drone LoRA ----
        base_glob = load_base()
        merge_global_lora(base_glob, WEIGHT_DIR / 'global_lora_40.pth')
        # recompute base preds on the globally-enhanced base for its own anchor/selection
        prep_g = prepare_traj(base_glob, traj)
        c_fde = None
        if prep_g is not None:
            st_c = train_perdrone_lora(base_glob, prep_g['hists'], prep_g['futs_t'],
                                       prep_g['bpred'], prep_g['tr_idx'], prep_g['val_idx'])
            # evaluate on the SAME te_idx as (a)/(b) for fair comparison
            if st_c is not None:
                ll, hl, ol, ho = inject_lora(base_glob)
                load_lora_state(ll, hl, st_c)
                c_fde = eval_fde(base_glob, prep['hists'], prep['futs_t'], te_idx)
                restore_model(base_glob, ol, ho)
        del base_glob
        torch.cuda.empty_cache()

        if c_fde is None:
            continue
        results.append((name, base_fde, b_fde, c_fde))
        print(f'  [{ti+1:2d}/{len(trajs)}] {name[:34]:34s}  base={base_fde:.3f}  '
              f'+local={b_fde:.3f}  +global+local={c_fde:.3f}')

    if not results:
        print('No valid trajectories.'); return

    base_arr = np.array([r[1] for r in results])
    b_arr = np.array([r[2] for r in results])
    c_arr = np.array([r[3] for r in results])
    print(f'\n{"=" * 80}')
    print(f'STACKING RESULTS ({len(results)} trajectories)')
    print(f'{"=" * 80}')
    print(f'  {"Config":<32} {"mean FDE":<12} {"median FDE":<12}')
    print(f'  {"-" * 56}')
    print(f'  {"(a) 40-frame base":<32} {base_arr.mean():<12.4f} {np.median(base_arr):<12.4f}')
    print(f'  {"(b) base + per-drone LoRA":<32} {b_arr.mean():<12.4f} {np.median(b_arr):<12.4f}')
    print(f'  {"(c) base + global + per-drone":<32} {c_arr.mean():<12.4f} {np.median(c_arr):<12.4f}')
    print(f'\n  (b) vs base:  {(base_arr.mean()-b_arr.mean())/base_arr.mean()*100:+.1f}%')
    print(f'  (c) vs base:  {(base_arr.mean()-c_arr.mean())/base_arr.mean()*100:+.1f}%')
    print(f'  (c) vs (b):   {(b_arr.mean()-c_arr.mean())/max(b_arr.mean(),1e-6)*100:+.1f}%  '
          f'<-- does stacking help?')
    n_c_wins = int(np.sum(c_arr < b_arr))
    print(f'  (c) beats (b) on {n_c_wins}/{len(results)} trajectories')

    summary = {
        'n_traj': len(results),
        'base_mean_fde': float(base_arr.mean()),
        'local_mean_fde': float(b_arr.mean()),
        'stacked_mean_fde': float(c_arr.mean()),
        'local_vs_base_pct': float((base_arr.mean() - b_arr.mean()) / base_arr.mean() * 100),
        'stacked_vs_base_pct': float((base_arr.mean() - c_arr.mean()) / base_arr.mean() * 100),
        'stacked_vs_local_pct': float((b_arr.mean() - c_arr.mean()) / max(b_arr.mean(), 1e-6) * 100),
        'stacked_wins': n_c_wins,
        'per_traj': [{'name': r[0], 'base': r[1], 'local': r[2], 'stacked': r[3]} for r in results],
    }
    json.dump(summary, open(OUT_DIR / 'lora_stack_40.json', 'w'), indent=2)
    print(f'\n  Results saved: pic-results/lora_stack_40.json')
    print('=' * 80)


if __name__ == '__main__':
    main()
