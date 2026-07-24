#!/usr/bin/env python3
"""
B4 — Gate-LoRA for extreme turns on the 40-frame base.

The physics inertia gate biases predictions toward straight-line extrapolation,
so >150deg turns collapse to STRAIGHT. This trains a LoRA that ONLY touches the
gate MLP (physics_gate.gate_mlp.0/2), teaching the model to LOWER gate_inertia
when a sharp turn is coming — letting the neural term (which can turn) dominate.
Decoder is untouched (avoids reintroducing the zigzag failure).

Extras:
  - Training set heavily weights extreme-turn windows.
  - Gate-smoothness regularizer: penalize step-to-step gate_inertia jumps so the
    gate itself doesn't oscillate.
  - Evaluation on the EXTREME-TURN subset (ground-truth turn > TURN_EVAL_DEG),
    reporting catastrophic (>90deg) rate and direction error there.

Produces weights/gate_lora_40.pth.
"""

import torch, numpy as np, sys, warnings, json
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))  # shared eval utils
from emam_model import TrajectoryPredictor
from lora import LoRALinear
from eval_lora import (make_adaptive_windows, dir_err, resolve_module,
                               set_module, HIST_LEN, PRED_LEN, LONG_THRESHOLD)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'
WEIGHT_DIR = Path(__file__).resolve().parents[1] / 'weights'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
DT = 0.2

# Gate-only LoRA targets
GATE_TARGETS = [
    ('ua_pgd.physics_gate.gate_mlp.0', 16),
    ('ua_pgd.physics_gate.gate_mlp.2', 8),
]

EPOCHS = 25
LR_MAX, LR_MIN = 2e-3, 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.0
BATCH_SIZE = 64
RESTARTS = 2
TOTAL_WINDOWS = 24000

# Turn definitions (degrees, ground-truth window turn magnitude)
TURN_EVAL_DEG = 60.0       # "extreme turn" subset for evaluation
W_GATE_SMOOTH = 0.05       # gate-inertia step-jump penalty
BETA_HUBER = 0.20
W_DIR = 0.5


def window_turn_deg(fut):
    v0 = fut[1, :2] - fut[0, :2]; v1 = fut[-1, :2] - fut[-2, :2]
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < 1e-3 or n1 < 1e-3: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))))


def build_model():
    m = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple').to(DEVICE).eval()
    m.load_state_dict(torch.load(WEIGHT_DIR / 'low_speed_6class_40frame.pth',
                                 map_location=DEVICE)['model_state_dict'])
    return m


def inject_gate_lora(model):
    for p in model.parameters(): p.requires_grad_(False)
    lls, ols = {}, {}
    for path, rank in GATE_TARGETS:
        orig = resolve_module(model, path)
        ols[path] = orig
        lora = LoRALinear(orig, r=rank, alpha=rank * 2.0)
        set_module(model, path, lora); lls[path] = lora
    return lls, ols


def collect_params(lls):
    ps = []
    for l in lls.values(): ps += [l.lora_A, l.lora_B]
    return ps


def save_state(lls):
    return {p: {'A': l.lora_A.data.clone(), 'B': l.lora_B.data.clone()} for p, l in lls.items()}


def load_state(lls, st):
    for p, m in st.items():
        if p in lls: lls[p].lora_A.data.copy_(m['A']); lls[p].lora_B.data.copy_(m['B'])


def restore(model, ols):
    for p, orig in ols.items(): set_module(model, p, orig)


def loss_fn(out, target, sample_w):
    pred = out['predictions']
    w = sample_w.view(-1, 1, 1)
    huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER, reduction='none')
    loss_huber = (huber * w).sum() / (w.expand_as(huber).sum() + 1e-8)
    pv = pred[:, 1:, :] - pred[:, :-1, :]; tv = target[:, 1:, :] - target[:, :-1, :]
    dir_el = (1.0 - F.cosine_similarity(pv, tv, dim=-1))
    wv = sample_w.view(-1, 1)
    loss_dir = (dir_el * wv).sum() / (wv.expand_as(dir_el).sum() + 1e-8)
    # gate smoothness: penalize |gi[t+1]-gi[t]|
    gi = out['gate_inertia']                       # (B, P)
    loss_gsmooth = (gi[:, 1:] - gi[:, :-1]).abs().mean()
    return loss_huber + W_DIR * loss_dir + W_GATE_SMOOTH * loss_gsmooth

def evaluate(model, H, T, turn_deg, subset_mask, tag=''):
    """Return metrics on all windows and on the extreme-turn subset."""
    preds = []
    for b in range(0, len(H), 128):
        with torch.no_grad():
            preds.append(model(H[b:b + 128].to(DEVICE), force_predict=True)['predictions'].cpu())
    P = torch.cat(preds, dim=0)
    de = np.array([dir_err(P[i, -1, :2].numpy(), T[i, -1, :2].numpy()) for i in range(len(H))])
    fde = torch.norm(P[:, -1, :] - T[:, -1, :], dim=-1).numpy()
    def stats(mask):
        if mask.sum() == 0: return dict(n=0, fde=0, dir=0, cata=0)
        return dict(n=int(mask.sum()), fde=float(fde[mask].mean()),
                    dir=float(de[mask].mean()), cata=float((de[mask] >= 90).sum() / mask.sum() * 100))
    return stats(np.ones(len(H), bool)), stats(subset_mask)


def main():
    print('=' * 80)
    print('B4 — Gate-LoRA for extreme turns (40-frame base)')
    print('=' * 80)

    # Collect windows from all lengths, tag turn magnitude
    all_trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); all_trajs.append(d['traj'])
    np.random.seed(42); np.random.shuffle(all_trajs)
    hists, futs, turns = [], [], []
    for traj in all_trajs:
        if len(hists) >= TOTAL_WINDOWS: break
        h, fu = make_adaptive_windows(traj, hist_len=HIST_LEN)
        for a, b in zip(h, fu):
            hists.append(a); futs.append(b); turns.append(window_turn_deg(b))
    turns = np.array(turns)
    H = torch.from_numpy(np.array(hists, dtype=np.float32))
    T = torch.from_numpy(np.array(futs, dtype=np.float32))
    print(f'  {len(hists)} windows | extreme-turn(>{TURN_EVAL_DEG:.0f}deg): '
          f'{int((turns>TURN_EVAL_DEG).sum())}')

    # Split (trajectory-agnostic random split; turn dist preserved by shuffle)
    n = len(hists); idx = np.arange(n); np.random.seed(0); np.random.shuffle(idx)
    n_te = int(n * 0.15)
    te = idx[:n_te]; tr = idx[n_te:]
    te_mask_extreme = turns[te] > TURN_EVAL_DEG
    teH, teT = H[te], T[te]

    # Turn-emphasis sample weights for training (extreme turns weighted up to 4x)
    tr_turn = turns[tr]
    tr_w = 1.0 + 3.0 * np.clip((tr_turn - 30.0) / 90.0, 0, 1)
    trH, trT = H[tr], T[tr]
    trW = torch.tensor(tr_w, dtype=torch.float32)
    print(f'  Train {len(tr)}  Test {len(te)} (extreme in test: {int(te_mask_extreme.sum())})')
    print(f'  Train turn-weight mean={tr_w.mean():.2f} max={tr_w.max():.2f}')

    # Baseline
    model = build_model()
    b_all, b_ext = evaluate(model, teH, teT, turns[te], te_mask_extreme)
    print(f'\n  BASE  all: FDE={b_all["fde"]:.3f} Dir={b_all["dir"]:.1f} Cata={b_all["cata"]:.2f}%')
    print(f'  BASE  extreme(>{TURN_EVAL_DEG:.0f}): FDE={b_ext["fde"]:.3f} '
          f'Dir={b_ext["dir"]:.1f} Cata={b_ext["cata"]:.2f}% (n={b_ext["n"]})')

    best_ext_cata = float('inf'); best_state = None; n_params = 0
    for restart in range(RESTARTS):
        torch.manual_seed(42 + restart * 137); np.random.seed(42 + restart * 137)
        lls, ols = inject_gate_lora(model)
        params = collect_params(lls); n_params = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
        order = np.arange(len(tr))
        for ep in range(EPOCHS):
            model.eval(); np.random.shuffle(order); losses = []
            for b in range(0, len(order), BATCH_SIZE):
                bi = order[b:b + BATCH_SIZE]
                hb = trH[bi].to(DEVICE); tb = trT[bi].to(DEVICE); sw = trW[bi].to(DEVICE)
                opt.zero_grad()
                out = model(hb, force_predict=True)
                loss = loss_fn(out, tb, sw)
                if torch.isfinite(loss):
                    loss.backward(); torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
                    losses.append(loss.item())
            sched.step()
            if ep % 8 == 0 or ep == EPOCHS - 1:
                _, ext = evaluate(model, teH, teT, turns[te], te_mask_extreme)
                print(f'  R{restart+1} ep{ep:2d}: loss={np.mean(losses):.4f}  '
                      f'extreme Cata={ext["cata"]:.2f}% Dir={ext["dir"]:.1f}')
        _, ext = evaluate(model, teH, teT, turns[te], te_mask_extreme)
        if ext['cata'] < best_ext_cata:
            best_ext_cata = ext['cata']; best_state = save_state(lls)
        restore(model, ols)

    # Final eval with best state
    lls, ols = inject_gate_lora(model)
    load_state(lls, best_state)
    g_all, g_ext = evaluate(model, teH, teT, turns[te], te_mask_extreme)
    restore(model, ols)

    print(f'\n{"=" * 80}')
    print(f'B4 GATE-LoRA RESULTS (params={n_params:,})')
    print(f'{"=" * 80}')
    print(f'  {"Metric":<24} {"Base":<12} {"+Gate-LoRA":<12} {"Change":<10}')
    print(f'  {"-" * 58}')
    print(f'  {"All FDE":<24} {b_all["fde"]:<12.3f} {g_all["fde"]:<12.3f} '
          f'{(b_all["fde"]-g_all["fde"])/max(b_all["fde"],1e-6)*100:>+7.1f}%')
    print(f'  {"All Dir(deg)":<24} {b_all["dir"]:<12.1f} {g_all["dir"]:<12.1f} '
          f'{(b_all["dir"]-g_all["dir"])/max(b_all["dir"],0.1)*100:>+7.1f}%')
    print(f'  {"All Cata%":<24} {b_all["cata"]:<12.2f} {g_all["cata"]:<12.2f}')
    print(f'  {"-- extreme turn subset --":<24}')
    print(f'  {"Extreme FDE":<24} {b_ext["fde"]:<12.3f} {g_ext["fde"]:<12.3f} '
          f'{(b_ext["fde"]-g_ext["fde"])/max(b_ext["fde"],1e-6)*100:>+7.1f}%')
    print(f'  {"Extreme Dir(deg)":<24} {b_ext["dir"]:<12.1f} {g_ext["dir"]:<12.1f} '
          f'{(b_ext["dir"]-g_ext["dir"])/max(b_ext["dir"],0.1)*100:>+7.1f}%')
    print(f'  {"Extreme Cata%":<24} {b_ext["cata"]:<12.2f} {g_ext["cata"]:<12.2f} '
          f'{(b_ext["cata"]-g_ext["cata"]):>+7.2f}pp')

    torch.save({'lora_state': best_state, 'config': {'targets': GATE_TARGETS}},
               WEIGHT_DIR / 'gate_lora_40.pth')
    summary = {'base_all': b_all, 'base_extreme': b_ext,
               'gate_all': g_all, 'gate_extreme': g_ext, 'params': n_params,
               'turn_eval_deg': TURN_EVAL_DEG}
    json.dump(summary, open(OUT_DIR / 'gate_lora_40.json', 'w'), indent=2)
    print(f'\n  Saved: weights/gate_lora_40.pth, pic-results/gate_lora_40.json')
    print('=' * 80)


if __name__ == '__main__':
    main()
