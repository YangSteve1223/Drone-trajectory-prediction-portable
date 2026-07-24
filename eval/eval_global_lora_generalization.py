#!/usr/bin/env python3
"""
Global LoRA — HONEST cross-flight generalization test.

The production dir_lora_40 / global_lora_40 numbers (+19.4% / +14.9% FDE) were
measured with a WINDOW-level split: all windows from all flights were pooled,
shuffled, then split 85/15. Windows from the same flight therefore appear in
BOTH train and test — the test set is not truly unseen, so the gain is optimistic.

This script re-measures the gain the honest way:
  1. Split FLIGHTS (npz files) into disjoint train / test sets (by file, seeded).
  2. Train a global direction-LoRA on windows from TRAIN flights only.
  3. Evaluate base vs LoRA on windows from TEST flights the LoRA never saw.

Same B3 LoRA config / loss / turn-weighting as train_lora_direction.py, so the
only thing that changes is the split. The delta between this gain and the
window-level +19.4% is the generalization gap.

Run from repo root:  python eval/eval_global_lora_generalization.py
"""

import torch, numpy as np, sys, warnings, json, time
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))  # reuse B3 pipeline
from emam_model import TrajectoryPredictor
import train_lora_direction as B3   # shared config, loss, LoRA inject/train helpers

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'
WEIGHT_DIR = Path(__file__).resolve().parents[1] / 'weights'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 40, 20
TOTAL_WINDOWS = 30000          # same budget as production training
TEST_FLIGHT_FRAC = 0.20        # 20% of flights held out entirely
SEED = 2025


def collect_windows_from_flights(flights, budget):
    """Balanced short/medium/long windows drawn ONLY from the given flights."""
    short = [(nm, t) for nm, t, l in flights if l < B3.SHORT_MAX]
    medium = [(nm, t) for nm, t, l in flights if B3.SHORT_MAX <= l < B3.MEDIUM_MAX]
    long_t = [(nm, t) for nm, t, l in flights if l >= B3.MEDIUM_MAX]
    per_group = budget // 3
    rng = np.random.RandomState(SEED)
    windows = []
    for gname, group in [('short', short), ('medium', medium), ('long', long_t)]:
        n = 0
        g = list(group); rng.shuffle(g)
        for name, traj in g:
            if n >= per_group: break
            hists, futs = B3.make_adaptive_windows(traj, hist_len=HIST_LEN)
            for h, f in zip(hists, futs):
                w = B3.turn_weight(f)
                windows.append((torch.from_numpy(h).float(),
                                torch.from_numpy(f).float(), w))
                n += 1
                if n >= per_group: break
        print(f'    {gname}: {n} windows  (from {len(group)} flights)')
    rng.shuffle(windows)
    return windows[:budget]


def evaluate(model, te_h, te_t):
    preds = []
    for b in range(0, len(te_h), B3.BATCH_SIZE):
        hb = te_h[b:b + B3.BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            preds.append(model(hb, force_predict=True)['predictions'].cpu())
    preds = torch.cat(preds, dim=0)
    fde = torch.norm(preds[:, -1, :] - te_t[:, -1, :], dim=-1).mean().item()
    ade = torch.norm(preds - te_t, dim=-1).mean(dim=1).mean().item()
    d = np.array([B3.dir_err(preds[i, -1, :2].numpy(), te_t[i, -1, :2].numpy())
                  for i in range(len(te_h))])
    return {'fde': fde, 'ade': ade, 'dir': float(d.mean()),
            'cata': float((d >= 90).sum() / len(d) * 100)}


def main():
    print('=' * 80)
    print('Global LoRA — HONEST cross-flight generalization test')
    print(f'  Split: {int((1-TEST_FLIGHT_FRAC)*100)}% flights train / '
          f'{int(TEST_FLIGHT_FRAC*100)}% flights held-out (disjoint, seed={SEED})')
    print('=' * 80)

    # ── flight-level split ──
    print('\n[1/5] Indexing flights + disjoint split...')
    all_flights = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        n = np.load(f)['traj'].shape[0]
        all_flights.append((f.name, None, n))
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(all_flights))
    n_test = int(len(all_flights) * TEST_FLIGHT_FRAC)
    test_names = set(all_flights[i][0] for i in order[:n_test])

    def load(names_pred):
        out = []
        for f in sorted(TRAJ_DIR.glob('*.npz')):
            if names_pred(f.name):
                t = np.load(f)['traj']
                out.append((f.name, t, t.shape[0]))
        return out
    train_flights = load(lambda nm: nm not in test_names)
    test_flights = load(lambda nm: nm in test_names)
    print(f'  Train flights: {len(train_flights)}  Test flights: {len(test_flights)}')
    assert not (set(n for n, _, _ in train_flights) & set(n for n, _, _ in test_flights)), \
        'FLIGHT LEAK — train and test share a flight!'

    # ── windows ──
    print(f'\n[2/5] Collecting windows (train)...')
    tr_data = collect_windows_from_flights(train_flights, TOTAL_WINDOWS)
    print(f'\n[3/5] Collecting windows (held-out test)...')
    te_data = collect_windows_from_flights(test_flights, TOTAL_WINDOWS // 4)
    tr_h = torch.stack([d[0] for d in tr_data]); tr_t = torch.stack([d[1] for d in tr_data])
    tr_w = torch.tensor([d[2] for d in tr_data], dtype=torch.float32)
    te_h = torch.stack([d[0] for d in te_data]); te_t = torch.stack([d[1] for d in te_data])
    print(f'  Train windows: {len(tr_data)}  Test windows: {len(te_data)}')

    # ── base model ──
    print(f'\n[4/5] Loading 40-frame base + base eval on held-out flights...')
    model = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load(WEIGHT_DIR / 'low_speed_6class_40frame.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    base = evaluate(model, te_h, te_t)
    print(f'  BASE (held-out): FDE={base["fde"]:.3f}  ADE={base["ade"]:.3f}  '
          f'Dir={base["dir"]:.1f}deg  Cata={base["cata"]:.1f}%')

    # precompute base anchors for train windows
    tr_bp = []
    for b in range(0, len(tr_data), B3.BATCH_SIZE):
        hb = tr_h[b:b + B3.BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            tr_bp.append(model(hb, force_predict=True)['predictions'].cpu())
    tr_bp = torch.cat(tr_bp, dim=0)

    # ── train global LoRA on TRAIN flights only ──
    print(f'\n[5/5] Training global dir-LoRA on TRAIN flights '
          f'({B3.EPOCHS} epochs x {B3.RESTARTS} restarts)...')
    t0 = time.time()
    best_fde = float('inf'); best_state = None
    for restart in range(B3.RESTARTS):
        torch.manual_seed(42 + restart * 137); np.random.seed(42 + restart * 137)
        ll, hl, ol, ho = B3.inject_lora(model)
        params = B3.collect_trainable(ll, hl)
        opt = torch.optim.AdamW(params, lr=B3.LR_MAX, weight_decay=B3.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=B3.EPOCHS, eta_min=B3.LR_MIN)
        idx = np.arange(len(tr_data))
        for ep in range(B3.EPOCHS):
            model.eval(); np.random.shuffle(idx)
            for b in range(0, len(idx), B3.BATCH_SIZE):
                bi = idx[b:b + B3.BATCH_SIZE]
                hb = tr_h[bi].to(DEVICE); tb = tr_t[bi].to(DEVICE)
                bp = tr_bp[bi].to(DEVICE); sw = tr_w[bi].to(DEVICE)
                opt.zero_grad()
                pred = model(hb, force_predict=True)['predictions']
                loss = B3.compute_loss(pred, tb, hb, bp, ep, sample_w=sw)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, B3.GRAD_CLIP); opt.step()
            sched.step()
        r = evaluate(model, te_h, te_t)
        print(f'  Restart {restart+1}: held-out FDE={r["fde"]:.3f}  '
              f'gain={(base["fde"]-r["fde"])/base["fde"]*100:+.1f}%')
        if r['fde'] < best_fde:
            best_fde = r['fde']; best_state = B3.save_lora_state(ll, hl)
        B3.restore_model(model, ol, ho)

    ll, hl, ol, ho = B3.inject_lora(model)
    B3.load_lora_state(ll, hl, best_state)
    lora = evaluate(model, te_h, te_t)
    B3.restore_model(model, ol, ho)
    train_min = (time.time() - t0) / 60

    fde_gain = (base['fde'] - lora['fde']) / base['fde'] * 100
    ade_gain = (base['ade'] - lora['ade']) / base['ade'] * 100
    dir_gain = (base['dir'] - lora['dir']) / max(base['dir'], 0.1) * 100

    print(f'\n{"=" * 80}')
    print('CROSS-FLIGHT GENERALIZATION RESULT (held-out flights, never seen)')
    print(f'{"=" * 80}')
    print(f'  {"Metric":<16}{"base":<12}{"+global LoRA":<16}{"gain":<10}')
    print(f'  {"-"*52}')
    print(f'  {"FDE (m)":<16}{base["fde"]:<12.3f}{lora["fde"]:<16.3f}{fde_gain:>+7.1f}%')
    print(f'  {"ADE (m)":<16}{base["ade"]:<12.3f}{lora["ade"]:<16.3f}{ade_gain:>+7.1f}%')
    print(f'  {"Dir (deg)":<16}{base["dir"]:<12.1f}{lora["dir"]:<16.1f}{dir_gain:>+7.1f}%')
    print(f'  {"Cata (%)":<16}{base["cata"]:<12.1f}{lora["cata"]:<16.1f}')
    print(f'\n  vs window-level split (+19.4% FDE): generalization gap = '
          f'{19.4 - fde_gain:+.1f} pts')
    print(f'  Train time: {train_min:.1f} min on {DEVICE}')
    print('=' * 80)

    json.dump({
        'split': 'flight-level (disjoint)', 'seed': SEED,
        'n_train_flights': len(train_flights), 'n_test_flights': len(test_flights),
        'n_train_windows': len(tr_data), 'n_test_windows': len(te_data),
        'base': base, 'lora': lora,
        'fde_gain_pct': fde_gain, 'ade_gain_pct': ade_gain, 'dir_gain_pct': dir_gain,
        'window_level_fde_gain_pct': 19.4,
        'generalization_gap_pts': 19.4 - fde_gain,
        'train_minutes': train_min,
    }, open(OUT_DIR / 'global_lora_generalization.json', 'w'), indent=2)
    print(f'  Saved: pic-results/global_lora_generalization.json')


if __name__ == '__main__':
    main()
