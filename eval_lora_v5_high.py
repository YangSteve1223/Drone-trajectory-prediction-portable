#!/usr/bin/env python3
"""
eval_lora_v5_high.py — v5 multi-target LoRA on HIGH (SimCruise) model, small-scale.

SimCruise windows are independent (each starts at [0,0]), no temporal continuity.
Strategy: sample N windows, train LoRA on 80%, test on 20%. Tests whether v5
multi-target + physics-aware LoRA generalizes to HIGH cruise-speed drones.
"""

import torch, numpy as np, sys, warnings, json, traceback
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from lora import LoRALinear

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = Path(__file__).parent.parent / 'SimCruise'
OUT_DIR = Path(__file__).parent / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
HIST_LEN, PRED_LEN = 20, 20
N_SAMPLE = 5000                   # sample windows
DIRERR_MAX = 90.0                 # skip catastrophically bad windows

# LoRA targets (same as v5)
LORA_TARGETS = [
    ('ua_pgd.feat_compress', 32),
    ('ua_pgd.neural_decoder.proj.0', 16),
    ('ua_pgd.neural_decoder.delta_head', 16),
]
HEAD_TARGETS = ['ua_pgd.anchor_to_pos.2']

# Training
EPOCHS = 20
LR_MAX, LR_MIN = 1e-3, 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8
RESTARTS = 3

# Loss weights (speed bound relaxed for 1Hz cruise data: 20m/s = 20m/step)
BETA_HUBER = 0.1
W_DIR = 0.15
W_SMOOTH = 0.03
W_SPEED = 0.02
W_ANCHOR_DIR = 0.05
SPEED_BOUND = 20.0               # 20m/s for 1Hz cruise (vs 3.0 for 5Hz LOW)


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def resolve_module(model, path):
    parts = path.split('.')
    obj = model
    for part in parts:
        obj = getattr(obj, part)
    return obj


def set_module(model, path, module):
    parts = path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def inject_lora(model, lora_targets, head_targets):
    for p in model.parameters():
        p.requires_grad_(False)
    lora_layers, original_layers = {}, {}
    for path, rank in lora_targets:
        original = resolve_module(model, path)
        original_layers[path] = original
        lora = LoRALinear(original, r=rank, alpha=rank * 2.0)
        set_module(model, path, lora)
        lora_layers[path] = lora
    head_layers, head_originals = {}, {}
    for path in head_targets:
        layer = resolve_module(model, path)
        head_originals[path] = {
            'weight': layer.weight.data.clone(),
            'bias': layer.bias.data.clone() if layer.bias is not None else None,
        }
        layer.weight.requires_grad_(True)
        if layer.bias is not None:
            layer.bias.requires_grad_(True)
        head_layers[path] = layer
    return lora_layers, head_layers, original_layers, head_originals


def trainable_params(lora_layers, head_layers):
    params = []
    for ll in lora_layers.values():
        params.extend([ll.lora_A, ll.lora_B])
    for layer in head_layers.values():
        if layer.weight.requires_grad:
            params.append(layer.weight)
        if layer.bias is not None and layer.bias.requires_grad:
            params.append(layer.bias)
    return params


def save_lora(lora_layers, head_layers):
    state = {
        'lora': {p: {'A': ll.lora_A.data.clone(), 'B': ll.lora_B.data.clone()}
                 for p, ll in lora_layers.items()},
        'head': {f'{p}.weight': l.weight.data.clone()
                 for p, l in head_layers.items()},
    }
    for p, l in head_layers.items():
        if l.bias is not None:
            state['head'][f'{p}.bias'] = l.bias.data.clone()
    return state


def load_lora(lora_layers, head_layers, state):
    for p, m in state['lora'].items():
        if p in lora_layers:
            lora_layers[p].lora_A.data.copy_(m['A'])
            lora_layers[p].lora_B.data.copy_(m['B'])
    for key, tensor in state['head'].items():
        p, attr = key.rsplit('.', 1)
        if p in head_layers:
            if attr == 'weight':
                head_layers[p].weight.data.copy_(tensor)
            elif attr == 'bias' and head_layers[p].bias is not None:
                head_layers[p].bias.data.copy_(tensor)


def restore_model(model, original_layers, head_originals):
    for path, original in original_layers.items():
        set_module(model, path, original)
    for path, orig in head_originals.items():
        layer = resolve_module(model, path)
        layer.weight.data.copy_(orig['weight'])
        layer.weight.requires_grad_(False)
        if orig['bias'] is not None:
            layer.bias.data.copy_(orig['bias'])
            layer.bias.requires_grad_(False)


def compute_loss(pred, target, base_pred):
    # Huber
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)
    # Direction
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    true_vel = target[:, 1:, :] - target[:, :-1, :]
    cos_sim = F.cosine_similarity(pred_vel, true_vel, dim=-1)
    loss_dir = (1.0 - cos_sim).mean()
    # Smoothness
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    loss_smooth = pred_acc.abs().mean()
    # Speed bound (20 m/s for 1Hz cruise)
    pred_speed = pred_vel.norm(dim=-1)
    loss_speed = F.relu(pred_speed - SPEED_BOUND).mean()
    # Base anchor direction
    base_dir = F.normalize(base_pred[:, -1, :2] - base_pred[:, 0, :2], dim=-1)
    pred_dir = F.normalize(pred[:, -1, :2] - pred[:, 0, :2], dim=-1)
    loss_anchor_dir = (1.0 - (base_dir * pred_dir).sum(dim=-1)).mean()
    return (loss_huber + W_DIR * loss_dir + W_SMOOTH * loss_smooth
            + W_SPEED * loss_speed + W_ANCHOR_DIR * loss_anchor_dir)


def main():
    print('=' * 80)
    print('LoRA v5 HIGH — Global LoRA on SimCruise Windows')
    print(f'  Sample: {N_SAMPLE} windows, train/test={TRAIN_SPLIT}')
    print(f'  Restarts={RESTARTS}  Epochs={EPOCHS}  LR={LR_MAX}→{LR_MIN}')
    print('=' * 80)

    # Load data
    print('\n[1/4] Loading SimCruise data...')
    train_files = sorted(DATA_ROOT.rglob('windows_train_chunk*.npz'))
    all_hist, all_pred = [], []
    for vf in train_files:
        d = np.load(vf)
        all_hist.append(d['hist'])
        all_pred.append(d['pred'])
    hist = np.concatenate(all_hist, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    # Sample
    idx = np.random.RandomState(42).choice(len(hist), min(N_SAMPLE, len(hist)), replace=False)
    hist, pred = hist[idx], pred[idx]
    print(f'  Sampled {len(hist)} windows from {len(train_files)} chunks')

    # Convert: target = displacement from last history point
    hist_t = torch.from_numpy(hist).float()
    pred_t = torch.from_numpy(pred).float()
    targets = pred_t - hist_t[:, -1, :3].unsqueeze(1)  # (N, 20, 3)

    # Load model
    print('\n[2/4] Loading HIGH model...')
    p = DronePredictor()
    model = p.high
    model.eval()
    device = p.device
    print(f'  Device: {device},  Base params: {sum(p.numel() for p in model.parameters()):,}')

    # Base model evaluation
    print('\n[3/4] Base model evaluation...')
    base_preds = []
    bs = 64
    for b in range(0, len(hist_t), bs):
        hb = hist_t[b:b + bs].to(device)
        with torch.no_grad():
            bp = model(hb, force_predict=True)['predictions'].cpu()
        base_preds.append(bp)
    base_preds = torch.cat(base_preds, dim=0)

    # Filter: skip catastrophic windows (DirErr > 90)
    bdir = np.array([
        dir_err(base_preds[i, -1, :2].numpy(), targets[i, -1, :2].numpy())
        for i in range(len(hist_t))
    ])
    good = bdir < DIRERR_MAX
    n_good = good.sum()
    print(f'  Good windows (DirErr<{DIRERR_MAX}): {n_good}/{len(hist_t)} '
          f'({n_good / len(hist_t) * 100:.1f}%)')

    if n_good < 50:
        print('  ERROR: Too few good windows!')
        return

    hist_t, targets, base_preds = hist_t[good], targets[good], base_preds[good]
    print(f'  Using {len(hist_t)} windows')

    # Train/test split
    n_tr = int(len(hist_t) * TRAIN_SPLIT)
    tr_h, tr_t, tr_bp = hist_t[:n_tr], targets[:n_tr], base_preds[:n_tr]
    te_h, te_t, te_bp = hist_t[n_tr:], targets[n_tr:], base_preds[n_tr:]

    # Training with restarts
    print(f'\n[4/4] Training LoRA ({n_tr} train, {len(te_h)} test)...')
    best_val_fde = float('inf')
    best_state = None

    for restart in range(RESTARTS):
        torch.manual_seed(42 + restart * 137)
        np.random.seed(42 + restart * 137)

        lora_layers, head_layers, original_layers, head_originals = \
            inject_lora(model, LORA_TARGETS, HEAD_TARGETS)
        params = trainable_params(lora_layers, head_layers)
        opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=LR_MIN)

        for ep in range(EPOCHS):
            model.eval()
            perm = np.random.permutation(n_tr)
            for b in range(0, n_tr, BATCH_SIZE):
                idx = perm[b:b + BATCH_SIZE]
                hb = tr_h[idx].to(device)
                tb = tr_t[idx].to(device)
                bb = tr_bp[idx].to(device)
                opt.zero_grad()
                preds = model(hb, force_predict=True)['predictions']
                loss = compute_loss(preds, tb, bb)
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    opt.step()
            scheduler.step()

        # Validation FDE
        model.eval()
        val_fdes = []
        for b in range(0, len(te_h), BATCH_SIZE):
            hb = te_h[b:b + BATCH_SIZE].to(device)
            tb = te_t[b:b + BATCH_SIZE]
            with torch.no_grad():
                preds = model(hb, force_predict=True)['predictions'].cpu()
            fde = torch.norm(preds[:, -1, :] - tb[:, -1, :], dim=-1)
            val_fdes.append(fde)
        val_fde = torch.cat(val_fdes).mean().item()

        print(f'  Restart {restart + 1}/{RESTARTS}: val FDE = {val_fde:.3f}m')
        if val_fde < best_val_fde:
            best_val_fde = val_fde
            best_state = save_lora(lora_layers, head_layers)

        restore_model(model, original_layers, head_originals)

    if best_state is None:
        print('  ERROR: No valid state!')
        return

    # Final evaluation
    lora_layers, head_layers, original_layers, head_originals = \
        inject_lora(model, LORA_TARGETS, HEAD_TARGETS)
    load_lora(lora_layers, head_layers, best_state)

    lora_preds = []
    for b in range(0, len(te_h), BATCH_SIZE):
        hb = te_h[b:b + BATCH_SIZE].to(device)
        with torch.no_grad():
            preds = model(hb, force_predict=True)['predictions'].cpu()
        lora_preds.append(preds)
    lora_preds = torch.cat(lora_preds, dim=0)

    restore_model(model, original_layers, head_originals)

    # Metrics
    b_ade = torch.norm(te_bp - te_t, dim=-1).mean(dim=1)  # per-window ADE
    b_fde = torch.norm(te_bp[:, -1, :] - te_t[:, -1, :], dim=-1)
    l_ade = torch.norm(lora_preds - te_t, dim=-1).mean(dim=1)
    l_fde = torch.norm(lora_preds[:, -1, :] - te_t[:, -1, :], dim=-1)

    b_dir = torch.tensor([dir_err(te_bp[i, -1, :2].numpy(), te_t[i, -1, :2].numpy())
                          for i in range(len(te_t))])
    l_dir = torch.tensor([dir_err(lora_preds[i, -1, :2].numpy(), te_t[i, -1, :2].numpy())
                          for i in range(len(te_t))])

    gain_ade = ((b_ade.mean() - l_ade.mean()) / b_ade.mean() * 100).item()
    gain_fde = ((b_fde.mean() - l_fde.mean()) / b_fde.mean() * 100).item()
    gain_dir = ((b_dir.mean() - l_dir.mean()) / max(b_dir.mean(), 0.1) * 100).item()

    n_fde_gain = (l_fde < b_fde).sum().item()
    n_dir_gain = (l_dir < b_dir).sum().item()
    n_test = len(te_t)

    # Summary
    print(f'\n{"=" * 80}')
    print(f'SUMMARY — LoRA v5 HIGH')
    print(f'  Test windows: {n_test}')
    print(f'  Base ADE: {b_ade.mean():.3f}m  |  LoRA ADE: {l_ade.mean():.3f}m  '
          f'({gain_ade:+.1f}%)')
    print(f'  Base FDE: {b_fde.mean():.3f}m  |  LoRA FDE: {l_fde.mean():.3f}m  '
          f'({gain_fde:+.1f}%)')
    print(f'  Base Dir: {b_dir.mean():.1f}°  |  LoRA Dir: {l_dir.mean():.1f}°  '
          f'({gain_dir:+.1f}%)')
    print(f'  FDE gain rate: {n_fde_gain}/{n_test} ({n_fde_gain / n_test * 100:.0f}%)')
    print(f'  Dir gain rate: {n_dir_gain}/{n_test} ({n_dir_gain / n_test * 100:.0f}%)')

    # Save JSON
    result = {
        'config': {
            'lora_targets': LORA_TARGETS,
            'head_targets': HEAD_TARGETS,
            'epochs': EPOCHS, 'restarts': RESTARTS,
            'n_sample': N_SAMPLE, 'n_test': n_test,
        },
        'base_ade': float(b_ade.mean()), 'lora_ade': float(l_ade.mean()),
        'base_fde': float(b_fde.mean()), 'lora_fde': float(l_fde.mean()),
        'base_dir': float(b_dir.mean()), 'lora_dir': float(l_dir.mean()),
        'ade_gain_pct': gain_ade, 'fde_gain_pct': gain_fde,
        'dir_gain_pct': gain_dir,
        'fde_gain_rate': f'{n_fde_gain}/{n_test}',
        'dir_gain_rate': f'{n_dir_gain}/{n_test}',
    }
    out_path = OUT_DIR / 'lora_v5_high.json'
    json.dump(result, open(out_path, 'w'), indent=2)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
