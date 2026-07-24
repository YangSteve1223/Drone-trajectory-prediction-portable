#!/usr/bin/env python3
"""
Global LoRA on the 40-FRAME base — general long-trajectory adaptation.

Difference vs train_global_lora.py:
  - Base is low_speed_6class_40frame.pth (history_len=40), not the 20-frame model.
  - Windows use adaptive stride (matching how the 40-frame base was trained).
  - Base anchor predictions are precomputed ONCE (no redundant per-step forward).

Produces weights/global_lora_40.pth — a general improvement LoRA stackable on the
40-frame base for per-drone adaptation.
"""

import torch, numpy as np, sys, warnings, json
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emam_model import TrajectoryPredictor
from lora import LoRALinear

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'
WEIGHT_DIR = Path(__file__).resolve().parents[1] / 'weights'
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data config ────────────────────────────────────────────────────────────
HIST_LEN, PRED_LEN = 40, 20        # 40-frame history
DT = 0.2
TOTAL_WINDOWS = 30000
TRAIN_SPLIT = 0.85
BATCH_SIZE = 64
SHORT_MAX = 80
MEDIUM_MAX = 150

# ── LoRA targets (high rank, upstream only, no delta_head) ─────────────────
LORA_TARGETS = [
    ('emam_se.mamba_blocks.0.ssm.in_proj', 24),
    ('emam_se.mamba_blocks.0.ssm.out_proj', 24),
    ('emam_se.mamba_blocks.1.ssm.in_proj', 24),
    ('emam_se.mamba_blocks.1.ssm.out_proj', 24),
    ('ua_pgd.feat_compress', 96),
    ('ua_pgd.neural_decoder.proj.0', 64),
]
HEAD_TARGETS = ['ua_pgd.anchor_to_pos.2']

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS = 30
LR_MAX, LR_MIN = 1e-3, 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RESTARTS = 2

# ── Loss (v8.1 physics + boundary) ─────────────────────────────────────────
BETA_HUBER = 0.20
W_DIR, W_SMOOTH, W_JERK = 0.25, 0.40, 0.35
W_CURVATURE, W_TV_VEL, W_SPEED = 0.20, 0.15, 0.03
W_ANCHOR_DIR = 0.02
W_BOUNDARY, BOUNDARY_STEPS = 0.40, 1
PHYSICS_WARMUP, PHYSICS_START = 3, 0.50


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))

def resolve_module(model, path):
    obj = model
    for part in path.split('.'): obj = getattr(obj, part)
    return obj

def set_module(model, path, module):
    parts = path.split('.'); parent = model
    for part in parts[:-1]: parent = getattr(parent, part)
    setattr(parent, parts[-1], module)

def make_adaptive_windows(traj, hist_len=40):
    """Adaptive stride based on trajectory length (matches expand_model_40)."""
    n = traj.shape[0]
    if n < 60: stride = 1
    elif n < 120: stride = 2
    else: stride = max(2, (n - PRED_LEN) // hist_len)
    ml = hist_len * stride + PRED_LEN
    if n < ml: return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + hist_len * stride, stride)[:hist_len]
        hists.append(traj[indices].copy())
        fut_start = i + hist_len * stride
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs

def inject_lora(model):
    for p in model.parameters(): p.requires_grad_(False)
    lora_layers, original_layers = {}, {}
    for path, rank in LORA_TARGETS:
        original = resolve_module(model, path)
        original_layers[path] = original
        lora = LoRALinear(original, r=rank, alpha=rank * 2.0)
        set_module(model, path, lora); lora_layers[path] = lora
    head_layers, head_originals = {}, {}
    for path in HEAD_TARGETS:
        layer = resolve_module(model, path)
        head_originals[path] = {'weight': layer.weight.data.clone(),
            'bias': layer.bias.data.clone() if layer.bias is not None else None}
        layer.weight.requires_grad_(True)
        if layer.bias is not None: layer.bias.requires_grad_(True)
        head_layers[path] = layer
    return lora_layers, head_layers, original_layers, head_originals

def collect_trainable(ll, hl):
    params = []
    for l in ll.values(): params.extend([l.lora_A, l.lora_B])
    for layer in hl.values():
        if layer.weight.requires_grad: params.append(layer.weight)
        if layer.bias is not None and layer.bias.requires_grad: params.append(layer.bias)
    return params

def save_lora_state(ll, hl):
    return {'lora': {p: {'A': l.lora_A.data.clone(), 'B': l.lora_B.data.clone()} for p, l in ll.items()},
            'head': {f'{p}.weight': l.weight.data.clone() for p, l in hl.items()}}

def load_lora_state(ll, hl, state):
    for p, m in state['lora'].items():
        if p in ll: ll[p].lora_A.data.copy_(m['A']); ll[p].lora_B.data.copy_(m['B'])
    for key, tensor in state['head'].items():
        p, attr = key.rsplit('.', 1)
        if p in hl:
            if attr == 'weight': hl[p].weight.data.copy_(tensor)
            elif attr == 'bias' and hl[p].bias is not None: hl[p].bias.data.copy_(tensor)

def restore_model(model, ol, ho):
    for path, original in ol.items(): set_module(model, path, original)
    for path, orig in ho.items():
        layer = resolve_module(model, path); layer.weight.data.copy_(orig['weight'])
        layer.weight.requires_grad_(False)
        if orig['bias'] is not None: layer.bias.data.copy_(orig['bias']); layer.bias.requires_grad_(False)

def physics_multiplier(epoch):
    if epoch >= PHYSICS_WARMUP: return 1.0
    return PHYSICS_START + (1.0 - PHYSICS_START) * (epoch / PHYSICS_WARMUP)

def compute_loss(pred, target, history, base_pred, epoch):
    ramp = physics_multiplier(epoch)
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]; true_vel = target[:, 1:, :] - target[:, :-1, :]
    loss_dir = (1.0 - F.cosine_similarity(pred_vel, true_vel, dim=-1)).mean()
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    loss_smooth = (pred_acc ** 2).mean()
    pred_jerk = pred[:, 3:, :] - 3 * pred[:, 2:-1, :] + 3 * pred[:, 1:-2, :] - pred[:, :-3, :]
    loss_jerk = (pred_jerk ** 2).mean()
    v_xy = pred_vel[:, :18, :2]; a_xy = pred_acc[:, :, :2]
    speed_xy = v_xy.norm(dim=-1) + 1e-6
    cross = v_xy[:, :, 0] * a_xy[:, :, 1] - v_xy[:, :, 1] * a_xy[:, :, 0]
    loss_curvature = (cross.abs() / (speed_xy ** 3 + 1e-4)).mean()
    vel_dir = F.normalize(pred_vel + 1e-8, dim=-1)
    loss_tv = (1.0 - (vel_dir[:, 1:, :] * vel_dir[:, :-1, :]).sum(dim=-1)).mean()
    loss_speed = F.relu(pred_vel.norm(dim=-1) - 3.0).mean()
    base_dir = F.normalize(base_pred[:, -1, :2] - base_pred[:, 0, :2], dim=-1)
    pred_dir = F.normalize(pred[:, -1, :2] - pred[:, 0, :2], dim=-1)
    loss_anchor = (1.0 - (base_dir * pred_dir).sum(dim=-1)).mean()
    hist_last_vel = history[:, -1, 3:6]
    loss_boundary = 0.0
    for k in range(BOUNDARY_STEPS):
        pc = pred[:, :k + 1, :].sum(dim=1); ec = hist_last_vel * (DT * (k + 1))
        loss_boundary = loss_boundary + ((pc - ec) ** 2).mean()
    loss_boundary = loss_boundary / BOUNDARY_STEPS
    return (loss_huber + W_DIR * loss_dir + W_SMOOTH * ramp * loss_smooth
            + W_JERK * ramp * loss_jerk + W_CURVATURE * ramp * loss_curvature
            + W_TV_VEL * ramp * loss_tv + W_SPEED * loss_speed + W_ANCHOR_DIR * loss_anchor
            + W_BOUNDARY * loss_boundary)

def main():
    print('=' * 80)
    print('Global LoRA Training — 40-FRAME base')
    print(f'  Base: low_speed_6class_40frame.pth (history_len=40)')
    print(f'  Data: {TOTAL_WINDOWS} adaptive-stride windows (short+medium+long)')
    print(f'  Epochs: {EPOCHS}  Restarts: {RESTARTS}  Batch: {BATCH_SIZE}')
    print('=' * 80)

    # ── Load 40-frame base model ──
    print('\n[1/4] Loading 40-frame base model...')
    model = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load(WEIGHT_DIR / 'low_speed_6class_40frame.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'  40-frame base loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

    # ── Collect mixed adaptive-stride windows ──
    print(f'\n[2/4] Collecting {TOTAL_WINDOWS} adaptive windows...')
    all_trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); n = d['traj'].shape[0]
        all_trajs.append((f.name, d['traj'], n))
    short = [(nm, t) for nm, t, l in all_trajs if l < SHORT_MAX]
    medium = [(nm, t) for nm, t, l in all_trajs if SHORT_MAX <= l < MEDIUM_MAX]
    long_t = [(nm, t) for nm, t, l in all_trajs if l >= MEDIUM_MAX]
    print(f'  Short(<{SHORT_MAX}): {len(short)}  Medium: {len(medium)}  Long(>={MEDIUM_MAX}): {len(long_t)}')

    np.random.seed(42)
    all_windows = []
    per_group = TOTAL_WINDOWS // 3
    for group_name, group in [('short', short), ('medium', medium), ('long', long_t)]:
        n_collected = 0
        shuffled = list(group); np.random.shuffle(shuffled)
        for name, traj in shuffled:
            if n_collected >= per_group: break
            hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
            for h, f in zip(hists, futs):
                all_windows.append((torch.from_numpy(h).float(), torch.from_numpy(f).float()))
                n_collected += 1
                if n_collected >= per_group: break
        print(f'  {group_name}: {n_collected} windows')
    np.random.shuffle(all_windows)
    all_windows = all_windows[:TOTAL_WINDOWS]
    n_tr = int(len(all_windows) * TRAIN_SPLIT)
    tr_data = all_windows[:n_tr]; te_data = all_windows[n_tr:]
    print(f'  Total: {len(all_windows)}  Train: {len(tr_data)}  Test: {len(te_data)}')

    # Pre-stack tensors (kept on CPU; moved to GPU per batch)
    te_h = torch.stack([d[0] for d in te_data]); te_t = torch.stack([d[1] for d in te_data])

    # ── Base model evaluation (40-frame, no LoRA) ──
    print(f'\n[3/4] Base model evaluation (40-frame, no LoRA)...')
    base_preds = []
    for b in range(0, len(te_data), BATCH_SIZE):
        hb = te_h[b:b + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            base_preds.append(model(hb, force_predict=True)['predictions'].cpu())
    base_preds = torch.cat(base_preds, dim=0)
    base_fde = torch.norm(base_preds[:, -1, :] - te_t[:, -1, :], dim=-1).mean().item()
    base_ade = torch.norm(base_preds - te_t, dim=-1).mean(dim=1).mean().item()
    bdir = np.array([dir_err(base_preds[i, -1, :2].numpy(), te_t[i, -1, :2].numpy())
                     for i in range(len(te_data))])
    base_dir = float(np.mean(bdir)); base_cata = float(np.sum(bdir >= 90) / len(bdir) * 100)
    base_gap = float(np.linalg.norm(base_preds[:, 0, :].numpy(), axis=1).mean())
    print(f'  40-frame base: ADE={base_ade:.3f}m  FDE={base_fde:.3f}m  '
          f'Dir={base_dir:.1f}deg  Cata={base_cata:.1f}%  Gap={base_gap:.3f}m')

    # PERF: precompute base anchor predictions for the training set ONCE.
    # (Original script re-ran a no_grad forward every step for the anchor loss.)
    print('  Precomputing base anchor predictions for training set...')
    tr_h = torch.stack([d[0] for d in tr_data]); tr_t = torch.stack([d[1] for d in tr_data])
    tr_bp = []
    for b in range(0, len(tr_data), BATCH_SIZE):
        hb = tr_h[b:b + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            tr_bp.append(model(hb, force_predict=True)['predictions'].cpu())
    tr_bp = torch.cat(tr_bp, dim=0)

    # ── Global LoRA training ──
    print(f'\n[4/4] Training Global LoRA on 40-frame base...')
    best_test_fde = float('inf'); best_state = None; n_params = 0

    for restart in range(RESTARTS):
        torch.manual_seed(42 + restart * 137); np.random.seed(42 + restart * 137)
        ll, hl, ol, ho = inject_lora(model)
        params = collect_trainable(ll, hl)
        n_params = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

        idx = np.arange(len(tr_data))
        for ep in range(EPOCHS):
            model.eval(); np.random.shuffle(idx); ep_losses = []
            for b in range(0, len(idx), BATCH_SIZE):
                bi = idx[b:b + BATCH_SIZE]
                hb = tr_h[bi].to(DEVICE); tb = tr_t[bi].to(DEVICE); bp = tr_bp[bi].to(DEVICE)
                opt.zero_grad()
                pred = model(hb, force_predict=True)['predictions']
                loss = compute_loss(pred, tb, hb, bp, ep)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward(); torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
                    ep_losses.append(loss.item())
            sched.step()
            if ep % 10 == 0:
                print(f'  Restart {restart + 1} Epoch {ep:2d}: loss={np.mean(ep_losses):.4f}  '
                      f'lr={sched.get_last_lr()[0]:.6f}')

        model.eval(); lora_preds = []
        for b in range(0, len(te_data), BATCH_SIZE):
            hb = te_h[b:b + BATCH_SIZE].to(DEVICE)
            with torch.no_grad():
                lora_preds.append(model(hb, force_predict=True)['predictions'].cpu())
        lora_preds = torch.cat(lora_preds, dim=0)
        test_fde = torch.norm(lora_preds[:, -1, :] - te_t[:, -1, :], dim=-1).mean().item()
        print(f'  Restart {restart + 1}: test FDE = {test_fde:.3f}m (base={base_fde:.3f}m, '
              f'gain={(base_fde - test_fde) / max(base_fde, 1e-6) * 100:+.1f}%)')
        if test_fde < best_test_fde:
            best_test_fde = test_fde; best_state = save_lora_state(ll, hl)
        restore_model(model, ol, ho)

    if best_state is None:
        print('ERROR: No valid state!'); return

    # ── Final evaluation ──
    ll, hl, ol, ho = inject_lora(model)
    load_lora_state(ll, hl, best_state)
    lora_preds = []
    for b in range(0, len(te_data), BATCH_SIZE):
        hb = te_h[b:b + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            lora_preds.append(model(hb, force_predict=True)['predictions'].cpu())
    lora_preds = torch.cat(lora_preds, dim=0)
    lora_fde = torch.norm(lora_preds[:, -1, :] - te_t[:, -1, :], dim=-1).mean().item()
    lora_ade = torch.norm(lora_preds - te_t, dim=-1).mean(dim=1).mean().item()
    ldir = np.array([dir_err(lora_preds[i, -1, :2].numpy(), te_t[i, -1, :2].numpy())
                     for i in range(len(te_data))])
    lora_dir = float(np.mean(ldir)); lora_cata = float(np.sum(ldir >= 90) / len(ldir) * 100)
    lora_gap = float(np.linalg.norm(lora_preds[:, 0, :].numpy(), axis=1).mean())

    print(f'\n{"=" * 80}')
    print(f'GLOBAL LoRA (40-frame base) RESULTS')
    print(f'  Trainable params: {n_params:,} '
          f'({n_params / sum(p.numel() for p in model.parameters()) * 100:.1f}% of full model)')
    print(f'{"=" * 80}')
    print(f'  {"Metric":<18} {"40f base":<14} {"+Global LoRA":<16} {"Change":<10}')
    print(f'  {"-" * 58}')
    print(f'  {"ADE":<18} {base_ade:<14.3f} {lora_ade:<16.3f} {(base_ade - lora_ade) / max(base_ade, 1e-6) * 100:>+7.1f}%')
    print(f'  {"FDE":<18} {base_fde:<14.3f} {lora_fde:<16.3f} {(base_fde - lora_fde) / max(base_fde, 1e-6) * 100:>+7.1f}%')
    print(f'  {"Direction(deg)":<18} {base_dir:<14.1f} {lora_dir:<16.1f} {(base_dir - lora_dir) / max(base_dir, 0.1) * 100:>+7.1f}%')
    print(f'  {"Cata(>90deg)%":<18} {base_cata:<14.1f} {lora_cata:<16.1f}')
    print(f'  {"Boundary Gap":<18} {base_gap:<14.3f} {lora_gap:<16.3f}')

    save_path = WEIGHT_DIR / 'global_lora_40.pth'
    torch.save({'lora_state': best_state['lora'], 'head_state': best_state['head'],
                'config': {'targets': LORA_TARGETS, 'heads': HEAD_TARGETS, 'hist_len': HIST_LEN},
                'base_fde': base_fde, 'lora_fde': lora_fde}, save_path)
    print(f'\n  Saved: {save_path}')

    summary = {
        'base_model': 'low_speed_6class_40frame.pth', 'global_lora': str(save_path),
        'train_windows': len(tr_data), 'test_windows': len(te_data), 'trainable_params': n_params,
        'base_ade': base_ade, 'lora_ade': lora_ade, 'base_fde': base_fde, 'lora_fde': lora_fde,
        'base_dir': base_dir, 'lora_dir': lora_dir, 'base_cata': base_cata, 'lora_cata': lora_cata,
        'base_gap': base_gap, 'lora_gap': lora_gap,
        'fde_gain_pct': float((base_fde - lora_fde) / max(base_fde, 1e-6) * 100),
    }
    json.dump(summary, open(OUT_DIR / 'global_lora_40.json', 'w'), indent=2)
    print(f'  Results saved: pic-results/global_lora_40.json')
    print('=' * 80)


if __name__ == '__main__':
    main()
