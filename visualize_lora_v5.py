#!/usr/bin/env python3
"""
LoRA visualization v5 — Multi-target high-rank adaptation with physics-aware loss.

Key improvements over v1-v4:
  1. Multi-target LoRA: feat_compress(r=32) + proj.0(r=16) + delta_head(r=16)
     + anchor_to_pos.2 (full finetune). Total ~14.5K params covering the full
     autoregressive decoder pipeline.
  2. Physics-aware loss: Huber + direction consistency + acceleration smoothness
     + speed bound + base-model direction anchor. Balances point accuracy with
     trajectory-level physical realism.
  3. Maximum context: stride=1 on >=150-frame trajectories (~160 windows each).
  4. Robust optimization: 20 epochs, cosine LR (1e-3→1e-5), AdamW (wd=1e-4),
     3 independent restarts (pick best on validation FDE).
  5. Long trajectories only: >=150 frames, pre-filtered by base model direction error.
"""

import torch, numpy as np, sys, warnings, traceback
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from lora import LoRALinear

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
OUT_DIR = Path(__file__).parent / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config ──────────────────────────────────────────────────────────
HIST_LEN, PRED_LEN = 20, 20
STRIDE = 1                    # maximum context: every possible window
MIN_FRAMES = 150              # only very long trajectories
DIRERR_MAX = 60.0             # trainable window direction error threshold
MIN_TRAINABLE = 30            # minimum trainable windows per trajectory
N_CANDIDATES = 50             # process top 50 longest that pass filters

# ── LoRA multi-target config ──────────────────────────────────────────────
# (path, rank) — each layer gets its own rank based on its role in the decoder
LORA_TARGETS = [
    ('ua_pgd.feat_compress', 32),                # (128,128) r=32: hidden state init
    ('ua_pgd.neural_decoder.proj.0', 16),         # (128,128) r=16: feature projection
    ('ua_pgd.neural_decoder.delta_head', 16),     # (128,3)   r=16: per-step output
]
HEAD_TARGETS = [
    'ua_pgd.anchor_to_pos.2',  # (64,3) full finetune (~195 params)
]

# ── Training config ───────────────────────────────────────────────────────
EPOCHS = 20
LR_MAX = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8             # 80/20 train/val
RESTARTS = 3                  # independent random restarts, pick best

# ── Loss weights ──────────────────────────────────────────────────────────
BETA_HUBER = 0.1              # Huber transition point (normalized displacement)
W_DIR = 0.15                  # direction cosine loss
W_SMOOTH = 0.03               # acceleration smoothness (L1)
W_SPEED = 0.02                # speed upper bound penalty
W_ANCHOR_DIR = 0.05           # base model direction anchor

# ── Plot config ───────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'font.family': 'sans-serif'})
C = {'hist': '#2196F3', 'base': '#F44336', 'lora': '#FF9800', 'truth': '#4CAF50'}
TIME_MARKS = {0: '0.0s', 5: '1.0s', 10: '2.0s', 15: '3.0s', 19: '4.0s'}
MK_C = ['#E91E63', '#9C27B0', '#00BCD4', '#FFEB3B', '#FF5722']
MK_S = ['o', 's', '^', 'D', 'P']


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def sliding_windows(traj):
    """Extract all (history, future_displacement) windows with STRIDE=1."""
    n = traj.shape[0]
    ml = HIST_LEN + PRED_LEN
    if n < ml:
        return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, STRIDE):
        hists.append(traj[i:i + HIST_LEN].copy())
        fut_abs = traj[i + HIST_LEN:i + HIST_LEN + PRED_LEN, :3]
        futs.append(fut_abs - traj[i + HIST_LEN - 1, :3])
    return hists, futs


def dir_err(pv, tv):
    """Direction error in degrees between two 2D/3D vectors."""
    pn = float(np.linalg.norm(pv))
    tn = float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def resolve_module(model, path):
    """Resolve dotted path to a submodule (handles Sequential integer indices)."""
    parts = path.split('.')
    obj = model
    for part in parts:
        obj = getattr(obj, part)
    return obj


def set_module(model, path, module):
    """Replace a submodule at the given dotted path."""
    parts = path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def inject_lora(model, lora_targets, head_targets):
    """
    Inject LoRA wrappers into target layers. Freezes entire model first.
    Returns (lora_layers, head_layers, original_layers, head_originals)
    for later restoration.
    """
    # Freeze entire model
    for p in model.parameters():
        p.requires_grad_(False)

    # Inject LoRA layers
    lora_layers = {}
    original_layers = {}
    for path, rank in lora_targets:
        original = resolve_module(model, path)
        if not isinstance(original, torch.nn.Linear):
            raise TypeError(f'{path} is not nn.Linear, got {type(original)}')
        original_layers[path] = original
        lora = LoRALinear(original, r=rank, alpha=rank * 2.0)
        set_module(model, path, lora)
        lora_layers[path] = lora

    # Enable head param training
    head_layers = {}
    head_originals = {}
    for path in head_targets:
        layer = resolve_module(model, path)
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f'{path} is not nn.Linear, got {type(layer)}')
        head_originals[path] = {
            'weight': layer.weight.data.clone(),
            'bias': layer.bias.data.clone() if layer.bias is not None else None,
        }
        layer.weight.requires_grad_(True)
        if layer.bias is not None:
            layer.bias.requires_grad_(True)
        head_layers[path] = layer

    return lora_layers, head_layers, original_layers, head_originals


def collect_trainable(lora_layers, head_layers):
    """Collect all trainable parameters."""
    params = []
    for ll in lora_layers.values():
        params.extend([ll.lora_A, ll.lora_B])
    for layer in head_layers.values():
        if layer.weight.requires_grad:
            params.append(layer.weight)
        if layer.bias is not None and layer.bias.requires_grad:
            params.append(layer.bias)
    return params


def save_lora_state(lora_layers, head_layers):
    """Export LoRA A,B matrices and head weights."""
    state = {
        'lora': {path: {'A': ll.lora_A.data.clone(), 'B': ll.lora_B.data.clone()}
                 for path, ll in lora_layers.items()},
        'head': {f'{path}.weight': layer.weight.data.clone()
                 for path, layer in head_layers.items()},
    }
    for path, layer in head_layers.items():
        if layer.bias is not None:
            state['head'][f'{path}.bias'] = layer.bias.data.clone()
    return state


def load_lora_state(lora_layers, head_layers, state):
    """Load LoRA A,B matrices and head weights from saved state."""
    for path, matrices in state['lora'].items():
        if path in lora_layers:
            lora_layers[path].lora_A.data.copy_(matrices['A'])
            lora_layers[path].lora_B.data.copy_(matrices['B'])
    for key, tensor in state['head'].items():
        path, attr = key.rsplit('.', 1)
        if path in head_layers:
            if attr == 'weight':
                head_layers[path].weight.data.copy_(tensor)
            elif attr == 'bias' and head_layers[path].bias is not None:
                head_layers[path].bias.data.copy_(tensor)


def reset_lora(lora_layers, head_layers, head_originals):
    """Reset all LoRA and head parameters to initial state."""
    for ll in lora_layers.values():
        ll.reset_lora_parameters()
    for path, orig in head_originals.items():
        head_layers[path].weight.data.copy_(orig['weight'])
        if orig['bias'] is not None:
            head_layers[path].bias.data.copy_(orig['bias'])


def restore_model(model, original_layers, head_originals):
    """Restore original Linear layers and unfreeze model."""
    for path, original in original_layers.items():
        set_module(model, path, original)
    for path, orig in head_originals.items():
        layer = resolve_module(model, path)
        layer.weight.data.copy_(orig['weight'])
        layer.weight.requires_grad_(False)
        if orig['bias'] is not None:
            layer.bias.data.copy_(orig['bias'])
            layer.bias.requires_grad_(False)


# ═══════════════════════════════════════════════════════════════════════════
# Physics-aware loss
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(pred, target, base_pred):
    """
    Multi-component physics-aware loss.

    pred, target, base_pred: (B, 20, 3) in meters (denormalized)

    Returns (total_loss, loss_dict).
    """
    # 1. Huber loss — robust position error
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)

    # 2. Direction consistency — per-step velocity direction
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]      # (B, 19, 3)
    true_vel = target[:, 1:, :] - target[:, :-1, :]   # (B, 19, 3)
    cos_sim = F.cosine_similarity(pred_vel, true_vel, dim=-1)  # (B, 19)
    loss_dir = (1.0 - cos_sim).mean()

    # 3. Acceleration smoothness — L1 on 2nd derivative (more robust than L2)
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]  # (B, 18, 3)
    loss_smooth = pred_acc.abs().mean()

    # 4. Speed bound — soft penalty for unrealistic per-step speeds
    pred_speed = pred_vel.norm(dim=-1)  # (B, 19), meters per 0.2s step
    loss_speed = F.relu(pred_speed - 3.0).mean()  # >3m/step = >15 m/s

    # 5. Base-model direction anchor — prevent LoRA from going completely off-track
    # Compare XY direction of the overall displacement vector
    base_dir = F.normalize(base_pred[:, -1, :2] - base_pred[:, 0, :2], dim=-1)
    pred_dir = F.normalize(pred[:, -1, :2] - pred[:, 0, :2], dim=-1)
    loss_anchor_dir = (1.0 - (base_dir * pred_dir).sum(dim=-1)).mean()

    total = (loss_huber
             + W_DIR * loss_dir
             + W_SMOOTH * loss_smooth
             + W_SPEED * loss_speed
             + W_ANCHOR_DIR * loss_anchor_dir)

    return total, {
        'huber': loss_huber.item(),
        'dir': loss_dir.item(),
        'smooth': loss_smooth.item(),
        'speed': loss_speed.item(),
        'anchor_dir': loss_anchor_dir.item(),
        'total': total.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-trajectory processing
# ═══════════════════════════════════════════════════════════════════════════

def process_v5(name, traj, model, device):
    """Full per-trajectory LoRA experiment: eval → train (3 restarts) → eval."""
    hists, futs = sliding_windows(traj)
    n_total = len(hists)
    if n_total < MIN_TRAINABLE + 20:
        return None

    # ── Phase 1: Base model evaluation on ALL windows (batched) ──────
    all_h = torch.stack([torch.from_numpy(h).float() for h in hists])
    all_t = [torch.from_numpy(t).float() for t in futs]

    bpred = []
    bs_eval = 64
    for b in range(0, n_total, bs_eval):
        hb = all_h[b:b + bs_eval].to(device)
        with torch.no_grad():
            preds = model(hb, force_predict=True)['predictions'].cpu()
        bpred.append(preds)
    bpred = torch.cat(bpred, dim=0)  # (n_total, 20, 3)

    # Compute per-window metrics
    bade_np = torch.norm(bpred - torch.stack(all_t), dim=-1).mean(dim=1).numpy()  # ADE per window
    bfde_np = torch.norm(bpred[:, -1, :] - torch.stack([t[-1] for t in all_t]), dim=-1).numpy()  # FDE
    bdir_np = np.array([
        dir_err(bpred[i, -1, :2].numpy(), all_t[i][-1, :2].numpy())
        for i in range(n_total)
    ])

    bade, bfde, bdir = bade_np, bfde_np, bdir_np
    # Keep list of tensors for later use (bpred)
    bpred_list = [bpred[i] for i in range(n_total)]

    bade, bfde, bdir = np.array(bade), np.array(bfde), np.array(bdir)
    trainable = bdir < DIRERR_MAX
    n_trainable = int(trainable.sum())
    n_cata = int((bdir >= 90).sum())

    if n_trainable < MIN_TRAINABLE:
        return None

    # Split train/val/test
    tidx = np.where(trainable)[0]
    np.random.seed(42)
    np.random.shuffle(tidx)
    n_tr = int(len(tidx) * TRAIN_SPLIT)
    tr_idx = tidx[:n_tr]
    val_idx = tidx[n_tr:n_tr + max(5, len(tidx) // 5)]
    te_idx = tidx[n_tr + len(val_idx):] if n_tr + len(val_idx) < len(tidx) else tidx[n_tr:n_tr + 10]

    if len(tr_idx) < 10 or len(te_idx) < 5:
        return None

    # Convert to tensors
    tr_h = torch.stack([torch.from_numpy(hists[i]).float() for i in tr_idx])
    tr_t = torch.stack([torch.from_numpy(futs[i]).float() for i in tr_idx])
    tr_bp = torch.stack([bpred_list[i] for i in tr_idx])
    val_h = torch.stack([torch.from_numpy(hists[i]).float() for i in val_idx])
    val_t = torch.stack([torch.from_numpy(futs[i]).float() for i in val_idx])

    # ── Phase 2: Multi-restart LoRA training ──────────────────────────
    best_val_fde = float('inf')
    best_state = None

    for restart in range(RESTARTS):
        seed = 42 + restart * 137
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Inject LoRA
        lora_layers, head_layers, original_layers, head_originals = \
            inject_lora(model, LORA_TARGETS, HEAD_TARGETS)
        trainable_params = collect_trainable(lora_layers, head_layers)

        opt = torch.optim.AdamW(trainable_params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=LR_MIN)

        # Warmup: first 2 epochs at LR_MAX/10
        warmup_epochs = 2
        warmup_factor = 1.0 / warmup_epochs

        bs = min(BATCH_SIZE, len(tr_idx))

        for ep in range(EPOCHS):
            model.eval()  # keep dropout/batchnorm in eval mode
            perm = np.random.permutation(len(tr_idx))

            # LR warmup
            if ep < warmup_epochs:
                for pg in opt.param_groups:
                    pg['lr'] = LR_MIN + (LR_MAX - LR_MIN) * (ep + 1) / warmup_epochs * warmup_factor

            ep_losses = []
            for b in range(0, len(tr_idx), bs):
                idx = perm[b:b + bs]
                hb, tb, bb = tr_h[idx].to(device), tr_t[idx].to(device), tr_bp[idx].to(device)

                opt.zero_grad()
                out = model(hb, force_predict=True)
                pred = out['predictions']

                loss, loss_dict = compute_loss(pred, tb, bb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                opt.step()
                ep_losses.append(loss_dict)

            if ep >= warmup_epochs:
                scheduler.step()

        # Validation FDE
        model.eval()
        val_fdes = []
        for b in range(0, len(val_idx), bs):
            hb = val_h[b:b + bs].to(device)
            tb = val_t[b:b + bs]
            with torch.no_grad():
                pred = model(hb, force_predict=True)['predictions'].cpu()
            fde_batch = torch.norm(pred[:, -1, :] - tb[:, -1, :], dim=-1)
            val_fdes.append(fde_batch)
        val_fde = torch.cat(val_fdes).mean().item()

        if val_fde < best_val_fde:
            best_val_fde = val_fde
            best_state = save_lora_state(lora_layers, head_layers)

        # Restore model
        restore_model(model, original_layers, head_originals)

    if best_state is None:
        return None

    # ── Phase 3: Final evaluation with best checkpoint ────────────────
    lora_layers, head_layers, original_layers, head_originals = \
        inject_lora(model, LORA_TARGETS, HEAD_TARGETS)
    load_lora_state(lora_layers, head_layers, best_state)

    lp_test = []
    model.eval()
    for i in te_idx:
        h = torch.from_numpy(hists[i]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(h, force_predict=True)['predictions'][0].cpu()
        lp_test.append(pred)

    # Restore
    restore_model(model, original_layers, head_originals)

    # Compute metrics
    la = [np.linalg.norm(lp.numpy() - futs[i], axis=1).mean() for i, lp in zip(te_idx, lp_test)]
    lf = [np.linalg.norm(lp.numpy()[-1] - futs[i][-1]) for i, lp in zip(te_idx, lp_test)]

    test_b_ade = float(bade[te_idx].mean())
    test_b_fde = float(bfde[te_idx].mean())
    test_l_ade = float(np.mean(la))
    test_l_fde = float(np.mean(lf))

    # Also compute direction metrics
    base_dir_errors = [dir_err(bpred_list[i].numpy()[-1, :2], futs[i][-1, :2]) for i in te_idx]
    lora_dir_errors = [dir_err(lp_test[j].numpy()[-1, :2], futs[te_idx[j]][-1, :2]) for j in range(len(te_idx))]
    test_b_dir = float(np.mean(base_dir_errors))
    test_l_dir = float(np.mean(lora_dir_errors))

    return {
        'name': name,
        'n_frames': traj.shape[0],
        'n_total': n_total,
        'n_trainable': n_trainable,
        'n_cata': n_cata,
        'n_train': len(tr_idx),
        'n_val': len(val_idx),
        'n_test': len(te_idx),
        'base_ade': test_b_ade,
        'base_fde': test_b_fde,
        'base_dir': test_b_dir,
        'lora_ade': test_l_ade,
        'lora_fde': test_l_fde,
        'lora_dir': test_l_dir,
        'ade_gain': float((test_b_ade - test_l_ade) / test_b_ade * 100),
        'fde_gain': float((test_b_fde - test_l_fde) / test_b_fde * 100),
        'dir_gain': float((test_b_dir - test_l_dir) / max(test_b_dir, 0.1) * 100),
        'te_idx': te_idx,
        'hists': hists,
        'futs': futs,
        'bpred': bpred_list,
        'lp_test': lp_test,
        'bdir_all': bdir,
        'bade_all': bade,
        'bfde_all': bfde,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Candidate collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_candidates():
    """Collect longest trajectories >= MIN_FRAMES. Pre-filtering is done in process_v5."""
    candidates = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        n = d['traj'].shape[0]
        if n >= MIN_FRAMES:
            candidates.append((f.name, d['traj'], n))

    candidates.sort(key=lambda x: x[2], reverse=True)  # longest first
    return candidates[:N_CANDIDATES]


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d(ax, hist, bp, lp, tp, title):
    """3D trajectory: History, Base, LoRA, Truth with time marks."""
    last = hist[-1, :3]
    hp = hist[:, :3]
    bp_a = bp + last
    lp_a = lp + last
    tp_a = tp + last
    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=C['hist'], lw=2, label='History')
    ax.plot(bp_a[:, 0], bp_a[:, 1], bp_a[:, 2], color=C['base'], lw=1.5,
            ls='--', alpha=0.7, label='Base')
    ax.plot(lp_a[:, 0], lp_a[:, 1], lp_a[:, 2], color=C['lora'], lw=2, label='LoRA')
    ax.plot(tp_a[:, 0], tp_a[:, 1], tp_a[:, 2], color=C['truth'], lw=2, label='Truth')
    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.scatter(lp_a[fidx, 0], lp_a[fidx, 1], lp_a[fidx, 2], c=MK_C[fi], s=50,
                   marker=MK_S[fi], edgecolors='black', lw=0.5, zorder=10)
        ax.scatter(tp_a[fidx, 0], tp_a[fidx, 1], tp_a[fidx, 2], c=MK_C[fi], s=40,
                   marker=MK_S[fi], edgecolors='black', lw=0.5, zorder=10, alpha=0.7)
    all_xy = np.concatenate([bp_a[:, :2], lp_a[:, :2], tp_a[:, :2], hp[:, :2]])
    rng = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.6
    zm = (bp_a[:, 2].mean() + tp_a[:, 2].mean()) / 2
    ax.set_zlim(zm - rng * 0.5, zm + rng * 0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.legend(fontsize=6)
    ax.view_init(elev=25, azim=-60)


def plot_xy(ax, hist, bp, lp, tp, title):
    """XY top-down view with time marks."""
    last = hist[-1, :2]
    hp = hist[:, :2]
    bp_a = bp[:, :2] + last
    lp_a = lp[:, :2] + last
    tp_a = tp[:, :2] + last
    ax.plot(hp[:, 0], hp[:, 1], color=C['hist'], lw=2, label='History')
    ax.plot(bp_a[:, 0], bp_a[:, 1], color=C['base'], lw=1.5, ls='--', alpha=0.7,
            label='Base')
    ax.plot(lp_a[:, 0], lp_a[:, 1], color=C['lora'], lw=2, label='LoRA')
    ax.plot(tp_a[:, 0], tp_a[:, 1], color=C['truth'], lw=2, label='Truth')
    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.scatter(lp_a[fidx, 0], lp_a[fidx, 1], c=MK_C[fi], s=60,
                   marker=MK_S[fi], edgecolors='black', lw=0.5, zorder=10)
        ax.scatter(tp_a[fidx, 0], tp_a[fidx, 1], c=MK_C[fi], s=50,
                   marker=MK_S[fi], edgecolors='black', lw=0.5, zorder=10, alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)


def plot_err(ax, bp, lp, tp, title):
    """Per-step error comparison: Base vs LoRA."""
    be = np.linalg.norm(bp - tp, axis=1)
    le = np.linalg.norm(lp - tp, axis=1)
    steps = np.arange(len(be)) * 0.2
    ax.plot(steps, be, 'o-', color=C['base'], lw=1.5, ms=3, label='Base')
    ax.plot(steps, le, 's-', color=C['lora'], lw=1.5, ms=3, label='LoRA')
    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.axvline(x=fidx * 0.2, color=MK_C[fi], ls=':', alpha=0.4, lw=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('LoRA v5 — Multi-Target High-Rank Autoregressive Decoder Correction')
    print(f'  LoRA targets:')
    for path, rank in LORA_TARGETS:
        print(f'    {path}  r={rank}')
    print(f'  Head targets: {HEAD_TARGETS}')
    print(f'  Loss: Huber(beta={BETA_HUBER}) + {W_DIR}*dir + {W_SMOOTH}*smooth '
          f'+ {W_SPEED}*speed + {W_ANCHOR_DIR}*anchor_dir')
    print(f'  Restarts={RESTARTS}  Epochs={EPOCHS}  Stride={STRIDE}  '
          f'LR={LR_MAX}→{LR_MIN}  wd={WEIGHT_DECAY}')
    print(f'  Min frames={MIN_FRAMES}  DirErr<{DIRERR_MAX}  '
          f'Min trainable={MIN_TRAINABLE}')
    print('=' * 80)

    # Load model
    p = DronePredictor()
    model = p.low
    model.eval()
    device = p.device
    print(f'\nDevice: {device}')
    print(f'Base params: {sum(p.numel() for p in model.parameters()):,}')

    # Collect candidates
    candidates = collect_candidates()
    print(f'\nCandidates >= {MIN_FRAMES} frames: {len(candidates)}')
    for i, (name, traj, nf) in enumerate(candidates[:5]):
        print(f'  {i + 1}. {name[:30]}  frames={nf}')

    if not candidates:
        print('No candidates found! Check MIN_FRAMES or DIRERR_MAX.')
        return

    # Process each candidate
    results = []
    total_cata, total_win = 0, 0
    for ti, (name, traj, nf) in enumerate(candidates):
        try:
            r = process_v5(name, traj, model, device)
            if r:
                results.append(r)
                total_cata += r['n_cata']
                total_win += r['n_total']
                g = 'GAIN' if r['fde_gain'] > 0 else 'DEGRADE'
                print(f'  [{ti + 1:2d}] {name[:25]}  f={nf}  '
                      f'trn={r["n_train"]}/{r["n_total"]}  cata={r["n_cata"]}  '
                      f'BFDE={r["base_fde"]:.3f}  LFDE={r["lora_fde"]:.3f}  '
                      f'({r["fde_gain"]:+.1f}%) [{g}]  '
                      f'dir: {r["base_dir"]:.1f}→{r["lora_dir"]:.1f}°')
        except Exception as e:
            print(f'  [{ti + 1:2d}] {name[:25]}  ERROR: {e}')
            traceback.print_exc()

    # Summary
    n_gain = sum(1 for r in results if r['fde_gain'] > 0)
    n_dir_improve = sum(1 for r in results if r['dir_gain'] > 0)
    avg_fde_gain = np.mean([r['fde_gain'] for r in results]) if results else 0
    avg_dir_gain = np.mean([r['dir_gain'] for r in results]) if results else 0
    print(f'\n{"=" * 80}')
    print(f'SUMMARY: {len(results)} trajectories processed')
    print(f'  FDE gain rate: {n_gain}/{len(results)} ({n_gain / max(len(results), 1) * 100:.0f}%)  '
          f'avg: {avg_fde_gain:+.1f}%')
    print(f'  Dir gain rate: {n_dir_improve}/{len(results)} '
          f'({n_dir_improve / max(len(results), 1) * 100:.0f}%)  avg: {avg_dir_gain:+.1f}%')
    print(f'  Catastrophic: {total_cata}/{total_win} ({total_cata / max(total_win, 1) * 100:.1f}%)')

    if not results:
        return

    # Sort and select display trajectories
    results.sort(key=lambda x: x['fde_gain'], reverse=True)
    disp = [r for r in results[:16] if r['fde_gain'] > 0][:10]
    if len(disp) < 6:
        disp = results[:10]

    def pick_window(r):
        """Pick best test window: largest FDE improvement with DirErr < 90."""
        best_j, best_g = 0, -999.0
        for j in range(len(r['te_idx'])):
            i = r['te_idx'][j]
            if r['bdir_all'][i] >= 90:
                continue
            bf = np.linalg.norm(r['bpred'][i].numpy()[-1] - r['futs'][i][-1])
            lf = np.linalg.norm(r['lp_test'][j].numpy()[-1] - r['futs'][i][-1])
            if bf - lf > best_g:
                best_g = bf - lf
                best_j = j
        return best_j

    nd = len(disp)
    nr = (nd + 3) // 4

    print(f'\nDisplaying {nd}/{len(results)} trajectories:')
    for i, r in enumerate(disp):
        print(f'  {i + 1}. {r["name"][:30]}  FDE: {r["base_fde"]:.3f}→{r["lora_fde"]:.3f}m  '
              f'({r["fde_gain"]:+.1f}%)  Dir: {r["base_dir"]:.1f}→{r["lora_dir"]:.1f}°')

    print('\nGenerating figures...')

    # ── 3D overview ──
    fig = plt.figure(figsize=(24, 6 * nr))
    fig.suptitle(
        f'LoRA v5 — Multi-Target Autoregressive Decoder Correction\n'
        f'Targets: feat_compress(r=32) + proj.0(r=16) + delta_head(r=16) + anchor_to_pos.2(full)\n'
        f'Loss: Huber + dir + smooth + speed + anchor_dir.  '
        f'Gain rate: {n_gain}/{len(results)} ({n_gain / max(len(results), 1) * 100:.0f}%).  '
        f'Avg FDE gain: {avg_fde_gain:+.1f}%',
        fontsize=12, fontweight='bold')
    for i, r in enumerate(disp):
        ax = fig.add_subplot(nr, min(4, nd), i + 1, projection='3d')
        j = pick_window(r)
        idx = r['te_idx'][j]
        plot_3d(ax, r['hists'][idx], r['bpred'][idx].numpy(),
                r['lp_test'][j].numpy(), r['futs'][idx],
                f'{r["name"][:18]}\nFDE: {r["base_fde"]:.3f}→{r["lora_fde"]:.3f}m '
                f'({r["fde_gain"]:+.1f}%)')
    plt.tight_layout(pad=2)
    p3d = OUT_DIR / 'lora_v5_3d.png'
    fig.savefig(p3d, dpi=150, bbox_inches='tight')
    print(f'  Saved: {p3d}')
    plt.close()

    # ── XY overview ──
    fig, axes = plt.subplots(nr, min(4, nd), figsize=(24, 6 * nr))
    if nd == 1:
        axes = np.array([[axes]])
    fig.suptitle(f'LoRA v5 — XY View.  Avg FDE gain: {avg_fde_gain:+.1f}%',
                 fontsize=12, fontweight='bold')
    for i, r in enumerate(disp):
        ax = axes.flat[i] if nd > 1 else axes[0, 0]
        j = pick_window(r)
        idx = r['te_idx'][j]
        plot_xy(ax, r['hists'][idx], r['bpred'][idx].numpy(),
                r['lp_test'][j].numpy(), r['futs'][idx],
                f'{r["name"][:18]}\nFDE: {r["base_fde"]:.3f}→{r["lora_fde"]:.3f}m '
                f'({r["fde_gain"]:+.1f}%)')
    for i in range(nd, nr * min(4, nd)):
        if nd > 1:
            axes.flat[i].axis('off')
    plt.tight_layout(pad=2)
    pxy = OUT_DIR / 'lora_v5_xy.png'
    fig.savefig(pxy, dpi=150, bbox_inches='tight')
    print(f'  Saved: {pxy}')
    plt.close()

    # ── Detailed pages (top 6) ──
    det = disp[:6]
    for pi in range(0, len(det), 2):
        pg = det[pi:pi + 2]
        ns = len(pg)
        fig = plt.figure(figsize=(24, 8 * ns))
        fig.suptitle(
            'LoRA v5 Detailed — Multi-Target + Physics-Aware Loss\n'
            'Blue=History  Red=Base  Orange=LoRA  Green=Truth  '
            'Color marks = 0.0/1.0/2.0/3.0/4.0s',
            fontsize=12, fontweight='bold')
        for ri, r in enumerate(pg):
            j = pick_window(r)
            idx = r['te_idx'][j]
            bp_np = r['bpred'][idx].numpy()
            lp_np = r['lp_test'][j].numpy()
            tp_np = r['futs'][idx]
            ax3 = fig.add_subplot(ns, 3, ri * 3 + 1, projection='3d')
            plot_3d(ax3, r['hists'][idx], bp_np, lp_np, tp_np,
                    f'{r["name"][:18]}\nFDE: {r["base_fde"]:.3f}→{r["lora_fde"]:.3f}m '
                    f'({r["fde_gain"]:+.1f}%)')
            ax_xy = fig.add_subplot(ns, 3, ri * 3 + 2)
            plot_xy(ax_xy, r['hists'][idx], bp_np, lp_np, tp_np,
                    f'XY: {r["name"][:18]}')
            ax_e = fig.add_subplot(ns, 3, ri * 3 + 3)
            plot_err(ax_e, bp_np, lp_np, tp_np,
                     f'Per-Step Error: {r["name"][:18]}')
        plt.tight_layout(pad=2)
        pd = OUT_DIR / f'lora_v5_detail_p{pi // 2 + 1}.png'
        fig.savefig(pd, dpi=150, bbox_inches='tight')
        print(f'  Saved: {pd}')
        plt.close()

    # ── Statistics table ──
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.axis('off')
    lines = [
        f'LoRA v5 — Multi-Target High-Rank Autoregressive Decoder Correction',
        f'Targets: feat_compress(r=32) + proj.0(r=16) + delta_head(r=16) + anchor_to_pos.2(full)',
        f'Loss: Huber(beta={BETA_HUBER}) + {W_DIR}*dir_cos + {W_SMOOTH}*smooth_L1 + '
        f'{W_SPEED}*speed_bound + {W_ANCHOR_DIR}*anchor_dir',
        f'Restarts={RESTARTS}  Epochs={EPOCHS}  LR={LR_MAX}→{LR_MIN}  wd={WEIGHT_DECAY}  '
        f'Stride={STRIDE}  MinFrames={MIN_FRAMES}  DirErr<{DIRERR_MAX}',
        '',
        f'Results: {len(results)} trajectories, FDE gain {n_gain}/{len(results)} '
        f'({n_gain / max(len(results), 1) * 100:.0f}%), '
        f'Avg FDE gain {avg_fde_gain:+.1f}%, Avg Dir gain {avg_dir_gain:+.1f}%',
        f'Catastrophic windows: {total_cata}/{total_win} '
        f'({total_cata / max(total_win, 1) * 100:.1f}%)',
        '',
        f'{"Trajectory":<32} {"Fr":<5} {"Tr/Vl/Te":<12} {"Cata":<6} '
        f'{"B.FDE":<8} {"L.FDE":<8} {"FDEG":<8} {"B.Dir":<7} {"L.Dir":<7} {"DirG":<8}',
        '-' * 110,
    ]
    for r in results[:30]:
        lines.append(
            f'{r["name"]:<32} {r["n_frames"]:<5} '
            f'{r["n_train"]}/{r["n_val"]}/{r["n_test"]:<12} '
            f'{r["n_cata"]:<6} '
            f'{r["base_fde"]:<8.3f} {r["lora_fde"]:<8.3f} {r["fde_gain"]:>+6.1f}%  '
            f'{r["base_dir"]:<7.1f} {r["lora_dir"]:<7.1f} {r["dir_gain"]:>+6.1f}%')
    text = '\n'.join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=6.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    pt = OUT_DIR / 'lora_v5_table.png'
    fig.savefig(pt, dpi=150, bbox_inches='tight')
    print(f'  Saved: {pt}')
    plt.close()

    print(f'\nDone! Files: {OUT_DIR}/lora_v5_*')


if __name__ == '__main__':
    main()
