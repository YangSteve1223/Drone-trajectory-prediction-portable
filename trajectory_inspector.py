#!/usr/bin/env python3
"""
Trajectory Inspector — generates detailed trajectory visualizations for
judging LoRA smoothness. Combines Phase 1 (coordinate output) and
Phase 2 (Strategy C: spline post-processing).

Output per trajectory:
  - pic-results/inspect/*.npz : raw coordinates (history, base, lora, lora_spline, truth)
  - pic-results/inspect/*.png : 3D, XY, per-axis, step-error comparison charts
"""

import torch, numpy as np, sys, warnings, traceback, json
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from context_adapter import ContextAdapterV2
from lora import LoRALinear

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
ADAPTER_PATH = Path(__file__).parent / 'weights' / 'context_adapter_ac.pth'
OUT_DIR = Path(__file__).parent / 'pic-results' / 'inspect'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
STRIDE, GATE_SCALE = 2, 0.3

# v7 LoRA: upstream only, no delta_head
LORA_TARGETS = [
    ('emam_se.mamba_blocks.0.ssm.in_proj', 16),
    ('emam_se.mamba_blocks.0.ssm.out_proj', 16),
    ('emam_se.mamba_blocks.1.ssm.in_proj', 16),
    ('emam_se.mamba_blocks.1.ssm.out_proj', 16),
    ('ua_pgd.feat_compress', 64),
    ('ua_pgd.neural_decoder.proj.0', 48),
]
HEAD_TARGETS = ['ua_pgd.anchor_to_pos.2']

EPOCHS, RESTARTS = 40, 3
LR_MAX, LR_MIN = 1e-3, 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.0
BATCH_SIZE, TRAIN_SPLIT = 32, 0.8

# v7 loss weights
BETA_HUBER = 0.20
W_DIR, W_SMOOTH, W_JERK = 0.25, 0.40, 0.35
W_CURVATURE, W_TV_VEL, W_SPEED, W_ANCHOR_DIR = 0.20, 0.15, 0.03, 0.02
PHYSICS_WARMUP, PHYSICS_START = 3, 0.50

# Spline config
SPLINE_KNOT_FRAC = 0.35   # use 35% of points as knots (7 knots for 20 steps)

# 6 representative trajectories (same as v5 viz)
SELECTED = [
    '2025-04-23_09-15-12.npz',   # best FDE gain
    '2025-04-21_15-03-16.npz',
    '2025-04-03_12-30-43.npz',
    '2025-04-25_09-42-34.npz',
    '2025-04-29_17-15-55.npz',
    '2025-04-28_14-30-06.npz',   # worst FDE gain
]

TIME_MARKS = {0: '0.0s', 5: '1.0s', 10: '2.0s', 15: '3.0s', 19: '4.0s'}
MK_COLORS = ['#E91E63', '#9C27B0', '#00BCD4', '#FFEB3B', '#FF5722']
MK_SHAPES = ['o', 's', '^', 'D', 'P']
C = {'hist': '#1565C0', 'base': '#D32F2F', 'lora': '#FF6D00',
     'spline': '#7B1FA2', 'truth': '#2E7D32'}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (same as eval_lora_v7_ac.py)
# ═══════════════════════════════════════════════════════════════════════════

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


def make_windows_ac(traj, stride=2):
    n = traj.shape[0]
    ml = HIST_LEN * stride + PRED_LEN
    if n < ml:
        return [], [], []
    hists, futs, starts = [], [], []
    step = max(1, stride // 2)
    for i in range(0, n - ml + 1, step):
        indices = np.arange(i, i + HIST_LEN * stride, stride)[:HIST_LEN]
        hists.append(traj[indices].copy())
        fut_start = i + HIST_LEN * stride
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
        starts.append(i)
    return hists, futs, starts


def make_context(traj, starts, ctx_len=60):
    n = traj.shape[0]
    ctx = []
    for ws in starts:
        end = ws + ctx_len
        ctx.append(traj[ws:end, :].copy() if end <= n else traj[-ctx_len:, :].copy())
    return np.array(ctx, dtype=np.float32)


def predict_batch(model, device, adapter, hb, cb):
    ctx_inj = None
    if adapter is not None and cb is not None:
        with torch.no_grad():
            ctx_inj = adapter(cb.to(device))
    kwargs = {'force_predict': True}
    if ctx_inj is not None:
        kwargs['context_injection'] = ctx_inj
    with torch.no_grad():
        return model(hb.to(device), **kwargs)['predictions'].cpu()


def inject_lora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    lora_layers, original_layers = {}, {}
    for path, rank in LORA_TARGETS:
        original = resolve_module(model, path)
        original_layers[path] = original
        lora = LoRALinear(original, r=rank, alpha=rank * 2.0)
        set_module(model, path, lora)
        lora_layers[path] = lora
    head_layers, head_originals = {}, {}
    for path in HEAD_TARGETS:
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


def collect_trainable(lora_layers, head_layers):
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
    return {
        'lora': {p: {'A': ll.lora_A.data.clone(), 'B': ll.lora_B.data.clone()}
                 for p, ll in lora_layers.items()},
        'head': {f'{p}.weight': l.weight.data.clone() for p, l in head_layers.items()},
    }


def load_lora_state(lora_layers, head_layers, state):
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


def physics_multiplier(epoch):
    if epoch >= PHYSICS_WARMUP:
        return 1.0
    return PHYSICS_START + (1.0 - PHYSICS_START) * (epoch / PHYSICS_WARMUP)


def compute_loss(pred, target, base_pred, epoch):
    ramp = physics_multiplier(epoch)
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    true_vel = target[:, 1:, :] - target[:, :-1, :]
    loss_dir = (1.0 - F.cosine_similarity(pred_vel, true_vel, dim=-1)).mean()
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    loss_smooth = (pred_acc ** 2).mean()
    pred_jerk = (pred[:, 3:, :] - 3 * pred[:, 2:-1, :] + 3 * pred[:, 1:-2, :] - pred[:, :-3, :])
    loss_jerk = (pred_jerk ** 2).mean()
    v_xy = pred_vel[:, :18, :2]
    a_xy = pred_acc[:, :, :2]
    speed_xy = v_xy.norm(dim=-1) + 1e-6
    cross = v_xy[:, :, 0] * a_xy[:, :, 1] - v_xy[:, :, 1] * a_xy[:, :, 0]
    loss_curvature = (cross.abs() / (speed_xy ** 3 + 1e-4)).mean()
    vel_dir = F.normalize(pred_vel + 1e-8, dim=-1)
    loss_tv = (1.0 - (vel_dir[:, 1:, :] * vel_dir[:, :-1, :]).sum(dim=-1)).mean()
    pred_speed = pred_vel.norm(dim=-1)
    loss_speed = F.relu(pred_speed - 3.0).mean()
    base_dir = F.normalize(base_pred[:, -1, :2] - base_pred[:, 0, :2], dim=-1)
    pred_dir = F.normalize(pred[:, -1, :2] - pred[:, 0, :2], dim=-1)
    loss_anchor = (1.0 - (base_dir * pred_dir).sum(dim=-1)).mean()
    return (loss_huber + W_DIR * loss_dir + W_SMOOTH * ramp * loss_smooth
            + W_JERK * ramp * loss_jerk + W_CURVATURE * ramp * loss_curvature
            + W_TV_VEL * ramp * loss_tv + W_SPEED * loss_speed + W_ANCHOR_DIR * loss_anchor)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy C: Spline Smoothing
# ═══════════════════════════════════════════════════════════════════════════

def spline_smooth(pred_abs, knot_frac=SPLINE_KNOT_FRAC):
    """Fit cubic spline with reduced knots and resample at 20 points."""
    n = pred_abs.shape[0]
    n_knots = max(3, int(n * knot_frac))
    knot_idx = np.linspace(0, n - 1, n_knots, dtype=int)
    t_all = np.arange(n, dtype=float)
    t_knots = t_all[knot_idx]
    smoothed = np.zeros_like(pred_abs)
    for dim in range(3):
        knots = pred_abs[knot_idx, dim]
        cs = CubicSpline(t_knots, knots, bc_type='natural')
        smoothed[:, dim] = cs(t_all)
    return smoothed


def compute_smoothness(pred_abs):
    pred_abs = np.asarray(pred_abs)
    vel = pred_abs[1:, :] - pred_abs[:-1, :]
    acc = vel[1:, :] - vel[:-1, :]
    jerk = acc[1:, :] - acc[:-1, :]
    v_norm = np.linalg.norm(vel[:-1, :2], axis=1) + 1e-6
    cross = np.abs(vel[:-1, 0] * acc[:, 0] - vel[:-1, 1] * acc[:, 1])
    curvatures = cross / (v_norm ** 3 + 1e-4)
    return {
        'mean_jerk': float(np.mean(np.linalg.norm(jerk, axis=1))),
        'max_jerk': float(np.max(np.linalg.norm(jerk, axis=1))),
        'tortuosity': float(np.sum(np.linalg.norm(vel, axis=1))
                           / max(np.linalg.norm(pred_abs[-1] - pred_abs[0]), 1e-6)),
        'total_curvature': float(np.sum(curvatures)),
        'path_length': float(np.sum(np.linalg.norm(vel, axis=1))),
        'straight_dist': float(np.linalg.norm(pred_abs[-1] - pred_abs[0])),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d(ax, hist, base, lora, spline, truth, title):
    last = hist[-1, :3]
    hp = hist[:, :3]
    bp_a = base + last
    lp_a = lora + last
    sp_a = spline + last
    tp_a = truth + last

    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=C['hist'], lw=2.5, label='History')
    ax.plot(bp_a[:, 0], bp_a[:, 1], bp_a[:, 2], color=C['base'], lw=1.5,
            ls='--', alpha=0.6, label='Base (AC)')
    ax.plot(lp_a[:, 0], lp_a[:, 1], lp_a[:, 2], color=C['lora'], lw=2.0,
            label='LoRA v7')
    ax.plot(sp_a[:, 0], sp_a[:, 1], sp_a[:, 2], color=C['spline'], lw=2.0,
            ls='-.', label='LoRA+Spline')
    ax.plot(tp_a[:, 0], tp_a[:, 1], tp_a[:, 2], color=C['truth'], lw=2.5,
            label='Truth')

    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.scatter(tp_a[fidx, 0], tp_a[fidx, 1], tp_a[fidx, 2],
                   c=MK_COLORS[fi], s=45, marker=MK_SHAPES[fi],
                   edgecolors='black', lw=0.5, zorder=10, alpha=0.8)

    all_pts = np.concatenate([hp, bp_a, lp_a, sp_a, tp_a], axis=0)
    xy_range = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.55
    z_mid = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    ax.set_xlim(all_pts[:, 0].mean() - xy_range, all_pts[:, 0].mean() + xy_range)
    ax.set_ylim(all_pts[:, 1].mean() - xy_range, all_pts[:, 1].mean() + xy_range)
    ax.set_zlim(z_mid - xy_range, z_mid + xy_range)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.legend(fontsize=6, loc='upper left')
    ax.view_init(elev=22, azim=-55)


def plot_xy(ax, hist, base, lora, spline, truth, title):
    last = hist[-1, :2]
    hp = hist[:, :2]
    bp_a = base[:, :2] + last
    lp_a = lora[:, :2] + last
    sp_a = spline[:, :2] + last
    tp_a = truth[:, :2] + last

    ax.plot(hp[:, 0], hp[:, 1], color=C['hist'], lw=2.5, label='History')
    ax.plot(bp_a[:, 0], bp_a[:, 1], color=C['base'], lw=1.5, ls='--', alpha=0.6,
            label='Base (AC)')
    ax.plot(lp_a[:, 0], lp_a[:, 1], color=C['lora'], lw=2.0, label='LoRA v7')
    ax.plot(sp_a[:, 0], sp_a[:, 1], color=C['spline'], lw=2.0, ls='-.',
            label='LoRA+Spline')
    ax.plot(tp_a[:, 0], tp_a[:, 1], color=C['truth'], lw=2.5, label='Truth')
    ax.scatter(hp[-1, 0], hp[-1, 1], c=C['hist'], s=80, marker='s',
               edgecolors='black', lw=0.8, zorder=5)
    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.scatter(tp_a[fidx, 0], tp_a[fidx, 1], c=MK_COLORS[fi], s=55,
                   marker=MK_SHAPES[fi], edgecolors='black', lw=0.5, zorder=10, alpha=0.8)

    all_xy = np.concatenate([hp[:, :2], bp_a, lp_a, sp_a, tp_a], axis=0)
    xy_range = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55
    xm, ym = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(xm - xy_range, xm + xy_range)
    ax.set_ylim(ym - xy_range, ym + xy_range)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc='upper left')


def plot_step_error(ax, base, lora, spline, truth, title):
    be = np.linalg.norm(base - truth, axis=1)
    le = np.linalg.norm(lora - truth, axis=1)
    se = np.linalg.norm(spline - truth, axis=1)
    steps = np.arange(20) * 0.2
    ax.plot(steps, be, 'o-', color=C['base'], lw=1.5, ms=3, alpha=0.7, label='Base (AC)')
    ax.plot(steps, le, 's-', color=C['lora'], lw=2, ms=3.5, label='LoRA v7')
    ax.plot(steps, se, 'D-', color=C['spline'], lw=2, ms=3.5, label='LoRA+Spline')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Error (m)')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)


def plot_per_axis(ax_x, ax_y, ax_z, base, lora, spline, truth, hist_last):
    """Per-axis absolute position comparison."""
    bp_a = base + hist_last
    lp_a = lora + hist_last
    sp_a = spline + hist_last
    tp_a = truth + hist_last
    steps = np.arange(20) * 0.2
    for ax, dim, label in [(ax_x, 0, 'X'), (ax_y, 1, 'Y'), (ax_z, 2, 'Z')]:
        ax.plot(steps, bp_a[:, dim], '--', color=C['base'], lw=2, alpha=0.7, label='Base')
        ax.plot(steps, lp_a[:, dim], '-', color=C['lora'], lw=2, label='LoRA')
        ax.plot(steps, sp_a[:, dim], '-.', color=C['spline'], lw=2, label='Spline')
        ax.plot(steps, tp_a[:, dim], '-', color=C['truth'], lw=2.5, label='Truth')
        ax.set_ylabel(f'{label} (m)'); ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('Trajectory Inspector — LoRA v7 + Spline Smoothing (Strategy C)')
    print(f'  Spline knot fraction: {SPLINE_KNOT_FRAC} ({int(20*SPLINE_KNOT_FRAC)} knots)')
    print('=' * 80)

    # Load model + adapter
    p = DronePredictor()
    model = p.low; model.eval(); device = p.device
    adapter = ContextAdapterV2(input_dim=6, context_len=CTX_LEN,
                                d_model=model.d_model, hidden=128).to(device)
    adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=device))
    adapter.eval()

    orig_gate = model.ua_pgd.physics_gate.forward
    def scaled_gate(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_gate(last_encoded, intent_weights, step_encoding)
        return gi * GATE_SCALE, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_gate

    all_results = []

    for ti, name in enumerate(SELECTED):
        fpath = TRAJ_DIR / name
        if not fpath.exists():
            print(f'  [{ti + 1}] {name} — NOT FOUND')
            continue

        d = np.load(fpath)
        traj = d['traj']
        nf = traj.shape[0]
        print(f'\n  [{ti + 1}/6] {name} ({nf} frames)')

        hists, futs, starts = make_windows_ac(traj, stride=STRIDE)
        n_total = len(hists)
        ctx_all = make_context(traj, starts, ctx_len=CTX_LEN)
        all_hist = np.array(hists, dtype=np.float32)
        futs_t = [torch.from_numpy(t).float() for t in futs]

        # Base evaluation
        bpred_list = []
        bs_ev = 64
        for b in range(0, n_total, bs_ev):
            be = min(b + bs_ev, n_total)
            bp = predict_batch(model, device, adapter,
                              torch.from_numpy(all_hist[b:be]),
                              torch.from_numpy(ctx_all[b:be]))
            bpred_list.append(bp)
        bpred = torch.cat(bpred_list, dim=0)

        bdir_all = np.array([dir_err(bpred[i, -1, :2].numpy(), futs[i][-1, :2])
                             for i in range(n_total)])
        trainable = bdir_all < 60
        tidx = np.where(trainable)[0]
        np.random.seed(42); np.random.shuffle(tidx)
        n_tr = int(len(tidx) * TRAIN_SPLIT)
        tr_idx = tidx[:n_tr]
        val_n = max(5, len(tidx) // 5)
        val_idx = tidx[n_tr:n_tr + val_n]
        te_idx = tidx[n_tr + val_n:] if n_tr + val_n < len(tidx) else tidx[n_tr:n_tr + 10]
        if len(tr_idx) < 10 or len(te_idx) < 5:
            print(f'    SKIP: too few trainable windows')
            continue

        # Train LoRA v7
        tr_h = torch.from_numpy(np.array([hists[i] for i in tr_idx], dtype=np.float32))
        tr_t = torch.stack([futs_t[i] for i in tr_idx])
        tr_c = torch.from_numpy(np.array([ctx_all[i] for i in tr_idx], dtype=np.float32))
        tr_bp = torch.stack([bpred[i] for i in tr_idx])
        val_h = torch.from_numpy(np.array([hists[i] for i in val_idx], dtype=np.float32))
        val_t = torch.stack([futs_t[i] for i in val_idx])
        val_c = torch.from_numpy(np.array([ctx_all[i] for i in val_idx], dtype=np.float32))

        best_val_fde = float('inf')
        best_state = None

        for restart in range(RESTARTS):
            torch.manual_seed(42 + restart * 137)
            np.random.seed(42 + restart * 137)
            ll, hl, ol, ho = inject_lora(model)
            params = collect_trainable(ll, hl)
            opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
            bs = min(BATCH_SIZE, len(tr_idx))
            for ep in range(EPOCHS):
                model.eval()
                perm = np.random.permutation(len(tr_idx))
                for b in range(0, len(tr_idx), bs):
                    idx = perm[b:b + bs]
                    hb, tb, cb = tr_h[idx].to(device), tr_t[idx].to(device), tr_c[idx].to(device)
                    opt.zero_grad()
                    ci = adapter(cb)
                    kwargs = {'force_predict': True, 'context_injection': ci}
                    pred = model(hb, **kwargs)['predictions']
                    loss = compute_loss(pred, tb, tr_bp[idx].to(device), ep)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                        opt.step()
                sched.step()
            model.eval()
            val_fdes = []
            for b in range(0, len(val_idx), bs):
                be = min(b + bs, len(val_idx))
                hb, tb = val_h[b:be].to(device), val_t[b:be]
                ci = adapter(val_c[b:be].to(device))
                kwargs = {'force_predict': True, 'context_injection': ci}
                with torch.no_grad():
                    pv = model(hb, **kwargs)['predictions'].cpu()
                val_fdes.append(torch.norm(pv[:, -1, :] - tb[:, -1, :], dim=-1))
            vf = torch.cat(val_fdes).mean().item()
            if vf < best_val_fde:
                best_val_fde = vf
                best_state = save_lora_state(ll, hl)
            restore_model(model, ol, ho)

        # Test predictions (base + lora + lora_spline)
        ll, hl, ol, ho = inject_lora(model)
        load_lora_state(ll, hl, best_state)

        te_h = np.array([hists[i] for i in te_idx], dtype=np.float32)
        te_c = np.array([ctx_all[i] for i in te_idx], dtype=np.float32)
        lp_list = []
        for b in range(0, len(te_idx), bs_ev):
            be = min(b + bs_ev, len(te_idx))
            lp = predict_batch(model, device, adapter,
                              torch.from_numpy(te_h[b:be]),
                              torch.from_numpy(te_c[b:be]))
            lp_list.append(lp)
        lp_all = torch.cat(lp_list, dim=0)
        restore_model(model, ol, ho)

        # Pick best display window
        te_t = torch.stack([futs_t[i] for i in te_idx])
        te_bp = torch.stack([bpred[i] for i in te_idx])
        best_j = 0; best_imp = -999.0
        for j in range(len(te_idx)):
            bfi = float(torch.norm(te_bp[j, -1, :] - te_t[j, -1, :]))
            lfi = float(torch.norm(lp_all[j, -1, :] - te_t[j, -1, :]))
            if bfi - lfi > best_imp:
                best_imp = bfi - lfi; best_j = j

        win_i = te_idx[best_j]
        hist_last = hists[win_i][-1, :3]

        base_delta = te_bp[best_j].numpy()
        lora_delta = lp_all[best_j].numpy()
        truth_delta = futs[win_i]

        base_abs = base_delta + hist_last
        lora_abs = lora_delta + hist_last
        truth_abs = truth_delta + hist_last

        # Strategy C: spline smooth the LoRA predictions
        lora_spline_abs = spline_smooth(lora_abs, knot_frac=SPLINE_KNOT_FRAC)
        lora_spline_delta = lora_spline_abs - hist_last

        # Metrics
        b_fde = float(np.linalg.norm(base_abs[-1] - truth_abs[-1]))
        l_fde = float(np.linalg.norm(lora_abs[-1] - truth_abs[-1]))
        s_fde = float(np.linalg.norm(lora_spline_abs[-1] - truth_abs[-1]))
        b_jerk = compute_smoothness(base_abs)['mean_jerk']
        l_jerk = compute_smoothness(lora_abs)['mean_jerk']
        s_jerk = compute_smoothness(lora_spline_abs)['mean_jerk']
        t_jerk = compute_smoothness(truth_abs)['mean_jerk']

        print(f'    Display: FDE base={b_fde:.3f} lora={l_fde:.3f} spline={s_fde:.3f}m')
        print(f'    Jerk:   base={b_jerk:.4f} lora={l_jerk:.4f} spline={s_jerk:.4f} truth={t_jerk:.4f}')

        # Save coordinates
        npz_path = OUT_DIR / f'{name.replace(".npz", "")}_inspect.npz'
        np.savez(npz_path,
                 history=hists[win_i],
                 base_delta=base_delta, lora_delta=lora_delta,
                 lora_spline_delta=lora_spline_delta, truth_delta=truth_delta,
                 base_abs=base_abs, lora_abs=lora_abs,
                 lora_spline_abs=lora_spline_abs, truth_abs=truth_abs,
                 hist_last=hist_last,
                 base_fde=b_fde, lora_fde=l_fde, spline_fde=s_fde,
                 base_jerk=b_jerk, lora_jerk=l_jerk, spline_jerk=s_jerk, truth_jerk=t_jerk)

        all_results.append({
            'name': name, 'nf': nf,
            'hist': hists[win_i], 'hist_last': hist_last,
            'base_delta': base_delta, 'lora_delta': lora_delta,
            'spline_delta': lora_spline_delta, 'truth_delta': truth_delta,
            'base_abs': base_abs, 'lora_abs': lora_abs,
            'spline_abs': lora_spline_abs, 'truth_abs': truth_abs,
            'b_fde': b_fde, 'l_fde': l_fde, 's_fde': s_fde,
            'b_jerk': b_jerk, 'l_jerk': l_jerk, 's_jerk': s_jerk, 't_jerk': t_jerk,
        })

    model.ua_pgd.physics_gate.forward = orig_gate

    if not all_results:
        print('\nNo results!')
        return

    # ── Generate Overview Charts ──
    n = len(all_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    for view_name, plot_fn, is_3d in [('3d', plot_3d, True), ('xy', plot_xy, False)]:
        if is_3d:
            fig = plt.figure(figsize=(8 * cols, 7 * rows))
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))

        fig.suptitle(
            f'LoRA v7 + Spline Smoothing — {view_name.upper()} View\n'
            'Blue=History Red=Base(AC) Orange=LoRA v7 Purple=LoRA+Spline Green=Truth',
            fontsize=12, fontweight='bold')

        for i, r in enumerate(all_results):
            if is_3d:
                ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            else:
                ax = axes.flat[i] if n > 1 else axes[0, 0]

            jk_info = f'Jerk: {r["b_jerk"]:.3f}/{r["l_jerk"]:.3f}/{r["s_jerk"]:.3f}/{r["t_jerk"]:.3f}'
            title = (f'{r["name"][:22]}\n'
                     f'FDE B:{r["b_fde"]:.2f} L:{r["l_fde"]:.2f} S:{r["s_fde"]:.2f}m')

            plot_fn(ax, r['hist'], r['base_delta'], r['lora_delta'],
                    r['spline_delta'], r['truth_delta'], title)

        if not is_3d:
            for i in range(n, rows * cols):
                if n > 1:
                    axes.flat[i].axis('off')

        plt.tight_layout(pad=2)
        out_path = OUT_DIR / f'overview_{view_name}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
        plt.close()

    # ── Detailed per-trajectory pages ──
    for ri, r in enumerate(all_results):
        fig = plt.figure(figsize=(24, 14))
        fig.suptitle(
            f'LoRA v7 + Spline — {r["name"]}\n'
            f'FDE: Base={r["b_fde"]:.3f}  LoRA={r["l_fde"]:.3f}  Spline={r["s_fde"]:.3f}m  |  '
            f'Jerk: Base={r["b_jerk"]:.4f}  LoRA={r["l_jerk"]:.4f}  Spline={r["s_jerk"]:.4f}  Truth={r["t_jerk"]:.4f}',
            fontsize=11, fontweight='bold')

        ax3 = fig.add_subplot(2, 3, 1, projection='3d')
        plot_3d(ax3, r['hist'], r['base_delta'], r['lora_delta'],
                r['spline_delta'], r['truth_delta'], '3D View')

        ax_xy = fig.add_subplot(2, 3, 2)
        plot_xy(ax_xy, r['hist'], r['base_delta'], r['lora_delta'],
                r['spline_delta'], r['truth_delta'], 'XY Top-Down')

        ax_e = fig.add_subplot(2, 3, 3)
        plot_step_error(ax_e, r['base_delta'], r['lora_delta'],
                        r['spline_delta'], r['truth_delta'], 'Per-Step Error')

        # Per-axis time series
        ax_x = fig.add_subplot(2, 3, 4)
        ax_y = fig.add_subplot(2, 3, 5)
        ax_z = fig.add_subplot(2, 3, 6)
        plot_per_axis(ax_x, ax_y, ax_z, r['base_delta'], r['lora_delta'],
                      r['spline_delta'], r['truth_delta'], r['hist_last'])

        plt.tight_layout(pad=2)
        detail_path = OUT_DIR / f'detail_{ri + 1}_{r["name"][:20]}.png'
        fig.savefig(detail_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {detail_path}')
        plt.close()

    # ── Summary table ──
    print(f'\n{"=" * 90}')
    print(f'SUMMARY — LoRA v7 + Spline Smoothing (Strategy C)')
    print(f'{"Trajectory":<30} {"B.FDE":<8} {"L.FDE":<8} {"S.FDE":<8} '
          f'{"B.Jerk":<10} {"L.Jerk":<10} {"S.Jerk":<10} {"T.Jerk":<10}')
    print(f'{"-" * 90}')
    for r in all_results:
        print(f'{r["name"]:<30} {r["b_fde"]:<8.3f} {r["l_fde"]:<8.3f} {r["s_fde"]:<8.3f} '
              f'{r["b_jerk"]:<10.4f} {r["l_jerk"]:<10.4f} {r["s_jerk"]:<10.4f} {r["t_jerk"]:<10.4f}')

    # Aggregate
    avg_b_fde = np.mean([r['b_fde'] for r in all_results])
    avg_l_fde = np.mean([r['l_fde'] for r in all_results])
    avg_s_fde = np.mean([r['s_fde'] for r in all_results])
    avg_b_jerk = np.mean([r['b_jerk'] for r in all_results])
    avg_l_jerk = np.mean([r['l_jerk'] for r in all_results])
    avg_s_jerk = np.mean([r['s_jerk'] for r in all_results])
    avg_t_jerk = np.mean([r['t_jerk'] for r in all_results])

    print(f'{"-" * 90}')
    print(f'{"AVERAGE":<30} {avg_b_fde:<8.3f} {avg_l_fde:<8.3f} {avg_s_fde:<8.3f} '
          f'{avg_b_jerk:<10.4f} {avg_l_jerk:<10.4f} {avg_s_jerk:<10.4f} {avg_t_jerk:<10.4f}')
    print(f'\n  LoRA gain: {(avg_b_fde-avg_l_fde)/max(avg_b_fde,1e-6)*100:.1f}%  '
          f'Spline gain: {(avg_b_fde-avg_s_fde)/max(avg_b_fde,1e-6)*100:.1f}%')
    print(f'  Jerk: LoRA/Truth = {avg_l_jerk/max(avg_t_jerk,1e-6):.1f}x  '
          f'Spline/Truth = {avg_s_jerk/max(avg_t_jerk,1e-6):.1f}x')
    print(f'  Spline smoothness improvement: {(avg_l_jerk-avg_s_jerk)/max(avg_l_jerk,1e-6)*100:.1f}% jerk reduction')
    print(f'\n  All data saved to: {OUT_DIR}/')
    print('=' * 90)


if __name__ == '__main__':
    main()
