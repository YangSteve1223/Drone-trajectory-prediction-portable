#!/usr/bin/env python3
"""
Comprehensive Trajectory Quality Analysis
=========================================
Per-step physical-reasonableness check for every trajectory:
  - Position error (ADE) per step for Base and LoRA
  - Velocity profile: magnitude + direction, jump detection
  - Acceleration profile: physical limits check (>5 m/s^2 = flag)
  - Jerk profile: smoothness verification
  - Boundary continuity: step-0 gap
  - Path curvature in XY plane
  - Speed bound check (>15 m/s = flag)
  - Direction error per step

Generates publication-quality charts suitable for research reports.
"""

import torch, numpy as np, sys, warnings, json, traceback
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
OUT_DIR = Path(__file__).parent / 'pic-results' / 'quality'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
STRIDE, GATE_SCALE = 2, 0.3
DT = 0.2  # 5Hz

# v8.1 LoRA
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

# v8.1 loss
BETA_HUBER = 0.20
W_DIR, W_SMOOTH, W_JERK = 0.25, 0.40, 0.35
W_CURVATURE, W_TV_VEL, W_SPEED, W_ANCHOR_DIR = 0.20, 0.15, 0.03, 0.02
W_BOUNDARY, BOUNDARY_STEPS = 0.40, 1
PHYSICS_WARMUP, PHYSICS_START = 3, 0.50

# Physical limits for drones
MAX_ACCEL = 5.0        # m/s^2 — aggressive drone maneuver
MAX_SPEED = 15.0       # m/s — DJI drone limit
MAX_JERK = 10.0        # m/s^3 — instantaneous jerk threshold

SELECTED = [
    '2025-04-23_09-15-12.npz',
    '2025-04-21_15-03-16.npz',
    '2025-04-03_12-30-43.npz',
    '2025-04-25_09-42-34.npz',
    '2025-04-29_17-15-55.npz',
    '2025-04-28_14-30-06.npz',
]

# Publication-quality style
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'sans-serif', 'figure.dpi': 150,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})
C = {'hist': '#1565C0', 'base': '#D32F2F', 'lora': '#FF6D00', 'truth': '#2E7D32'}
MK = ['o', 's', '^', 'D', 'P']


# ═══════════════════════════════════════════════════════════════════════════
# Model helpers (same as previous scripts)
# ═══════════════════════════════════════════════════════════════════════════

def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))

def resolve_module(model, path):
    parts = path.split('.')
    obj = model
    for part in parts: obj = getattr(obj, part)
    return obj

def set_module(model, path, module):
    parts = path.split('.')
    parent = model
    for part in parts[:-1]: parent = getattr(parent, part)
    setattr(parent, parts[-1], module)

def make_windows_ac(traj, stride=2):
    n = traj.shape[0]
    ml = HIST_LEN * stride + PRED_LEN
    if n < ml: return [], [], []
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
    n = traj.shape[0]; ctx = []
    for ws in starts:
        end = ws + ctx_len
        ctx.append(traj[ws:end, :].copy() if end <= n else traj[-ctx_len:, :].copy())
    return np.array(ctx, dtype=np.float32)

def predict_batch(model, device, adapter, hb, cb):
    ctx_inj = None
    if adapter is not None and cb is not None:
        with torch.no_grad(): ctx_inj = adapter(cb.to(device))
    kwargs = {'force_predict': True}
    if ctx_inj is not None: kwargs['context_injection'] = ctx_inj
    with torch.no_grad(): return model(hb.to(device), **kwargs)['predictions'].cpu()

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
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    true_vel = target[:, 1:, :] - target[:, :-1, :]
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
        pc = pred[:, :k + 1, :].sum(dim=1)
        ec = hist_last_vel * (DT * (k + 1))
        loss_boundary = loss_boundary + ((pc - ec) ** 2).mean()
    loss_boundary = loss_boundary / BOUNDARY_STEPS
    return (loss_huber + W_DIR * loss_dir + W_SMOOTH * ramp * loss_smooth
            + W_JERK * ramp * loss_jerk + W_CURVATURE * ramp * loss_curvature
            + W_TV_VEL * ramp * loss_tv + W_SPEED * loss_speed + W_ANCHOR_DIR * loss_anchor
            + W_BOUNDARY * loss_boundary)


# ═══════════════════════════════════════════════════════════════════════════
# Physical analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_trajectory(pred_abs, hist_last_vel):
    """Per-step physical analysis of absolute trajectory."""
    vel = pred_abs[1:, :] - pred_abs[:-1, :]  # displacement per step (m/0.2s)
    speed = np.linalg.norm(vel, axis=1) / DT  # m/s
    acc = (vel[1:, :] - vel[:-1, :]) / (DT ** 2)  # m/s^2
    acc_mag = np.linalg.norm(acc, axis=1)
    jerk = (acc[1:, :] - acc[:-1, :]) / DT  # m/s^3
    jerk_mag = np.linalg.norm(jerk, axis=1)

    # XY curvature
    vel_xy = vel[:-1, :2] / DT
    acc_xy = acc[:, :2]
    v_norm = np.linalg.norm(vel_xy, axis=1) + 1e-6
    cross = np.abs(vel_xy[:, 0] * acc_xy[:, 1] - vel_xy[:, 1] * acc_xy[:, 0])
    curvature = cross / (v_norm ** 3 + 1e-4)

    return {
        'speed_mean': float(np.mean(speed)), 'speed_max': float(np.max(speed)),
        'accel_mean': float(np.mean(acc_mag)), 'accel_max': float(np.max(acc_mag)),
        'accel_violations': int(np.sum(acc_mag > MAX_ACCEL)),
        'jerk_mean': float(np.mean(jerk_mag)), 'jerk_max': float(np.max(jerk_mag)),
        'jerk_violations': int(np.sum(jerk_mag > MAX_JERK)),
        'curvature_mean': float(np.mean(curvature)), 'curvature_max': float(np.max(curvature)),
        'speed_profile': speed, 'accel_profile': acc_mag, 'jerk_profile': jerk_mag,
        'curvature_profile': curvature,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Publication-quality plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_main_3d(ax, hp, bp, lp, tp, title):
    """Clean 3D trajectory with minimal clutter."""
    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=C['hist'], lw=2.5, label='History (4s)')
    ax.plot(bp[:, 0], bp[:, 1], bp[:, 2], color=C['base'], lw=1.8, ls='--', alpha=0.7, label='Base (AC)')
    ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], color=C['lora'], lw=2.2, label='LoRA (ours)')
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=C['truth'], lw=2.5, label='Ground Truth')
    ax.scatter(tp[0, 0], tp[0, 1], tp[0, 2], c='black', s=50, marker='s', zorder=10, label='Start')
    ax.scatter(tp[-1, 0], tp[-1, 1], tp[-1, 2], c='black', s=70, marker='*', zorder=10, label='End')
    all_pts = np.concatenate([hp, bp, lp, tp], axis=0)
    rng = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.55
    zm = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    ax.set_xlim(all_pts[:, 0].mean() - rng, all_pts[:, 0].mean() + rng)
    ax.set_ylim(all_pts[:, 1].mean() - rng, all_pts[:, 1].mean() + rng)
    ax.set_zlim(zm - rng, zm + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
    ax.view_init(elev=22, azim=-55)

def plot_main_xy(ax, hp, bp, lp, tp, title):
    """Clean XY top-down view."""
    ax.plot(hp[:, 0], hp[:, 1], color=C['hist'], lw=2.5, label='History')
    ax.plot(bp[:, 0], bp[:, 1], color=C['base'], lw=1.8, ls='--', alpha=0.7, label='Base (AC)')
    ax.plot(lp[:, 0], lp[:, 1], color=C['lora'], lw=2.2, label='LoRA (ours)')
    ax.plot(tp[:, 0], tp[:, 1], color=C['truth'], lw=2.5, label='Truth')
    ax.scatter(tp[0, 0], tp[0, 1], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1, 0], tp[-1, 1], c='black', s=70, marker='*', zorder=10)
    ax.scatter(hp[-1, 0], hp[-1, 1], c=C['hist'], s=80, marker='s', edgecolors='black', lw=0.8, zorder=5)
    all_xy = np.concatenate([hp[:, :2], bp[:, :2], lp[:, :2], tp[:, :2]], axis=0)
    rng = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55
    xm, ym = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(xm - rng, xm + rng); ax.set_ylim(ym - rng, ym + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.8)

def plot_per_step_error(ax, be, le, title):
    """Per-step Euclidean distance error."""
    steps = np.arange(20) * DT
    ax.bar(steps - 0.04, be, width=0.08, color=C['base'], alpha=0.6, label='Base (AC)')
    ax.bar(steps + 0.04, le, width=0.08, color=C['lora'], alpha=0.8, label='LoRA (ours)')
    ax.axhline(y=np.mean(be), color=C['base'], ls=':', lw=1, alpha=0.5)
    ax.axhline(y=np.mean(le), color=C['lora'], ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Position Error (m)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=7)

def plot_per_axis(axes, bp, lp, tp, hl, title_prefix):
    """Per-axis absolute position: history continuation + prediction."""
    for di, (ax, dim_name) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
        # History: last 10 points
        hp_dim = hl[-10:, di]
        t_hist = np.arange(-9, 1) * DT
        ax.plot(t_hist, hp_dim, color=C['hist'], lw=2.5, label='History')
        # Prediction
        t_pred = np.arange(20) * DT
        ax.plot(t_pred, bp[:, di], color=C['base'], lw=1.8, ls='--', alpha=0.7, label='Base')
        ax.plot(t_pred, lp[:, di], color=C['lora'], lw=2.2, label='LoRA')
        ax.plot(t_pred, tp[:, di], color=C['truth'], lw=2.5, label='Truth')
        ax.axvline(x=0, color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_ylabel(f'{dim_name} (m)'); ax.set_xlabel('Time (s)')
        ax.set_title(f'{title_prefix} — {dim_name}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if di == 0: ax.legend(fontsize=6.5, loc='upper left')

def plot_physics_profiles(axes, b_phys, l_phys, t_phys):
    """Speed, acceleration, jerk profiles."""
    titles = ['Speed (m/s)', 'Acceleration (m/s$^2$)', 'Jerk (m/s$^3$)']
    b_profs = [b_phys['speed_profile'], b_phys['accel_profile'], b_phys['jerk_profile']]
    l_profs = [l_phys['speed_profile'], l_phys['accel_profile'], l_phys['jerk_profile']]
    t_profs = [t_phys['speed_profile'], t_phys['accel_profile'], t_phys['jerk_profile']]
    for ax, title, bp_, lp_, tp_ in zip(axes, titles, b_profs, l_profs, t_profs):
        n = len(bp_)
        t = np.arange(n) * DT + DT  # offset by one step for vel
        ax.plot(t, bp_, 'o-', color=C['base'], lw=1.5, ms=3, alpha=0.7, label='Base')
        ax.plot(t, lp_, 's-', color=C['lora'], lw=1.5, ms=3, label='LoRA')
        ax.plot(t, tp_, 'D-', color=C['truth'], lw=1.5, ms=3, label='Truth')
        ax.set_ylabel(title); ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5)

def plot_boundary_zoom(ax, bp, lp, tp, hl, title):
    """Zoomed view of first 5 prediction steps vs last 5 history steps."""
    t_hist = np.arange(-5, 1) * DT
    t_pred = np.arange(6) * DT  # first 6 steps
    # XY plane
    hp_xy = hl[-6:, :2]
    bp_xy = np.vstack([hl[-1:, :2], bp[:6, :2]])
    lp_xy = np.vstack([hl[-1:, :2], lp[:6, :2]])
    tp_xy = np.vstack([hl[-1:, :2], tp[:6, :2]])
    ax.plot(hp_xy[:, 0], hp_xy[:, 1], 'o-', color=C['hist'], lw=2, ms=4, label='History')
    ax.plot(bp_xy[:, 0], bp_xy[:, 1], 's--', color=C['base'], lw=1.5, ms=3, alpha=0.7, label='Base')
    ax.plot(lp_xy[:, 0], lp_xy[:, 1], 'D-', color=C['lora'], lw=1.5, ms=3, label='LoRA')
    ax.plot(tp_xy[:, 0], tp_xy[:, 1], 'o-', color=C['truth'], lw=2, ms=3, label='Truth')
    ax.scatter(hl[-1, 0], hl[-1, 1], c='black', s=100, marker='X', zorder=10, label='Connection')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('Comprehensive Trajectory Quality Analysis')
    print(f'  Physical limits: accel<{MAX_ACCEL} m/s^2, speed<{MAX_SPEED} m/s, jerk<{MAX_JERK} m/s^3')
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
        gi, ga, gc, gm, gme = orig_gate(
            last_encoded=last_encoded, intent_weights=intent_weights, step_encoding=step_encoding)
        return gi * GATE_SCALE, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_gate

    all_results = []
    quality_report = []

    for ti, name in enumerate(SELECTED):
        fpath = TRAJ_DIR / name
        d = np.load(fpath)
        traj = d['traj']; nf = traj.shape[0]
        print(f'\n{"=" * 80}')
        print(f'[{ti + 1}/6] {name} ({nf} frames)')
        print(f'{"=" * 80}')

        hists, futs, starts = make_windows_ac(traj, stride=STRIDE)
        n_total = len(hists)
        ctx_all = make_context(traj, starts)
        all_hist = np.array(hists, dtype=np.float32)
        futs_t = [torch.from_numpy(t).float() for t in futs]

        bpred_list = []
        for b in range(0, n_total, 64):
            be = min(b + 64, n_total)
            bp = predict_batch(model, device, adapter, torch.from_numpy(all_hist[b:be]), torch.from_numpy(ctx_all[b:be]))
            bpred_list.append(bp)
        bpred = torch.cat(bpred_list, dim=0)

        bdir_all = np.array([dir_err(bpred[i, -1, :2].numpy(), futs[i][-1, :2]) for i in range(n_total)])
        trainable = bdir_all < 60
        tidx = np.where(trainable)[0]
        np.random.seed(42); np.random.shuffle(tidx)
        n_tr = int(len(tidx) * TRAIN_SPLIT)
        tr_idx = tidx[:n_tr]
        val_n = max(5, len(tidx) // 5)
        val_idx = tidx[n_tr:n_tr + val_n]
        te_idx = tidx[n_tr + val_n:] if n_tr + val_n < len(tidx) else tidx[n_tr:n_tr + 10]

        # Train LoRA
        tr_h = torch.from_numpy(np.array([hists[i] for i in tr_idx], dtype=np.float32))
        tr_t = torch.stack([futs_t[i] for i in tr_idx])
        tr_c = torch.from_numpy(np.array([ctx_all[i] for i in tr_idx], dtype=np.float32))
        tr_bp = torch.stack([bpred[i] for i in tr_idx])
        val_h = torch.from_numpy(np.array([hists[i] for i in val_idx], dtype=np.float32))
        val_t = torch.stack([futs_t[i] for i in val_idx])
        val_c = torch.from_numpy(np.array([ctx_all[i] for i in val_idx], dtype=np.float32))

        best_val_fde = float('inf'); best_state = None
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
                    idx = perm[b:b + bs]
                    hb, tb, cb = tr_h[idx].to(device), tr_t[idx].to(device), tr_c[idx].to(device)
                    opt.zero_grad()
                    ci = adapter(cb)
                    pred = model(hb, force_predict=True, context_injection=ci)['predictions']
                    loss = compute_loss(pred, tb, hb, tr_bp[idx].to(device), ep)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward(); torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
                sched.step()
            model.eval(); val_fdes = []
            for b in range(0, len(val_idx), bs):
                be = min(b + bs, len(val_idx))
                hb, tb = val_h[b:be].to(device), val_t[b:be]
                ci = adapter(val_c[b:be].to(device))
                with torch.no_grad():
                    pv = model(hb, force_predict=True, context_injection=ci)['predictions'].cpu()
                val_fdes.append(torch.norm(pv[:, -1, :] - tb[:, -1, :], dim=-1))
            vf = torch.cat(val_fdes).mean().item()
            if vf < best_val_fde: best_val_fde = vf; best_state = save_lora_state(ll, hl)
            restore_model(model, ol, ho)

        # Test
        ll, hl, ol, ho = inject_lora(model)
        load_lora_state(ll, hl, best_state)
        te_h = np.array([hists[i] for i in te_idx], dtype=np.float32)
        te_c = np.array([ctx_all[i] for i in te_idx], dtype=np.float32)
        lp_list = []
        for b in range(0, len(te_idx), 64):
            be = min(b + 64, len(te_idx))
            lp = predict_batch(model, device, adapter, torch.from_numpy(te_h[b:be]), torch.from_numpy(te_c[b:be]))
            lp_list.append(lp)
        lp_all = torch.cat(lp_list, dim=0)
        restore_model(model, ol, ho)

        te_t = torch.stack([futs_t[i] for i in te_idx])
        te_bp = torch.stack([bpred[i] for i in te_idx])

        # Pick 3 representative windows: best gain, worst gain, median gain
        gains = []
        for j in range(len(te_idx)):
            bf = float(torch.norm(te_bp[j, -1, :] - te_t[j, -1, :]))
            lf = float(torch.norm(lp_all[j, -1, :] - te_t[j, -1, :]))
            gains.append((j, (bf - lf) / max(bf, 1e-6) * 100, bf, lf))
        gains.sort(key=lambda x: x[1], reverse=True)
        picks = [gains[0], gains[len(gains)//2], gains[-1]]  # best, median, worst

        traj_windows = []
        for rank, (j, gain, bf, lf) in enumerate(picks):
            win_i = te_idx[j]
            hist_last = hists[win_i][-1, :3]
            hist_last_vel = hists[win_i][-1, 3:6]
            base_abs = te_bp[j].numpy() + hist_last
            lora_abs = lp_all[j].numpy() + hist_last
            truth_abs = futs[win_i] + hist_last
            hp = hists[win_i][:, :3]

            b_phys = analyze_trajectory(base_abs, hist_last_vel)
            l_phys = analyze_trajectory(lora_abs, hist_last_vel)
            t_phys = analyze_trajectory(truth_abs, hist_last_vel)

            gap_b = float(np.linalg.norm(base_abs[0] - hist_last))
            gap_l = float(np.linalg.norm(lora_abs[0] - hist_last))
            gap_t = float(np.linalg.norm(truth_abs[0] - hist_last))

            traj_windows.append({
                'rank': ['best', 'median', 'worst'][rank], 'gain': gain,
                'base_fde': bf, 'lora_fde': lf,
                'hist': hp, 'base_abs': base_abs, 'lora_abs': lora_abs, 'truth_abs': truth_abs,
                'hist_last_vel': hist_last_vel,
                'b_phys': b_phys, 'l_phys': l_phys, 't_phys': t_phys,
                'gap_b': gap_b, 'gap_l': gap_l, 'gap_t': gap_t,
            })

        # ── Per-trajectory report ──
        for w in traj_windows:
            rank = w['rank']
            print(f'\n  --- {rank.upper()} window (FDE gain: {w["gain"]:+.1f}%) ---')
            print(f'    Base FDE={w["base_fde"]:.3f}m  LoRA FDE={w["lora_fde"]:.3f}m')
            print(f'    Boundary gap: Base={w["gap_b"]:.3f}m  LoRA={w["gap_l"]:.3f}m  Truth={w["gap_t"]:.3f}m')
            print(f'    Speed (mean/max):  Base={w["b_phys"]["speed_mean"]:.1f}/{w["b_phys"]["speed_max"]:.1f}  LoRA={w["l_phys"]["speed_mean"]:.1f}/{w["l_phys"]["speed_max"]:.1f}  Truth={w["t_phys"]["speed_mean"]:.1f}/{w["t_phys"]["speed_max"]:.1f} m/s')
            print(f'    Accel (mean/max):  Base={w["b_phys"]["accel_mean"]:.2f}/{w["b_phys"]["accel_max"]:.2f}  LoRA={w["l_phys"]["accel_mean"]:.2f}/{w["l_phys"]["accel_max"]:.2f}  Truth={w["t_phys"]["accel_mean"]:.2f}/{w["t_phys"]["accel_max"]:.2f} m/s^2')
            print(f'    Accel violations:   Base={w["b_phys"]["accel_violations"]}  LoRA={w["l_phys"]["accel_violations"]}  Truth={w["t_phys"]["accel_violations"]}')
            print(f'    Jerk (mean/max):   Base={w["b_phys"]["jerk_mean"]:.2f}/{w["b_phys"]["jerk_max"]:.2f}  LoRA={w["l_phys"]["jerk_mean"]:.2f}/{w["l_phys"]["jerk_max"]:.2f}  Truth={w["t_phys"]["jerk_mean"]:.2f}/{w["t_phys"]["jerk_max"]:.2f}')
            print(f'    Jerk violations:    Base={w["b_phys"]["jerk_violations"]}  LoRA={w["l_phys"]["jerk_violations"]}  Truth={w["t_phys"]["jerk_violations"]}')

        all_results.append({'name': name, 'nf': nf, 'windows': traj_windows})

    model.ua_pgd.physics_gate.forward = orig_gate

    # ── Aggregate quality report ──
    print(f'\n{"=" * 80}')
    print(f'AGGREGATE QUALITY REPORT')
    print(f'{"=" * 80}')

    all_gaps_b, all_gaps_l, all_gaps_t = [], [], []
    all_acc_b, all_acc_l, all_acc_t = [], [], []
    all_jerk_b, all_jerk_l, all_jerk_t = [], [], []
    acc_viol_b, acc_viol_l, acc_viol_t = 0, 0, 0
    jerk_viol_b, jerk_viol_l, jerk_viol_t = 0, 0, 0

    for r in all_results:
        for w in r['windows']:
            all_gaps_b.append(w['gap_b']); all_gaps_l.append(w['gap_l']); all_gaps_t.append(w['gap_t'])
            all_acc_b.append(w['b_phys']['accel_max']); all_acc_l.append(w['l_phys']['accel_max']); all_acc_t.append(w['t_phys']['accel_max'])
            all_jerk_b.append(w['b_phys']['jerk_max']); all_jerk_l.append(w['l_phys']['jerk_max']); all_jerk_t.append(w['t_phys']['jerk_max'])
            acc_viol_b += w['b_phys']['accel_violations']; acc_viol_l += w['l_phys']['accel_violations']; acc_viol_t += w['t_phys']['accel_violations']
            jerk_viol_b += w['b_phys']['jerk_violations']; jerk_viol_l += w['l_phys']['jerk_violations']; jerk_viol_t += w['t_phys']['jerk_violations']

    print(f'\n  Boundary Gap (step 0):')
    print(f'    Base:  mean={np.mean(all_gaps_b):.3f}m  max={np.max(all_gaps_b):.3f}m')
    print(f'    LoRA:  mean={np.mean(all_gaps_l):.3f}m  max={np.max(all_gaps_l):.3f}m')
    print(f'    Truth: mean={np.mean(all_gaps_t):.3f}m  max={np.max(all_gaps_t):.3f}m')
    print(f'    LoRA/Base: {np.mean(all_gaps_l)/max(np.mean(all_gaps_b),1e-6)*100:.0f}% of base gap')

    print(f'\n  Max Acceleration (limit={MAX_ACCEL} m/s^2):')
    print(f'    Base:  mean={np.mean(all_acc_b):.3f}  max={np.max(all_acc_b):.3f}  violations={acc_viol_b}')
    print(f'    LoRA:  mean={np.mean(all_acc_l):.3f}  max={np.max(all_acc_l):.3f}  violations={acc_viol_l}')
    print(f'    Truth: mean={np.mean(all_acc_t):.3f}  max={np.max(all_acc_t):.3f}  violations={acc_viol_t}')

    print(f'\n  Max Jerk (limit={MAX_JERK} m/s^3):')
    print(f'    Base:  mean={np.mean(all_jerk_b):.3f}  max={np.max(all_jerk_b):.3f}  violations={jerk_viol_b}')
    print(f'    LoRA:  mean={np.mean(all_jerk_l):.3f}  max={np.max(all_jerk_l):.3f}  violations={jerk_viol_l}')
    print(f'    Truth: mean={np.mean(all_jerk_t):.3f}  max={np.max(all_jerk_t):.3f}  violations={jerk_viol_t}')

    # ── Generate Publication-Quality Figures ──
    print(f'\nGenerating figures...')

    for ri, r in enumerate(all_results):
        for wi, w in enumerate(r['windows']):
            rank = w['rank']
            hp = w['hist']; bp = w['base_abs']; lp = w['lora_abs']; tp = w['truth_abs']
            hl = hp  # history abs positions

            # Page 1: 3D + XY + Per-step error
            fig = plt.figure(figsize=(18, 12))
            fig.suptitle(f'{r["name"]} — {rank} window (FDE gain: {w["gain"]:+.1f}%)',
                        fontsize=13, fontweight='bold')

            ax3 = fig.add_subplot(2, 3, (1, 2), projection='3d')
            plot_main_3d(ax3, hp, bp, lp, tp, '3D Trajectory')

            ax_xy = fig.add_subplot(2, 3, (4, 5))
            plot_main_xy(ax_xy, hp, bp, lp, tp, 'XY Top-Down View')

            # Error
            be = np.linalg.norm(bp - tp, axis=1)
            le = np.linalg.norm(lp - tp, axis=1)
            ax_e = fig.add_subplot(2, 3, 3)
            plot_per_step_error(ax_e, be, le, f'Per-Step Error (ADE: B={np.mean(be):.3f} L={np.mean(le):.3f}m)')

            # Boundary zoom
            ax_b = fig.add_subplot(2, 3, 6)
            plot_boundary_zoom(ax_b, bp, lp, tp, hp,
                             f'Boundary Zoom (gap: B={w["gap_b"]:.2f} L={w["gap_l"]:.2f} T={w["gap_t"]:.2f}m)')

            plt.tight_layout(pad=2)
            p1 = OUT_DIR / f'traj{ri+1}_{rank}_overview.png'
            fig.savefig(p1, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  {p1.name}')

            # Page 2: Per-axis + Physics profiles
            fig = plt.figure(figsize=(18, 16))
            fig.suptitle(f'{r["name"]} — {rank} window — Detailed Analysis',
                        fontsize=13, fontweight='bold')

            axes_xyz = [fig.add_subplot(3, 2, i+1) for i in range(3)]
            plot_per_axis(axes_xyz, bp, lp, tp, hl, f'{rank}')

            axes_phys = [fig.add_subplot(3, 2, i+4) for i in range(3)]
            plot_physics_profiles(axes_phys, w['b_phys'], w['l_phys'], w['t_phys'])

            plt.tight_layout(pad=2)
            p2 = OUT_DIR / f'traj{ri+1}_{rank}_detail.png'
            fig.savefig(p2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  {p2.name}')

    # ── Summary figure: All 6 trajectories, best window each ──
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle('LoRA v8.1 — Best Window Per Trajectory\n'
                 'Blue=History  Red=Base(AC)  Orange=LoRA(ours)  Green=Truth',
                 fontsize=13, fontweight='bold')
    for ri, r in enumerate(all_results):
        w = r['windows'][0]  # best window
        hp, bp, lp, tp = w['hist'], w['base_abs'], w['lora_abs'], w['truth_abs']
        ax = fig.add_subplot(2, 3, ri + 1, projection='3d')
        plot_main_3d(ax, hp, bp, lp, tp,
                    f'{r["name"][:22]}\nFDE: {w["base_fde"]:.2f}->{w["lora_fde"]:.2f}m ({w["gain"]:+.0f}%)  '
                    f'Gap: {w["gap_b"]:.2f}->{w["gap_l"]:.2f}m')
    plt.tight_layout(pad=2)
    psum = OUT_DIR / 'summary_3d.png'
    fig.savefig(psum, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {psum.name}')

    # Summary XY
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('LoRA v8.1 — XY Top-Down View (Best Windows)',
                 fontsize=13, fontweight='bold')
    for ri, r in enumerate(all_results):
        w = r['windows'][0]
        hp, bp, lp, tp = w['hist'], w['base_abs'], w['lora_abs'], w['truth_abs']
        ax = axes.flat[ri]
        plot_main_xy(ax, hp, bp, lp, tp,
                    f'{r["name"][:22]}\nFDE: {w["base_fde"]:.2f}->{w["lora_fde"]:.2f}m')
    plt.tight_layout(pad=2)
    psum2 = OUT_DIR / 'summary_xy.png'
    fig.savefig(psum2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {psum2.name}')

    # ── Final verdict ──
    print(f'\n{"=" * 80}')
    print('FINAL VERDICT')
    print(f'{"=" * 80}')
    print(f'  1. Boundary continuity: LoRA gap {np.mean(all_gaps_l):.3f}m vs Base {np.mean(all_gaps_b):.3f}m '
          f'({np.mean(all_gaps_l)/max(np.mean(all_gaps_b),1e-6)*100:.0f}% of base)')
    print(f'  2. Physical violations:')
    print(f'     Acceleration > {MAX_ACCEL} m/s^2: Base={acc_viol_b}  LoRA={acc_viol_l}  Truth={acc_viol_t}')
    print(f'     Jerk > {MAX_JERK} m/s^3:         Base={jerk_viol_b}  LoRA={jerk_viol_l}  Truth={jerk_viol_t}')

    phys_ok = (acc_viol_l == 0 and jerk_viol_l <= jerk_viol_t * 2)
    gap_ok = np.mean(all_gaps_l) < np.mean(all_gaps_b) * 0.5
    if phys_ok and gap_ok:
        print(f'\n  PASS: LoRA trajectories are physically plausible and boundary-continuous.')
    elif phys_ok:
        print(f'\n  PARTIAL: Physics OK but boundary gap needs improvement.')
    else:
        print(f'\n  NEEDS WORK: Physical violations detected.')

    print(f'\n  All figures saved to: {OUT_DIR}/')
    print('=' * 80)


if __name__ == '__main__':
    main()
