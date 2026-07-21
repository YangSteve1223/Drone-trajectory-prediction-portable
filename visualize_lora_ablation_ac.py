#!/usr/bin/env python3
"""
Visualize LoRA v5 ablation: 3D + XY views for selected long LOW trajectories.

All predictions use ContextAdapter(AC) + A(stride=2) + C(gate=0.3).
Compares: baseline(AC) vs LoRA(AC+LoRA) vs Ground Truth.

Colors: Blue=History  Red=Base(AC)  Orange=LoRA(AC+LoRA)  Green=Truth
Z-axis uses same scale as XY for undistorted spatial perception.
"""

import torch, numpy as np, sys, warnings, traceback
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from context_adapter import ContextAdapterV2
from lora import LoRALinear

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
ADAPTER_PATH = Path(__file__).parent / 'weights' / 'context_adapter_ac.pth'
OUT_DIR = Path(__file__).parent / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
STRIDE = 2
GATE_SCALE = 0.3

LORA_TARGETS = [
    ('ua_pgd.feat_compress', 32),
    ('ua_pgd.neural_decoder.proj.0', 16),
    ('ua_pgd.neural_decoder.delta_head', 16),
]
HEAD_TARGETS = ['ua_pgd.anchor_to_pos.2']

EPOCHS, RESTARTS = 20, 3
LR_MAX, LR_MIN = 1e-3, 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.0
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
BETA_HUBER = 0.1
W_DIR, W_SMOOTH, W_SPEED, W_ANCHOR_DIR = 0.15, 0.03, 0.02, 0.05

# Select 6 representative trajectories (best → median → worst FDE gain)
SELECTED = [
    '2025-04-23_09-15-12.npz',   # +88.4% — best
    '2025-04-21_15-03-16.npz',   # +83.8%
    '2025-04-03_12-30-43.npz',   # +80.5%
    '2025-04-25_09-42-34.npz',   # +78.1%
    '2025-04-29_17-15-55.npz',   # +63.2%
    '2025-04-28_14-30-06.npz',   # +34.0% — worst
]

# Time marks for 4-second prediction at 5Hz
TIME_MARKS = {0: '0.0s', 5: '1.0s', 10: '2.0s', 15: '3.0s', 19: '4.0s'}
MK_COLORS = ['#E91E63', '#9C27B0', '#00BCD4', '#FFEB3B', '#FF5722']
MK_SHAPES = ['o', 's', '^', 'D', 'P']

C = {'hist': '#1565C0', 'base': '#D32F2F', 'lora': '#FF6D00', 'truth': '#2E7D32'}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
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


def make_windows(traj, stride=2):
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


def compute_loss(pred, target, base_pred):
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    true_vel = target[:, 1:, :] - target[:, :-1, :]
    loss_dir = (1.0 - F.cosine_similarity(pred_vel, true_vel, dim=-1)).mean()
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    loss_smooth = pred_acc.abs().mean()
    loss_speed = F.relu(pred_vel.norm(dim=-1) - 3.0).mean()
    base_dir = F.normalize(base_pred[:, -1, :2] - base_pred[:, 0, :2], dim=-1)
    pred_dir = F.normalize(pred[:, -1, :2] - pred[:, 0, :2], dim=-1)
    loss_anchor = (1.0 - (base_dir * pred_dir).sum(dim=-1)).mean()
    return loss_huber + W_DIR * loss_dir + W_SMOOTH * loss_smooth + W_SPEED * loss_speed + W_ANCHOR_DIR * loss_anchor


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d(ax, hist, base_pred, lora_pred, truth, title):
    """3D trajectory view. Z uses same scale as XY."""
    last = hist[-1, :3]
    hp = hist[:, :3]
    bp_a = base_pred + last
    lp_a = lora_pred + last
    tp_a = truth + last

    ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=C['hist'], lw=2.5, label='History (4s)')
    ax.plot(bp_a[:, 0], bp_a[:, 1], bp_a[:, 2], color=C['base'], lw=1.8,
            ls='--', alpha=0.75, label='Base (AC)')
    ax.plot(lp_a[:, 0], lp_a[:, 1], lp_a[:, 2], color=C['lora'], lw=2.2,
            label='LoRA (AC+LoRA)')
    ax.plot(tp_a[:, 0], tp_a[:, 1], tp_a[:, 2], color=C['truth'], lw=2.5,
            label='Ground Truth')

    # Time marks on LoRA and Truth
    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.scatter(lp_a[fidx, 0], lp_a[fidx, 1], lp_a[fidx, 2],
                   c=MK_COLORS[fi], s=60, marker=MK_SHAPES[fi],
                   edgecolors='black', lw=0.6, zorder=10)
        ax.scatter(tp_a[fidx, 0], tp_a[fidx, 1], tp_a[fidx, 2],
                   c=MK_COLORS[fi], s=45, marker=MK_SHAPES[fi],
                   edgecolors='black', lw=0.5, zorder=10, alpha=0.7)

    # ── Equal aspect ratio for all 3 axes ──
    all_pts = np.concatenate([hp, bp_a, lp_a, tp_a], axis=0)
    xy_range = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.55
    z_mid = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    ax.set_xlim(all_pts[:, 0].mean() - xy_range, all_pts[:, 0].mean() + xy_range)
    ax.set_ylim(all_pts[:, 1].mean() - xy_range, all_pts[:, 1].mean() + xy_range)
    ax.set_zlim(z_mid - xy_range, z_mid + xy_range)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.view_init(elev=22, azim=-55)


def plot_xy(ax, hist, base_pred, lora_pred, truth, title):
    """XY top-down view. Same scale as 3D's XY."""
    last = hist[-1, :2]
    hp = hist[:, :2]
    bp_a = base_pred[:, :2] + last
    lp_a = lora_pred[:, :2] + last
    tp_a = truth[:, :2] + last

    ax.plot(hp[:, 0], hp[:, 1], color=C['hist'], lw=2.5, label='History (4s)')
    ax.plot(bp_a[:, 0], bp_a[:, 1], color=C['base'], lw=1.8, ls='--', alpha=0.75,
            label='Base (AC)')
    ax.plot(lp_a[:, 0], lp_a[:, 1], color=C['lora'], lw=2.2, label='LoRA (AC+LoRA)')
    ax.plot(tp_a[:, 0], tp_a[:, 1], color=C['truth'], lw=2.5, label='Ground Truth')

    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.scatter(lp_a[fidx, 0], lp_a[fidx, 1], c=MK_COLORS[fi], s=70,
                   marker=MK_SHAPES[fi], edgecolors='black', lw=0.6, zorder=10)
        ax.scatter(tp_a[fidx, 0], tp_a[fidx, 1], c=MK_COLORS[fi], s=55,
                   marker=MK_SHAPES[fi], edgecolors='black', lw=0.5, zorder=10, alpha=0.7)

    ax.scatter(hp[-1, 0], hp[-1, 1], c=C['hist'], s=80, marker='s',
               edgecolors='black', lw=0.8, zorder=5)

    # Same XY range as 3D
    all_xy = np.concatenate([hp[:, :2], bp_a, lp_a, tp_a], axis=0)
    xy_range = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55
    xm, ym = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(xm - xy_range, xm + xy_range)
    ax.set_ylim(ym - xy_range, ym + xy_range)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper left')


def plot_step_error(ax, base_pred, lora_pred, truth, title):
    """Per-step Euclidean error: base vs lora."""
    be = np.linalg.norm(base_pred - truth, axis=1)
    le = np.linalg.norm(lora_pred - truth, axis=1)
    steps = np.arange(20) * 0.2
    ax.plot(steps, be, 'o-', color=C['base'], lw=2, ms=4, label=f'Base (AC)')
    ax.plot(steps, le, 's-', color=C['lora'], lw=2, ms=4, label=f'LoRA (AC+LoRA)')
    for fi, (fidx, _) in enumerate(TIME_MARKS.items()):
        ax.axvline(x=fidx * 0.2, color=MK_COLORS[fi], ls=':', alpha=0.4, lw=0.8)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Error (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('LoRA Ablation Visualization — 6 representative long LOW trajectories')
    print(f'  Baseline: ContextAdapter(AC) + A(stride=2) + C(gate=0.3)')
    print('=' * 80)

    # Load model
    p = DronePredictor()
    model = p.low; model.eval(); device = p.device

    # Load adapter
    adapter = ContextAdapterV2(input_dim=6, context_len=CTX_LEN,
                                d_model=model.d_model, hidden=128).to(device)
    adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=device))
    adapter.eval()

    # Apply gate_scale (C)
    orig_gate = model.ua_pgd.physics_gate.forward
    def scaled_gate(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_gate(last_encoded, intent_weights, step_encoding)
        return gi * GATE_SCALE, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_gate

    results = []

    for ti, name in enumerate(SELECTED):
        fpath = TRAJ_DIR / name
        if not fpath.exists():
            print(f'  [{ti + 1}] {name} — NOT FOUND, skipping')
            continue

        d = np.load(fpath)
        traj = d['traj']
        nf = traj.shape[0]
        print(f'\n  [{ti + 1}/6] {name} ({nf} frames)')

        hists, futs, starts = make_windows(traj, stride=STRIDE)
        n_total = len(hists)
        ctx_all = make_context(traj, starts, ctx_len=CTX_LEN)

        # ── Base evaluation ──
        all_hist = np.array(hists, dtype=np.float32)
        bpred_list = []
        bs_ev = 64
        for b in range(0, n_total, bs_ev):
            be_ = min(b + bs_ev, n_total)
            hb = torch.from_numpy(all_hist[b:be_])
            cb = torch.from_numpy(ctx_all[b:be_])
            bp = predict_batch(model, device, adapter, hb, cb)
            bpred_list.append(bp)
        bpred_all = torch.cat(bpred_list, dim=0)

        futs_t = [torch.from_numpy(t).float() for t in futs]
        bdir_all = np.array([dir_err(bpred_all[i, -1, :2].numpy(), futs[i][-1, :2])
                             for i in range(n_total)])

        trainable = bdir_all < 60
        tidx_all = np.where(trainable)[0]
        np.random.seed(42)
        np.random.shuffle(tidx_all)
        n_tr = int(len(tidx_all) * TRAIN_SPLIT)
        tr_idx = tidx_all[:n_tr]
        val_n = max(5, len(tidx_all) // 5)
        val_idx = tidx_all[n_tr:n_tr + val_n]
        te_idx = tidx_all[n_tr + val_n:] if n_tr + val_n < len(tidx_all) else tidx_all[n_tr:n_tr + 10]

        if len(tr_idx) < 10 or len(te_idx) < 5:
            print(f'    SKIP: too few trainable windows')
            continue

        # ── LoRA training ──
        tr_h = torch.from_numpy(np.array([hists[i] for i in tr_idx], dtype=np.float32))
        tr_t = torch.stack([futs_t[i] for i in tr_idx])
        tr_c = torch.from_numpy(np.array([ctx_all[i] for i in tr_idx], dtype=np.float32))
        tr_bp = torch.stack([bpred_all[i] for i in tr_idx])
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
                    loss = compute_loss(pred, tb, tr_bp[idx].to(device))
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                        opt.step()
                sched.step()
            # Validation
            model.eval()
            val_fdes = []
            for b in range(0, len(val_idx), bs):
                be_ = min(b + bs, len(val_idx))
                hb, tb = val_h[b:be_].to(device), val_t[b:be_]
                ci = adapter(val_c[b:be_].to(device))
                kwargs = {'force_predict': True, 'context_injection': ci}
                with torch.no_grad():
                    pv = model(hb, **kwargs)['predictions'].cpu()
                val_fdes.append(torch.norm(pv[:, -1, :] - tb[:, -1, :], dim=-1))
            vf = torch.cat(val_fdes).mean().item()
            if vf < best_val_fde:
                best_val_fde = vf
                best_state = save_lora_state(ll, hl)
            restore_model(model, ol, ho)

        # ── Test predictions ──
        ll, hl, ol, ho = inject_lora(model)
        load_lora_state(ll, hl, best_state)

        te_h = np.array([hists[i] for i in te_idx], dtype=np.float32)
        te_c = np.array([ctx_all[i] for i in te_idx], dtype=np.float32)
        lp_list = []
        for b in range(0, len(te_idx), bs_ev):
            be_ = min(b + bs_ev, len(te_idx))
            lp = predict_batch(model, device, adapter,
                              torch.from_numpy(te_h[b:be_]),
                              torch.from_numpy(te_c[b:be_]))
            lp_list.append(lp)
        lp_all = torch.cat(lp_list, dim=0)

        restore_model(model, ol, ho)

        # Pick best display window: largest FDE improvement, DirErr < 90
        best_j = 0; best_imp = -999.0
        for j in range(len(te_idx)):
            i = te_idx[j]
            if bdir_all[i] >= 90:
                continue
            bf = float(torch.norm(bpred_all[i, -1, :] - futs_t[i][-1, :]))
            lf = float(torch.norm(lp_all[j, -1, :] - futs_t[i][-1, :]))
            if bf - lf > best_imp:
                best_imp = bf - lf; best_j = j

        win_idx = te_idx[best_j]
        hist_np = hists[win_idx]
        base_np = bpred_all[win_idx].numpy()
        lora_np = lp_all[best_j].numpy()
        truth_np = futs[win_idx]

        b_fde = float(np.linalg.norm(base_np[-1] - truth_np[-1]))
        l_fde = float(np.linalg.norm(lora_np[-1] - truth_np[-1]))
        b_dir = dir_err(base_np[-1, :2], truth_np[-1, :2])
        l_dir = dir_err(lora_np[-1, :2], truth_np[-1, :2])
        gain = (b_fde - l_fde) / max(b_fde, 1e-6) * 100

        print(f'    Display window: FDE {b_fde:.3f}→{l_fde:.3f}m ({gain:+.1f}%)  '
              f'Dir {b_dir:.1f}°→{l_dir:.1f}°')

        results.append({
            'name': name, 'nf': nf,
            'hist': hist_np, 'base': base_np, 'lora': lora_np, 'truth': truth_np,
            'b_fde': b_fde, 'l_fde': l_fde, 'b_dir': b_dir, 'l_dir': l_dir,
            'gain': gain,
        })

    # Restore gate
    model.ua_pgd.physics_gate.forward = orig_gate

    if not results:
        print('\nNo results!')
        return

    # ── Plot: Overview (2 rows × 3 cols, 3D) ──
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(8 * cols, 7 * rows))
    fig.suptitle(
        'LoRA v5 Ablation — LOW Long Trajectories\n'
        'Baseline: ContextAdapter(AC) + A(stride=2) + C(gate=0.3)\n'
        'Blue=History  Red=Base(AC)  Orange=LoRA(AC+LoRA)  Green=Truth\n'
        'Colored markers = 0.0s / 1.0s / 2.0s / 3.0s / 4.0s',
        fontsize=13, fontweight='bold')
    for i, r in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        title = (f'{r["name"][:22]}\n'
                 f'FDE: {r["b_fde"]:.3f}→{r["l_fde"]:.3f}m ({r["gain"]:+.1f}%)  '
                 f'Dir: {r["b_dir"]:.0f}°→{r["l_dir"]:.0f}°')
        plot_3d(ax, r['hist'], r['base'], r['lora'], r['truth'], title)
    plt.tight_layout(pad=2)
    p3d = OUT_DIR / 'lora_ablation_ac_3d.png'
    fig.savefig(p3d, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {p3d}')
    plt.close()

    # ── Plot: Overview (2 rows × 3 cols, XY) ──
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
    if n == 1:
        axes = np.array([[axes]])
    fig.suptitle(
        'LoRA v5 Ablation — XY Top-Down View\n'
        'Blue=History  Red=Base(AC)  Orange=LoRA(AC+LoRA)  Green=Truth',
        fontsize=13, fontweight='bold')
    for i, r in enumerate(results):
        ax = axes.flat[i] if n > 1 else axes[0, 0]
        title = (f'{r["name"][:22]}\n'
                 f'FDE: {r["b_fde"]:.3f}→{r["l_fde"]:.3f}m ({r["gain"]:+.1f}%)')
        plot_xy(ax, r['hist'], r['base'], r['lora'], r['truth'], title)
    for i in range(n, rows * cols):
        if n > 1:
            axes.flat[i].axis('off')
    plt.tight_layout(pad=2)
    pxy = OUT_DIR / 'lora_ablation_ac_xy.png'
    fig.savefig(pxy, dpi=150, bbox_inches='tight')
    print(f'Saved: {pxy}')
    plt.close()

    # ── Plot: Detailed pages (2 per page, 3D + XY + Step Error each) ──
    for pi in range(0, n, 2):
        pg = results[pi:pi + 2]
        ns = len(pg)
        fig = plt.figure(figsize=(24, 8 * ns))
        fig.suptitle(
            'LoRA v5 Ablation — Detailed View\n'
            'Left=3D  Center=XY  Right=Per-Step Error  '
            'Colored dots = 0.0/1.0/2.0/3.0/4.0s',
            fontsize=12, fontweight='bold')
        for ri, r in enumerate(pg):
            ax3 = fig.add_subplot(ns, 3, ri * 3 + 1, projection='3d')
            title3 = f'{r["name"][:22]}\nFDE: {r["b_fde"]:.3f}→{r["l_fde"]:.3f}m ({r["gain"]:+.1f}%)'
            plot_3d(ax3, r['hist'], r['base'], r['lora'], r['truth'], title3)

            ax_xy = fig.add_subplot(ns, 3, ri * 3 + 2)
            plot_xy(ax_xy, r['hist'], r['base'], r['lora'], r['truth'],
                    f'XY: {r["name"][:22]}')

            ax_e = fig.add_subplot(ns, 3, ri * 3 + 3)
            plot_step_error(ax_e, r['base'], r['lora'], r['truth'],
                          f'Step Error: {r["name"][:22]}')
        plt.tight_layout(pad=2)
        pd = OUT_DIR / f'lora_ablation_ac_detail_p{pi // 2 + 1}.png'
        fig.savefig(pd, dpi=150, bbox_inches='tight')
        print(f'Saved: {pd}')
        plt.close()

    print('\nDone! All figures saved to pic-results/')


if __name__ == '__main__':
    main()
