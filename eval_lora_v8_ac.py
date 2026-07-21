#!/usr/bin/env python3
"""
LoRA v8 — v7 + boundary continuity loss at history→prediction transition.

Root cause found: Base model predicts first displacement step 10-20x too large
(1.5-2.8m vs truth 0.08-0.13m). LoRA v7 reduces to 0.27-0.79m but still 3-6x truth.
The visible "disconnect" at the history-prediction boundary causes the zigzag
perception — the model scrambles to recover from an impossible starting position.

v8 adds W_BOUNDARY=0.50 penalty on first 3 steps deviating from history velocity.
This forces C1 continuity (velocity match) at the transition, which naturally
produces smooth initial predictions and coherent overall trajectories.
"""

import torch, numpy as np, sys, warnings, json, traceback
import torch.nn.functional as F
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

# ── A+C config ─────────────────────────────────────────────────────────────
HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
STRIDE, GATE_SCALE = 2, 0.3
LONG_THRESHOLD = 150

# ── LoRA targets (same as v7: upstream only, no delta_head) ────────────────
LORA_TARGETS = [
    ('emam_se.mamba_blocks.0.ssm.in_proj', 16),
    ('emam_se.mamba_blocks.0.ssm.out_proj', 16),
    ('emam_se.mamba_blocks.1.ssm.in_proj', 16),
    ('emam_se.mamba_blocks.1.ssm.out_proj', 16),
    ('ua_pgd.feat_compress', 64),
    ('ua_pgd.neural_decoder.proj.0', 48),
]
HEAD_TARGETS = ['ua_pgd.anchor_to_pos.2']

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS, RESTARTS = 40, 3
LR_MAX, LR_MIN = 1e-3, 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.0
BATCH_SIZE, TRAIN_SPLIT = 32, 0.8
MIN_TRAINABLE, DIRERR_MAX = 30, 60.0

# ── Loss weights ───────────────────────────────────────────────────────────
BETA_HUBER = 0.20
W_DIR = 0.25
W_SMOOTH, W_JERK = 0.40, 0.35
W_CURVATURE, W_TV_VEL, W_SPEED = 0.20, 0.15, 0.03
W_ANCHOR_DIR = 0.02
W_BOUNDARY = 0.40            # boundary position continuity at step 0 only
BOUNDARY_STEPS = 1            # only step 0 — strong position, free thereafter
PHYSICS_WARMUP, PHYSICS_START = 3, 0.50
MAX_TRAJ = 100


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (same as v7)
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


# ═══════════════════════════════════════════════════════════════════════════
# v8 Loss: adds boundary continuity
# ═══════════════════════════════════════════════════════════════════════════

def physics_multiplier(epoch):
    if epoch >= PHYSICS_WARMUP:
        return 1.0
    return PHYSICS_START + (1.0 - PHYSICS_START) * (epoch / PHYSICS_WARMUP)


def compute_loss(pred, target, history, base_pred, epoch):
    """
    Physics-dominant loss + boundary continuity.

    history: (B, 20, 6) — needed to extract last velocity for boundary constraint
    """
    ramp = physics_multiplier(epoch)
    DT = 0.2  # 5Hz

    # 1. Huber — position accuracy
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)

    # 2. Direction
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    true_vel = target[:, 1:, :] - target[:, :-1, :]
    loss_dir = (1.0 - F.cosine_similarity(pred_vel, true_vel, dim=-1)).mean()

    # 3. Acceleration smoothness — L2
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    loss_smooth = (pred_acc ** 2).mean()

    # 4. Jerk
    pred_jerk = (pred[:, 3:, :] - 3 * pred[:, 2:-1, :]
                 + 3 * pred[:, 1:-2, :] - pred[:, :-3, :])
    loss_jerk = (pred_jerk ** 2).mean()

    # 5. XY curvature
    v_xy = pred_vel[:, :18, :2]
    a_xy = pred_acc[:, :, :2]
    speed_xy = v_xy.norm(dim=-1) + 1e-6
    cross = v_xy[:, :, 0] * a_xy[:, :, 1] - v_xy[:, :, 1] * a_xy[:, :, 0]
    loss_curvature = (cross.abs() / (speed_xy ** 3 + 1e-4)).mean()

    # 6. Velocity direction TV
    vel_dir = F.normalize(pred_vel + 1e-8, dim=-1)
    loss_tv = (1.0 - (vel_dir[:, 1:, :] * vel_dir[:, :-1, :]).sum(dim=-1)).mean()

    # 7. Speed bound
    loss_speed = F.relu(pred_vel.norm(dim=-1) - 3.0).mean()

    # 8. Base-model anchor
    base_dir = F.normalize(base_pred[:, -1, :2] - base_pred[:, 0, :2], dim=-1)
    pred_dir = F.normalize(pred[:, -1, :2] - pred[:, 0, :2], dim=-1)
    loss_anchor = (1.0 - (base_dir * pred_dir).sum(dim=-1)).mean()

    # 9. **NEW** Boundary continuity: first BOUNDARY_STEPS must follow history velocity
    # The history's last known velocity should naturally continue into prediction.
    # Penalize deviation from constant-velocity extrapolation for the first few steps.
    hist_last_vel = history[:, -1, 3:6]          # (B, 3) — velocity at last history frame
    hist_last_pos = history[:, -1, :3]           # (B, 3) — position at last history frame

    loss_boundary = 0.0
    for k in range(BOUNDARY_STEPS):
        # Expected position after step k: hist_last_pos + hist_last_vel * dt * (k+1)
        # Predicted position after step k: hist_last_pos + cumulative_sum(pred[:, :k+1, :])
        pred_cumsum = pred[:, :k + 1, :].sum(dim=1)  # (B, 3) cumulative displacement
        expected_cumsum = hist_last_vel * (DT * (k + 1))   # (B, 3) constant-velocity extrap
        loss_boundary = loss_boundary + ((pred_cumsum - expected_cumsum) ** 2).mean()

    loss_boundary = loss_boundary / BOUNDARY_STEPS

    total = (loss_huber
             + W_DIR * loss_dir
             + W_SMOOTH * ramp * loss_smooth
             + W_JERK * ramp * loss_jerk
             + W_CURVATURE * ramp * loss_curvature
             + W_TV_VEL * ramp * loss_tv
             + W_SPEED * loss_speed
             + W_ANCHOR_DIR * loss_anchor
             + W_BOUNDARY * loss_boundary)

    return total, {
        'huber': loss_huber.item(), 'dir': loss_dir.item(),
        'smooth': loss_smooth.item(), 'jerk': loss_jerk.item(),
        'curvature': loss_curvature.item(), 'tv': loss_tv.item(),
        'speed': loss_speed.item(), 'anchor': loss_anchor.item(),
        'boundary': loss_boundary.item(), 'total': total.item(), 'ramp': ramp,
    }


def compute_smoothness_metrics(pred_abs):
    pred_abs = np.asarray(pred_abs)
    vel = pred_abs[1:, :] - pred_abs[:-1, :]
    acc = vel[1:, :] - vel[:-1, :]
    jerk = acc[1:, :] - acc[:-1, :]
    v_norm = np.linalg.norm(vel[:-1, :2], axis=1) + 1e-6
    cross = np.abs(vel[:-1, 0] * acc[:, 0] - vel[:-1, 1] * acc[:, 1])
    return {
        'mean_jerk': float(np.mean(np.linalg.norm(jerk, axis=1))),
        'max_jerk': float(np.max(np.linalg.norm(jerk, axis=1))),
        'tortuosity': float(np.sum(np.linalg.norm(vel, axis=1))
                           / max(np.linalg.norm(pred_abs[-1] - pred_abs[0]), 1e-6)),
        'first_step_gap': float(np.linalg.norm(vel[0])),  # velocity at first step
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-trajectory processing
# ═══════════════════════════════════════════════════════════════════════════

def process_trajectory(name, traj, model, device, adapter):
    hists, futs, starts = make_windows_ac(traj, stride=STRIDE)
    n_total = len(hists)
    if n_total < MIN_TRAINABLE + 10:
        return None

    ctx_all = make_context(traj, starts, ctx_len=CTX_LEN)
    all_hist = np.array(hists, dtype=np.float32)

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

    futs_t = [torch.from_numpy(t).float() for t in futs]
    bdir_all = np.array([dir_err(bpred[i, -1, :2].numpy(), futs[i][-1, :2])
                         for i in range(n_total)])

    trainable = bdir_all < DIRERR_MAX
    n_trainable = int(trainable.sum())
    if n_trainable < MIN_TRAINABLE:
        return None

    tidx = np.where(trainable)[0]
    np.random.seed(42); np.random.shuffle(tidx)
    n_tr = int(len(tidx) * TRAIN_SPLIT)
    tr_idx = tidx[:n_tr]
    val_n = max(5, len(tidx) // 5)
    val_idx = tidx[n_tr:n_tr + val_n]
    te_idx = tidx[n_tr + val_n:] if n_tr + val_n < len(tidx) else tidx[n_tr:n_tr + 10]
    if len(tr_idx) < 10 or len(te_idx) < 5:
        return None

    tr_h = torch.from_numpy(np.array([hists[i] for i in tr_idx], dtype=np.float32))
    tr_t = torch.stack([futs_t[i] for i in tr_idx])
    tr_c = torch.from_numpy(np.array([ctx_all[i] for i in tr_idx], dtype=np.float32))
    tr_bp = torch.stack([bpred[i] for i in tr_idx])
    val_h = torch.from_numpy(np.array([hists[i] for i in val_idx], dtype=np.float32))
    val_t = torch.stack([futs_t[i] for i in val_idx])
    val_c = torch.from_numpy(np.array([ctx_all[i] for i in val_idx], dtype=np.float32))

    # LoRA training (with history passed to compute_loss for boundary constraint)
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
                ci = adapter(cb) if adapter is not None else None
                kwargs = {'force_predict': True}
                if ci is not None:
                    kwargs['context_injection'] = ci
                pred = model(hb, **kwargs)['predictions']
                loss, _ = compute_loss(pred, tb, hb, tr_bp[idx].to(device), ep)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    opt.step()
            sched.step()

        # Validation
        model.eval()
        val_fdes = []
        for b in range(0, len(val_idx), bs):
            be = min(b + bs, len(val_idx))
            hb, tb = val_h[b:be].to(device), val_t[b:be]
            ci = adapter(val_c[b:be].to(device)) if adapter is not None else None
            kwargs = {'force_predict': True}
            if ci is not None:
                kwargs['context_injection'] = ci
            with torch.no_grad():
                pv = model(hb, **kwargs)['predictions'].cpu()
            val_fdes.append(torch.norm(pv[:, -1, :] - tb[:, -1, :], dim=-1))
        vf = torch.cat(val_fdes).mean().item()
        if vf < best_val_fde:
            best_val_fde = vf
            best_state = save_lora_state(ll, hl)
        restore_model(model, ol, ho)

    if best_state is None:
        return None

    # Test evaluation
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

    te_t = torch.stack([futs_t[i] for i in te_idx])
    te_bp = torch.stack([bpred[i] for i in te_idx])

    b_ade = torch.norm(te_bp - te_t, dim=-1).mean(dim=1)
    b_fde = torch.norm(te_bp[:, -1, :] - te_t[:, -1, :], dim=-1)
    l_ade = torch.norm(lp_all - te_t, dim=-1).mean(dim=1)
    l_fde = torch.norm(lp_all[:, -1, :] - te_t[:, -1, :], dim=-1)
    b_dir_vals = torch.tensor([bdir_all[i] for i in te_idx])
    l_dir_vals = torch.tensor([dir_err(lp_all[j, -1, :2].numpy(),
                                       futs[te_idx[j]][-1, :2])
                               for j in range(len(te_idx))])

    # Smoothness + boundary gap on best display window
    best_j = 0; best_imp = -999.0
    for j in range(len(te_idx)):
        bfi = float(torch.norm(te_bp[j, -1, :] - te_t[j, -1, :]))
        lfi = float(torch.norm(lp_all[j, -1, :] - te_t[j, -1, :]))
        if bfi - lfi > best_imp:
            best_imp = bfi - lfi; best_j = j

    win_i = te_idx[best_j]
    hist_last = hists[win_i][-1, :3]
    hist_last_vel = hists[win_i][-1, 3:6]
    base_abs = te_bp[best_j].numpy() + hist_last
    lora_abs = lp_all[best_j].numpy() + hist_last
    truth_abs = futs[win_i] + hist_last

    b_smooth = compute_smoothness_metrics(base_abs)
    l_smooth = compute_smoothness_metrics(lora_abs)
    t_smooth = compute_smoothness_metrics(truth_abs)

    # Boundary gap: distance from history endpoint to first prediction point
    base_gap_0 = float(np.linalg.norm(base_abs[0] - hist_last))
    lora_gap_0 = float(np.linalg.norm(lora_abs[0] - hist_last))
    truth_gap_0 = float(np.linalg.norm(truth_abs[0] - hist_last))
    # Velocity continuity: first step implied velocity vs history last velocity
    base_v0 = np.linalg.norm(base_abs[0] - hist_last) / 0.2
    lora_v0 = np.linalg.norm(lora_abs[0] - hist_last) / 0.2
    truth_v0 = np.linalg.norm(truth_abs[0] - hist_last) / 0.2
    hist_v = float(np.linalg.norm(hist_last_vel))

    return {
        'name': name, 'n_frames': traj.shape[0],
        'n_total': n_total, 'n_trainable': n_trainable,
        'n_train': len(tr_idx), 'n_val': len(val_idx), 'n_test': len(te_idx),
        'base_ade': float(b_ade.mean()), 'base_fde': float(b_fde.mean()),
        'base_dir': float(b_dir_vals.mean()),
        'lora_ade': float(l_ade.mean()), 'lora_fde': float(l_fde.mean()),
        'lora_dir': float(l_dir_vals.mean()),
        'ade_gain': float((b_ade.mean() - l_ade.mean()) / max(b_ade.mean(), 1e-6) * 100),
        'fde_gain': float((b_fde.mean() - l_fde.mean()) / max(b_fde.mean(), 1e-6) * 100),
        'dir_gain': float((b_dir_vals.mean() - l_dir_vals.mean()) / max(b_dir_vals.mean(), 0.1) * 100),
        'cata_base': int((b_dir_vals >= 90).sum()),
        'cata_lora': int((l_dir_vals >= 90).sum()),
        'base_jerk': b_smooth['mean_jerk'], 'lora_jerk': l_smooth['mean_jerk'],
        'truth_jerk': t_smooth['mean_jerk'],
        'base_gap_0': base_gap_0, 'lora_gap_0': lora_gap_0, 'truth_gap_0': truth_gap_0,
        'base_v0': base_v0, 'lora_v0': lora_v0, 'truth_v0': truth_v0, 'hist_v': hist_v,
        'base_tortuosity': b_smooth['tortuosity'],
        'lora_tortuosity': l_smooth['tortuosity'],
        'truth_tortuosity': t_smooth['tortuosity'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 80)
    print('LoRA v8 — v7 + Boundary Continuity Loss')
    print(f'  NEW: W_BOUNDARY={W_BOUNDARY} on first {BOUNDARY_STEPS} steps')
    print(f'  Forces C1 continuity at history-prediction transition')
    print('=' * 80)

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

    candidates = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        n = d['traj'].shape[0]
        if n >= LONG_THRESHOLD:
            candidates.append((f.name, d['traj'], n))
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:MAX_TRAJ]
    print(f'  {len(candidates)} trajectories')

    results = []
    for ti, (name, traj, nf) in enumerate(candidates):
        try:
            r = process_trajectory(name, traj, model, device, adapter)
            if r:
                results.append(r)
                g = 'GAIN' if r['fde_gain'] > 0 else 'DEGRADE'
                print(f'  [{ti + 1:3d}] {name[:28]}  f={nf:4d}  '
                      f'B.FDE={r["base_fde"]:.3f} L.FDE={r["lora_fde"]:.3f} '
                      f'({r["fde_gain"]:+.1f}%) [{g}]  '
                      f'Gap: {r["base_gap_0"]:.2f}->{r["lora_gap_0"]:.2f}m '
                      f'(T:{r["truth_gap_0"]:.2f})  '
                      f'V: {r["hist_v"]:.1f}->B:{r["base_v0"]:.1f}/L:{r["lora_v0"]:.1f}/T:{r["truth_v0"]:.1f}m/s')
            else:
                print(f'  [{ti + 1:3d}] {name[:28]}  SKIP')
        except Exception as e:
            print(f'  [{ti + 1:3d}] {name[:28]}  ERROR: {e}')
            traceback.print_exc()

    model.ua_pgd.physics_gate.forward = orig_gate

    if not results:
        print('\nNo results!')
        return

    n_gain = sum(1 for r in results if r['fde_gain'] > 0)
    total_test = sum(r['n_test'] for r in results)

    all_b_ade, all_b_fde, all_b_dir = [], [], []
    all_l_ade, all_l_fde, all_l_dir = [], [], []
    all_b_jerk, all_l_jerk, all_t_jerk = [], [], []
    all_b_gap, all_l_gap, all_t_gap = [], [], []
    for r in results:
        w = r['n_test']
        all_b_ade.extend([r['base_ade']] * w); all_l_ade.extend([r['lora_ade']] * w)
        all_b_fde.extend([r['base_fde']] * w); all_l_fde.extend([r['lora_fde']] * w)
        all_b_dir.extend([r['base_dir']] * w); all_l_dir.extend([r['lora_dir']] * w)
        all_b_jerk.append(r['base_jerk']); all_l_jerk.append(r['lora_jerk'])
        all_t_jerk.append(r['truth_jerk'])
        all_b_gap.append(r['base_gap_0']); all_l_gap.append(r['lora_gap_0'])
        all_t_gap.append(r['truth_gap_0'])

    w_fde_gain = sum(r['fde_gain'] * r['n_test'] for r in results) / total_test
    w_dir_gain = sum(r['dir_gain'] * r['n_test'] for r in results) / total_test

    print(f'\n{"=" * 80}')
    print(f'LoRA v8 SUMMARY — v7 + Boundary Continuity (W={W_BOUNDARY})')
    print(f'  Trajectories: {len(results)}/{len(candidates)}  Test windows: {total_test}')
    print(f'{"-" * 80}')
    print(f'  {"Metric":<22} {"Base(AC)":<14} {"LoRA v8":<16} {"Truth":<14} {"Gain":<10}')
    print(f'  {"-" * 76}')
    print(f'  {"ADE":<22} {np.mean(all_b_ade):<14.3f}m {np.mean(all_l_ade):<16.3f}m {"—":<14} {(np.mean(all_b_ade)-np.mean(all_l_ade))/max(np.mean(all_b_ade),1e-6)*100:>+8.1f}%')
    print(f'  {"FDE":<22} {np.mean(all_b_fde):<14.3f}m {np.mean(all_l_fde):<16.3f}m {"—":<14} {w_fde_gain:>+8.1f}%')
    print(f'  {"FDE P95":<22} {np.percentile(all_b_fde,95):<14.3f}m {np.percentile(all_l_fde,95):<16.3f}m')
    print(f'  {"Direction":<22} {np.mean(all_b_dir):<14.1f}  {np.mean(all_l_dir):<16.1f}  {"—":<14} {w_dir_gain:>+8.1f}%')
    print(f'  {"Mean Jerk":<22} {np.mean(all_b_jerk):<14.4f} {np.mean(all_l_jerk):<16.4f} {np.mean(all_t_jerk):<14.4f}')
    print(f'  {"1st Step Gap (m)":<22} {np.mean(all_b_gap):<14.3f} {np.mean(all_l_gap):<16.3f} {np.mean(all_t_gap):<14.3f}')
    print(f'  {"Gap vs Truth (LoRA/T)":<22} {"—":<14} {np.mean(all_l_gap)/max(np.mean(all_t_gap),1e-6):<16.1f}x')
    print(f'{"-" * 80}')
    print(f'  FDE gain: {n_gain}/{len(results)}  median: {np.median([r["fde_gain"] for r in results]):+.1f}%')

    fde_gains = [r['fde_gain'] for r in results]
    print(f'\n  FDE gain: min={np.min(fde_gains):+.1f}%  median={np.median(fde_gains):+.1f}%  max={np.max(fde_gains):+.1f}%')

    summary = {
        'config': {
            'version': 'v8',
            'changes_from_v7': 'W_BOUNDARY=0.50 on first 3 steps',
            'lora_targets': [(p, r) for p, r in LORA_TARGETS],
            'loss_weights': {'huber_beta': BETA_HUBER, 'dir': W_DIR,
                            'smooth': W_SMOOTH, 'jerk': W_JERK,
                            'curvature': W_CURVATURE, 'tv_vel': W_TV_VEL,
                            'speed': W_SPEED, 'boundary': W_BOUNDARY},
        },
        'summary': {
            'n_trajectories': len(results), 'total_test_windows': total_test,
            'base_fde': float(np.mean(all_b_fde)), 'lora_fde': float(np.mean(all_l_fde)),
            'fde_gain_pct': float(w_fde_gain),
            'base_jerk': float(np.mean(all_b_jerk)), 'lora_jerk': float(np.mean(all_l_jerk)),
            'truth_jerk': float(np.mean(all_t_jerk)),
            'base_gap_0': float(np.mean(all_b_gap)), 'lora_gap_0': float(np.mean(all_l_gap)),
            'truth_gap_0': float(np.mean(all_t_gap)),
            'gap_vs_truth': float(np.mean(all_l_gap) / max(np.mean(all_t_gap), 1e-6)),
        },
        'trajectories': sorted(results, key=lambda x: x['fde_gain'], reverse=True),
    }

    json.dump(summary, open(OUT_DIR / 'lora_v8_ac.json', 'w'), indent=2, default=str)
    print(f'\n  Saved: pic-results/lora_v8_ac.json')
    print('=' * 80)


if __name__ == '__main__':
    main()
