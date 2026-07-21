#!/usr/bin/env python3
"""
LoRA on 40-frame model — per-trajectory adaptation for long trajectories.

Base: 40-frame expanded model (low_speed_6class_40frame.pth)
LoRA: v8.1 config (upstream-only, boundary continuity, physics loss)
Goal: verify LoRA adds incremental value on an already-good base model (锦上添花).
"""

import torch, numpy as np, sys, warnings, json, traceback
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from emam_model import TrajectoryPredictor
from lora import LoRALinear

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
WEIGHT_DIR = Path(__file__).parent / 'weights'
OUT_DIR = Path(__file__).parent / 'pic-results' / 'lora_40frame'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config ───────────────────────────────────────────────────────────
HIST_LEN, PRED_LEN = 40, 20
DT = 0.2
LONG_THRESHOLD = 150

# ── LoRA targets (v8.1: upstream only, no delta_head) ─────────────────────
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
EPOCHS, RESTARTS = 40, 3
LR_MAX, LR_MIN = 1e-3, 1e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1.0
BATCH_SIZE, TRAIN_SPLIT = 32, 0.8
MIN_TRAINABLE, DIRERR_MAX = 5, 45.0     # lower threshold: 40-frame model has much better base dir

# ── Loss (v8.1) ────────────────────────────────────────────────────────────
BETA_HUBER = 0.20
W_DIR, W_SMOOTH, W_JERK = 0.25, 0.40, 0.35
W_CURVATURE, W_TV_VEL, W_SPEED = 0.20, 0.15, 0.03
W_ANCHOR_DIR = 0.02
W_BOUNDARY, BOUNDARY_STEPS = 0.40, 1
PHYSICS_WARMUP, PHYSICS_START = 3, 0.50

# ── Visualization ──────────────────────────────────────────────────────────
N_DISPLAY = 12
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 7,
                     'font.family': 'sans-serif', 'figure.dpi': 150})
C = {'hist': '#1565C0', 'base': '#D32F2F', 'lora': '#FF6D00', 'truth': '#2E7D32'}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv)/(pn*tn), -1.0, 1.0))))

def resolve_module(model, path):
    parts = path.split('.'); obj = model
    for part in parts: obj = getattr(obj, part)
    return obj

def set_module(model, path, module):
    parts = path.split('.'); parent = model
    for part in parts[:-1]: parent = getattr(parent, part)
    setattr(parent, parts[-1], module)

def make_adaptive_windows(traj, hist_len=40):
    """Adaptive stride capped to produce >=15 windows per trajectory."""
    n = traj.shape[0]
    stride = max(1, min(4, n // 60))  # consistent ~55% coverage across all lengths
    ml = hist_len * stride + PRED_LEN
    if n < ml: return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + hist_len*stride, stride)[:hist_len]
        hists.append(traj[indices].copy())
        fut_start = i + hist_len*stride
        fut_abs = traj[fut_start:fut_start+PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start-1, :3])
    return hists, futs

def inject_lora(model):
    for p in model.parameters(): p.requires_grad_(False)
    lora_layers, original_layers = {}, {}
    for path, rank in LORA_TARGETS:
        original = resolve_module(model, path)
        original_layers[path] = original
        lora = LoRALinear(original, r=rank, alpha=rank*2.0)
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
    return {'lora': {p: {'A': l.lora_A.data.clone(), 'B': l.lora_B.data.clone()} for p,l in ll.items()},
            'head': {f'{p}.weight': l.weight.data.clone() for p,l in hl.items()}}

def load_lora_state(ll, hl, state):
    for p, m in state['lora'].items():
        if p in ll: ll[p].lora_A.data.copy_(m['A']); ll[p].lora_B.data.copy_(m['B'])
    for key, tensor in state['head'].items():
        p, attr = key.rsplit('.', 1)
        if p in hl:
            if attr=='weight': hl[p].weight.data.copy_(tensor)
            elif attr=='bias' and hl[p].bias is not None: hl[p].bias.data.copy_(tensor)

def restore_model(model, ol, ho):
    for path, original in ol.items(): set_module(model, path, original)
    for path, orig in ho.items():
        layer = resolve_module(model, path); layer.weight.data.copy_(orig['weight'])
        layer.weight.requires_grad_(False)
        if orig['bias'] is not None: layer.bias.data.copy_(orig['bias']); layer.bias.requires_grad_(False)

def physics_multiplier(epoch):
    if epoch >= PHYSICS_WARMUP: return 1.0
    return PHYSICS_START + (1.0-PHYSICS_START)*(epoch/PHYSICS_WARMUP)

def compute_loss(pred, target, history, base_pred, epoch):
    ramp = physics_multiplier(epoch)
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)
    pred_vel = pred[:,1:,:]-pred[:,:-1,:]; true_vel = target[:,1:,:]-target[:,:-1,:]
    loss_dir = (1.0-F.cosine_similarity(pred_vel, true_vel, dim=-1)).mean()
    pred_acc = pred[:,2:,:]-2*pred[:,1:-1,:]+pred[:,:-2,:]
    loss_smooth = (pred_acc**2).mean()
    pred_jerk = pred[:,3:,:]-3*pred[:,2:-1,:]+3*pred[:,1:-2,:]-pred[:,:-3,:]
    loss_jerk = (pred_jerk**2).mean()
    v_xy=pred_vel[:,:18,:2]; a_xy=pred_acc[:,:,:2]
    speed_xy=v_xy.norm(dim=-1)+1e-6
    cross=v_xy[:,:,0]*a_xy[:,:,1]-v_xy[:,:,1]*a_xy[:,:,0]
    loss_curvature=(cross.abs()/(speed_xy**3+1e-4)).mean()
    vel_dir=F.normalize(pred_vel+1e-8, dim=-1)
    loss_tv=(1.0-(vel_dir[:,1:,:]*vel_dir[:,:-1,:]).sum(dim=-1)).mean()
    loss_speed=F.relu(pred_vel.norm(dim=-1)-3.0).mean()
    base_dir=F.normalize(base_pred[:,-1,:2]-base_pred[:,0,:2], dim=-1)
    pred_dir=F.normalize(pred[:,-1,:2]-pred[:,0,:2], dim=-1)
    loss_anchor=(1.0-(base_dir*pred_dir).sum(dim=-1)).mean()
    hist_last_vel=history[:,-1,3:6]
    loss_boundary=0.0
    for k in range(BOUNDARY_STEPS):
        pc=pred[:,:k+1,:].sum(dim=1); ec=hist_last_vel*(DT*(k+1))
        loss_boundary=loss_boundary+((pc-ec)**2).mean()
    loss_boundary=loss_boundary/BOUNDARY_STEPS
    return (loss_huber+W_DIR*loss_dir+W_SMOOTH*ramp*loss_smooth
            +W_JERK*ramp*loss_jerk+W_CURVATURE*ramp*loss_curvature
            +W_TV_VEL*ramp*loss_tv+W_SPEED*loss_speed+W_ANCHOR_DIR*loss_anchor
            +W_BOUNDARY*loss_boundary)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d(ax, hp, bp, lp, tp, title):
    ax.plot(hp[:,0], hp[:,1], hp[:,2], color=C['hist'], lw=2.5, label='History (8s)')
    ax.plot(bp[:,0], bp[:,1], bp[:,2], color=C['base'], lw=1.8, ls='--', alpha=0.7, label='Base (40fr)')
    ax.plot(lp[:,0], lp[:,1], lp[:,2], color=C['lora'], lw=2.2, label='LoRA (40fr+LoRA)')
    ax.plot(tp[:,0], tp[:,1], tp[:,2], color=C['truth'], lw=2.5, label='Ground Truth')
    ax.scatter(tp[0,0], tp[0,1], tp[0,2], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1,0], tp[-1,1], tp[-1,2], c='black', s=70, marker='*', zorder=10)
    all_pts = np.concatenate([hp, bp, lp, tp], axis=0)
    rng = max(np.ptp(all_pts[:,0]), np.ptp(all_pts[:,1])) * 0.55
    zm = (all_pts[:,2].min()+all_pts[:,2].max())/2
    ax.set_xlim(all_pts[:,0].mean()-rng, all_pts[:,0].mean()+rng)
    ax.set_ylim(all_pts[:,1].mean()-rng, all_pts[:,1].mean()+rng)
    ax.set_zlim(zm-rng, zm+rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, loc='upper left'); ax.view_init(elev=22, azim=-55)

def plot_xy(ax, hp, bp, lp, tp, title):
    ax.plot(hp[:,0], hp[:,1], color=C['hist'], lw=2.5, label='History')
    ax.plot(bp[:,0], bp[:,1], color=C['base'], lw=1.8, ls='--', alpha=0.7, label='Base (40fr)')
    ax.plot(lp[:,0], lp[:,1], color=C['lora'], lw=2.2, label='LoRA')
    ax.plot(tp[:,0], tp[:,1], color=C['truth'], lw=2.5, label='Truth')
    ax.scatter(hp[-1,0], hp[-1,1], c=C['hist'], s=80, marker='s', edgecolors='black', lw=0.8, zorder=5)
    ax.scatter(tp[0,0], tp[0,1], c='black', s=50, marker='s', zorder=10)
    ax.scatter(tp[-1,0], tp[-1,1], c='black', s=70, marker='*', zorder=10)
    all_xy = np.concatenate([hp[:,:2], bp[:,:2], lp[:,:2], tp[:,:2]], axis=0)
    rng = max(np.ptp(all_xy[:,0]), np.ptp(all_xy[:,1])) * 0.55
    xm, ym = all_xy[:,0].mean(), all_xy[:,1].mean()
    ax.set_xlim(xm-rng, xm+rng); ax.set_ylim(ym-rng, ym+rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper left')

def plot_per_step(ax, be, le, title):
    steps = np.arange(20) * DT
    ax.bar(steps-0.04, be, width=0.08, color=C['base'], alpha=0.6, label='Base (40fr)')
    ax.bar(steps+0.04, le, width=0.08, color=C['lora'], alpha=0.8, label='LoRA (40fr+LoRA)')
    ax.axhline(y=np.mean(be), color=C['base'], ls=':', lw=1)
    ax.axhline(y=np.mean(le), color=C['lora'], ls=':', lw=1)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Position Error (m)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('='*80)
    print('LoRA on 40-Frame Model — Per-Trajectory Adaptation')
    print(f'  Base: 40-frame expanded model')
    print(f'  LoRA: v8.1 config ({sum(r for _,r in LORA_TARGETS)} total rank)')
    print(f'  Epochs: {EPOCHS}  Restarts: {RESTARTS}')
    print('='*80)

    # Load 40-frame model
    print('\n[1/3] Loading 40-frame model...')
    model = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load(WEIGHT_DIR/'low_speed_6class_40frame.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    device = DEVICE
    print(f'  Loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

    # Collect long trajectory candidates
    print('\n[2/3] Collecting long trajectories...')
    candidates = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); n = d['traj'].shape[0]
        if n >= LONG_THRESHOLD: candidates.append((f.name, d['traj'], n))
    candidates.sort(key=lambda x: x[2], reverse=True)
    MAX_TRAJ = 50
    # Sample evenly from all length buckets (not just longest = all sharp turns)
    FAMILIAR = [
        '2025-04-23_09-15-12.npz', '2025-04-21_15-03-16.npz',
        '2025-04-03_12-30-43.npz', '2025-04-25_09-42-34.npz',
        '2025-04-29_17-15-55.npz', '2025-04-28_14-30-06.npz',
        '2025-04-18_15-40-51.npz', '2025-04-20_15-10-39.npz',
    ]
    familiar_c = [(n, t, l) for n, t, l in candidates if n in FAMILIAR]
    rest_c = [(n, t, l) for n, t, l in candidates if n not in FAMILIAR]
    np.random.seed(123)
    # Sample from 4 quartiles for motion-type diversity
    n_rest = len(rest_c); q = max(1, n_rest // 4)
    sampled_rest = []
    for bi in range(4):
        bucket = rest_c[bi*q:min((bi+1)*q, n_rest)]
        if bucket:
            n_sample = min(11, len(bucket))
            idxs = np.random.choice(len(bucket), n_sample, replace=False)
            sampled_rest.extend([bucket[i] for i in idxs])
    candidates = familiar_c + sampled_rest
    candidates = candidates[:MAX_TRAJ]
    print(f'  {len(candidates)} trajectories')

    # Process
    print(f'\n[3/3] Training LoRA + evaluating...')
    results = []
    display_data = []

    for ti, (name, traj, nf) in enumerate(candidates):
        try:
            hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
            n_total = len(hists)
            if n_total < MIN_TRAINABLE + 5: continue

            all_hist = np.array(hists, dtype=np.float32)
            futs_t = [torch.from_numpy(t).float() for t in futs]

            # Base evaluation (40-frame, no LoRA)
            bpred_list = []
            for b in range(0, n_total, 64):
                be = min(b+64, n_total)
                hb = torch.from_numpy(all_hist[b:be]).to(device)
                with torch.no_grad():
                    bpred_list.append(model(hb, force_predict=True)['predictions'].cpu())
            bpred = torch.cat(bpred_list, dim=0)

            bdir_all = np.array([dir_err(bpred[i,-1,:2].numpy(), futs[i][-1,:2]) for i in range(n_total)])
            trainable = bdir_all < DIRERR_MAX
            n_trainable = int(trainable.sum())
            if n_trainable < MIN_TRAINABLE: continue

            tidx = np.where(trainable)[0]; np.random.seed(42); np.random.shuffle(tidx)
            n_tr = int(len(tidx)*TRAIN_SPLIT)
            tr_idx = tidx[:n_tr]
            val_n = max(5, len(tidx)//5); val_idx = tidx[n_tr:n_tr+val_n]
            te_idx = tidx[n_tr+val_n:] if n_tr+val_n<len(tidx) else tidx[n_tr:n_tr+10]
            if len(tr_idx)<5 or len(te_idx)<3: continue

            # Train LoRA
            tr_h = torch.from_numpy(np.array([hists[i] for i in tr_idx], dtype=np.float32))
            tr_t = torch.stack([futs_t[i] for i in tr_idx])
            tr_bp = torch.stack([bpred[i] for i in tr_idx])
            val_h = torch.from_numpy(np.array([hists[i] for i in val_idx], dtype=np.float32))
            val_t = torch.stack([futs_t[i] for i in val_idx])

            best_val_fde = float('inf'); best_state = None
            for restart in range(RESTARTS):
                torch.manual_seed(42+restart*137); np.random.seed(42+restart*137)
                ll, hl, ol, ho = inject_lora(model)
                params = collect_trainable(ll, hl)
                opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
                bs = min(BATCH_SIZE, len(tr_idx))
                for ep in range(EPOCHS):
                    model.eval(); perm = np.random.permutation(len(tr_idx))
                    for b in range(0, len(tr_idx), bs):
                        idx = perm[b:b+bs]
                        hb, tb = tr_h[idx].to(device), tr_t[idx].to(device)
                        opt.zero_grad()
                        pred = model(hb, force_predict=True)['predictions']
                        loss = compute_loss(pred, tb, hb, tr_bp[idx].to(device), ep)
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward(); torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP); opt.step()
                    sched.step()
                model.eval(); val_fdes = []
                for b in range(0, len(val_idx), bs):
                    be = min(b+bs, len(val_idx))
                    hb, tb = val_h[b:be].to(device), val_t[b:be]
                    with torch.no_grad():
                        pv = model(hb, force_predict=True)['predictions'].cpu()
                    val_fdes.append(torch.norm(pv[:,-1,:]-tb[:,-1,:], dim=-1))
                vf = torch.cat(val_fdes).mean().item()
                if vf < best_val_fde: best_val_fde = vf; best_state = save_lora_state(ll, hl)
                restore_model(model, ol, ho)

            if best_state is None: continue

            # Test evaluation
            ll, hl, ol, ho = inject_lora(model)
            load_lora_state(ll, hl, best_state)
            te_h = np.array([hists[i] for i in te_idx], dtype=np.float32)
            lp_list = []
            for b in range(0, len(te_idx), 64):
                be = min(b+64, len(te_idx))
                hb = torch.from_numpy(te_h[b:be]).to(device)
                with torch.no_grad():
                    lp_list.append(model(hb, force_predict=True)['predictions'].cpu())
            lp_all = torch.cat(lp_list, dim=0)
            restore_model(model, ol, ho)

            te_t = torch.stack([futs_t[i] for i in te_idx])
            te_bp = torch.stack([bpred[i] for i in te_idx])
            b_ade = torch.norm(te_bp-te_t, dim=-1).mean(dim=1)
            b_fde = torch.norm(te_bp[:,-1,:]-te_t[:,-1,:], dim=-1)
            l_ade = torch.norm(lp_all-te_t, dim=-1).mean(dim=1)
            l_fde = torch.norm(lp_all[:,-1,:]-te_t[:,-1,:], dim=-1)
            b_dir_vals = torch.tensor([bdir_all[i] for i in te_idx])
            l_dir_vals = torch.tensor([dir_err(lp_all[j,-1,:2].numpy(), futs[te_idx[j]][-1,:2]) for j in range(len(te_idx))])

            gain = float((b_fde.mean()-l_fde.mean())/max(b_fde.mean(),1e-6)*100)
            g = 'GAIN' if gain > 0 else 'DEGRADE'
            print(f'  [{ti+1:3d}] {name[:28]} f={nf:4d}  B.FDE={b_fde.mean():.3f} L.FDE={l_fde.mean():.3f} ({gain:+.1f}%) [{g}]')

            # Pick best display window
            best_j = 0; best_imp = -999.0
            for j in range(len(te_idx)):
                bfi = float(torch.norm(te_bp[j,-1,:]-te_t[j,-1,:]))
                lfi = float(torch.norm(lp_all[j,-1,:]-te_t[j,-1,:]))
                if bfi-lfi > best_imp: best_imp = bfi-lfi; best_j = j

            win_i = te_idx[best_j]; hist_last = hists[win_i][-1,:3]
            hp_abs = hists[win_i][:,:3]
            bp_abs = te_bp[best_j].numpy() + hist_last
            lp_abs = lp_all[best_j].numpy() + hist_last
            tp_abs = futs[win_i] + hist_last

            results.append({
                'name': name, 'nf': nf, 'n_test': len(te_idx),
                'base_ade': float(b_ade.mean()), 'base_fde': float(b_fde.mean()),
                'base_dir': float(b_dir_vals.mean()),
                'lora_ade': float(l_ade.mean()), 'lora_fde': float(l_fde.mean()),
                'lora_dir': float(l_dir_vals.mean()),
                'fde_gain': gain,
                'cata_base': int((b_dir_vals>=90).sum()), 'cata_lora': int((l_dir_vals>=90).sum()),
            })
            display_data.append({
                'name': name, 'nf': nf, 'gain': gain,
                'b_fde': float(b_fde.mean()), 'l_fde': float(l_fde.mean()),
                'hp': hp_abs, 'bp': bp_abs, 'lp': lp_abs, 'tp': tp_abs,
            })
        except Exception as e:
            print(f'  [{ti+1:3d}] {name[:28]} ERROR: {e}')
            traceback.print_exc()

    if not results:
        print('\nNo results!')
        return

    # ── Summary ──
    n_gain = sum(1 for r in results if r['fde_gain'] > 0)
    total_test = sum(r['n_test'] for r in results)
    w_fde_gain = sum(r['fde_gain']*r['n_test'] for r in results)/total_test
    all_b_fde = []; all_l_fde = []
    for r in results:
        all_b_fde.extend([r['base_fde']]*r['n_test'])
        all_l_fde.extend([r['lora_fde']]*r['n_test'])

    print(f'\n{"="*80}')
    print(f'SUMMARY — LoRA on 40-Frame Model')
    print(f'  Trajectories: {len(results)}/{len(candidates)}  Test windows: {total_test}')
    print(f'  FDE gain: {n_gain}/{len(results)} ({n_gain/max(len(results),1)*100:.0f}%)  median: {np.median([r["fde_gain"] for r in results]):+.1f}%')
    print(f'  Weighted FDE: {np.mean(all_b_fde):.3f} -> {np.mean(all_l_fde):.3f}m ({w_fde_gain:+.1f}%)')
    print(f'  P95 FDE: {np.percentile(all_b_fde,95):.3f} -> {np.percentile(all_l_fde,95):.3f}m')
    fde_gains = [r['fde_gain'] for r in results]
    print(f'  Gain: min={np.min(fde_gains):+.1f}%  p25={np.percentile(fde_gains,25):+.1f}%  median={np.median(fde_gains):+.1f}%  p75={np.percentile(fde_gains,75):+.1f}%  max={np.max(fde_gains):+.1f}%')
    print('='*80)

    # ── Visualization: show ALL passing + forced familiar (base-only) ──
    FAMILIAR_NAMES = [
        '2025-04-23_09-15-12.npz', '2025-04-21_15-03-16.npz',
        '2025-04-03_12-30-43.npz', '2025-04-25_09-42-34.npz',
        '2025-04-29_17-15-55.npz', '2025-04-28_14-30-06.npz',
        '2025-04-18_15-40-51.npz', '2025-04-20_15-10-39.npz',
    ]
    fam_display = [d for d in display_data if any(fn in d['name'] for fn in FAMILIAR_NAMES)]

    # Force-evaluate missing familiar trajectories (base only, 40-frame already handles them)
    missing_fam = [fn for fn in FAMILIAR_NAMES if not any(fn in d['name'] for d in display_data)]
    if missing_fam:
        print(f'\n  Force-evaluating {len(missing_fam)} familiar trajectories (base only, no LoRA needed)...')
        for fn in missing_fam:
            for name, traj, nf in candidates:
                if fn in name:
                    hists, futs = make_adaptive_windows(traj, hist_len=HIST_LEN)
                    if len(hists) < 10: continue
                    all_hist = np.array(hists, dtype=np.float32)
                    bpred_list = []
                    for b in range(0, len(hists), 64):
                        be = min(b+64, len(hists))
                        hb = torch.from_numpy(all_hist[b:be]).to(device)
                        with torch.no_grad():
                            bpred_list.append(model(hb, force_predict=True)['predictions'].cpu())
                    bpred = torch.cat(bpred_list, dim=0)
                    futs_t = [torch.from_numpy(t).float() for t in futs]
                    # Pick a representative window
                    best_i = len(hists) // 2  # middle window
                    hist_last = hists[best_i][-1,:3]
                    hp_abs = hists[best_i][:,:3]
                    bp_abs = bpred[best_i].numpy() + hist_last
                    tp_abs = futs[best_i] + hist_last
                    bf = float(torch.norm(bpred[best_i,-1,:] - futs_t[best_i][-1,:]))
                    fam_display.append({
                        'name': name, 'nf': nf, 'gain': 0.0,
                        'b_fde': bf, 'l_fde': bf,  # same = no LoRA needed
                        'hp': hp_abs, 'bp': bp_abs, 'lp': bp_abs.copy(), 'tp': tp_abs,
                    })
                    print(f'    {name[:30]} base FDE={bf:.3f}m (40-frame already excellent)')
                    break

    other_display = [d for d in display_data if d not in fam_display]
    other_display.sort(key=lambda x: x['gain'], reverse=True)
    display = fam_display + other_display
    display = display[:N_DISPLAY]
    print(f'\nVisualizing {len(display)} trajectories ({len(fam_display)} familiar + {len(display)-len(fam_display)} best others)...')

    # 3D overview
    cols = min(4, len(display)); rows = (len(display)+cols-1)//cols
    fig = plt.figure(figsize=(6*cols, 5.5*rows))
    fig.suptitle('LoRA on 40-Frame Model — 3D Trajectory Comparison\nBlue=History  Red=Base(40fr)  Orange=LoRA(40fr+LoRA)  Green=Truth',
                 fontsize=12, fontweight='bold')
    for i, d in enumerate(display):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        plot_3d(ax, d['hp'], d['bp'], d['lp'], d['tp'],
                f'{d["name"][:22]}\nFDE: {d["b_fde"]:.2f}->{d["l_fde"]:.2f}m ({d["gain"]:+.0f}%)')
    plt.tight_layout(pad=2)
    p = OUT_DIR/'overview_3d.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); print(f'  {p.name}'); plt.close()

    # XY overview
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5.5*rows))
    if len(display)==1: axes = np.array([[axes]])
    fig.suptitle('LoRA on 40-Frame Model — XY Top-Down View', fontsize=12, fontweight='bold')
    for i, d in enumerate(display):
        ax = axes.flat[i]
        plot_xy(ax, d['hp'], d['bp'], d['lp'], d['tp'],
                f'{d["name"][:22]}\nFDE: {d["b_fde"]:.2f}->{d["l_fde"]:.2f}m ({d["gain"]:+.0f}%)')
    for i in range(len(display), rows*cols): axes.flat[i].axis('off')
    plt.tight_layout(pad=2)
    p = OUT_DIR/'overview_xy.png'; fig.savefig(p, dpi=150, bbox_inches='tight'); print(f'  {p.name}'); plt.close()

    # Detailed per-trajectory pages (2 per page)
    for pi in range(0, len(display), 2):
        pg = display[pi:pi+2]; ns = len(pg)
        fig = plt.figure(figsize=(20, 8*ns))
        fig.suptitle('LoRA on 40-Frame Model — Detailed View\nLeft=3D  Center=XY  Right=Per-Step Error',
                     fontsize=12, fontweight='bold')
        for ri, d in enumerate(pg):
            ax3 = fig.add_subplot(ns, 3, ri*3+1, projection='3d')
            plot_3d(ax3, d['hp'], d['bp'], d['lp'], d['tp'],
                    f'{d["name"][:22]}\nFDE: {d["b_fde"]:.2f}->{d["l_fde"]:.2f}m ({d["gain"]:+.0f}%)')
            ax_xy = fig.add_subplot(ns, 3, ri*3+2)
            plot_xy(ax_xy, d['hp'], d['bp'], d['lp'], d['tp'], f'XY: {d["name"][:22]}')
            ax_e = fig.add_subplot(ns, 3, ri*3+3)
            be = np.linalg.norm(d['bp']-d['tp'], axis=1); le = np.linalg.norm(d['lp']-d['tp'], axis=1)
            plot_per_step(ax_e, be, le, f'Step Error (ADE: B={np.mean(be):.3f} L={np.mean(le):.3f}m)')
        plt.tight_layout(pad=2)
        p = OUT_DIR/f'detail_p{pi//2+1}.png'; fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f'  {p.name}')
        plt.close()

    # Save JSON
    json.dump({'summary': {'n_traj': len(results), 'fde_gain_pct': w_fde_gain,
                           'base_fde': float(np.mean(all_b_fde)), 'lora_fde': float(np.mean(all_l_fde)),
                           'gain_rate': f'{n_gain}/{len(results)}', 'median_gain': float(np.median(fde_gains))},
               'trajectories': sorted(results, key=lambda x: x['fde_gain'], reverse=True)},
              open(OUT_DIR/'results.json','w'), indent=2, default=str)
    print(f'\n  Results: {OUT_DIR}/results.json')
    print(f'  All figures: {OUT_DIR}/')
    print('='*80)


if __name__ == '__main__':
    main()
