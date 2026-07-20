#!/usr/bin/env python3
"""
Retrain ContextAdapterV2 jointly with A+C configuration:
  A: stride=2 downsampling (8s context)
  C: gate_scale=0.3 (neural 70%)

The adapter processes 60 frames of FULL-RESOLUTION context (not downsampled)
and injects into the decoder. This gives the model both:
  - Extended temporal insight from 12s context (adapter)
  - Better direction from gate rebalancing (C)
  - Reduced aliasing from downsampled input (A)

Training data: long trajectories only (>=150 frames).
"""

import torch, numpy as np, sys, warnings
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from context_adapter import ContextAdapterV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
WEIGHTS_DIR = Path(__file__).parent / 'weights'
OUT = WEIGHTS_DIR / 'context_adapter_ac.pth'

HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
MIN_FRAMES = 150
STRIDE = 2       # A: downsample history
GATE_SCALE = 0.3  # C: gate scaling

# Training
EPOCHS = 30
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5


def collect_data():
    """Collect (history, context, target) from long trajectories with stride=2."""
    all_hist, all_ctx, all_target = [], [], []
    n_traj = 0

    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        traj = d['traj']
        n = traj.shape[0]
        if n < MIN_FRAMES:
            continue
        n_traj += 1

        # Slide windows with stride=2 for history, full 60-frame context
        hist_span = HIST_LEN * STRIDE  # 40 frames covered by downsampled history
        total_needed = CTX_LEN + hist_span + PRED_LEN
        if n < total_needed:
            continue

        for start in range(0, n - total_needed + 1, 5):
            ctx_end = start + CTX_LEN
            hist_end = ctx_end + hist_span
            fut_end = hist_end + PRED_LEN

            ctx_win = traj[start:ctx_end].copy()           # (60, 6) full res
            # Downsampled history: every STRIDE-th frame
            idx = np.arange(ctx_end, ctx_end + hist_span, STRIDE)[:HIST_LEN]
            hist_win = traj[idx].copy()                     # (20, 6)
            fut_abs = traj[hist_end:fut_end, :3]
            target = fut_abs - traj[hist_end - 1, :3]      # (20, 3)

            all_hist.append(hist_win)
            all_ctx.append(ctx_win)
            all_target.append(target)

    print(f'  {len(all_hist)} windows from {n_traj} long trajectories')
    return (np.array(all_hist, dtype=np.float32),
            np.array(all_ctx, dtype=np.float32),
            np.array(all_target, dtype=np.float32))


def main():
    print('=' * 80)
    print('ContextAdapterV2 Retrain — A+C Compatible')
    print(f'  A: stride={STRIDE} (8s history)')
    print(f'  C: gate_scale={GATE_SCALE} (neural 70%)')
    print(f'  Adapter: {CTX_LEN}f context (12s)')
    print('=' * 80)

    p = DronePredictor()
    model = p.low; model.eval(); device = p.device
    for param in model.parameters():
        param.requires_grad_(False)

    # Apply gate scale during training
    orig_forward = model.ua_pgd.physics_gate.forward
    def scaled_forward(last_encoded, intent_weights, step_encoding):
        gi, ga, gc, gm, gme = orig_forward(last_encoded, intent_weights, step_encoding)
        return gi * GATE_SCALE, ga, gc, gm, gme
    model.ua_pgd.physics_gate.forward = scaled_forward

    print('\n[1/3] Collecting data...')
    hists, ctxs, targets = collect_data()
    if len(hists) < 100:
        print(f'  ERROR: Only {len(hists)} windows!')
        return

    # Filter catastrophic windows
    print('  Filtering...')
    all_h = torch.from_numpy(hists).float()
    all_t = torch.from_numpy(targets).float()
    base_preds = []
    for b in range(0, len(all_h), 128):
        with torch.no_grad():
            base_preds.append(model(all_h[b:b + 128].to(device),
                                    force_predict=True)['predictions'].cpu())
    base_preds = torch.cat(base_preds, dim=0)
    fde = torch.norm(base_preds[:, -1, :] - all_t[:, -1, :], dim=-1)
    good = fde < 10.0
    hists, ctxs, targets = (hists[good.numpy()], ctxs[good.numpy()],
                            targets[good.numpy()])
    print(f'  After filter: {len(hists)} windows')

    # Train/val split
    n = len(hists)
    n_tr = int(n * 0.8)
    idx = np.random.RandomState(42).permutation(n)
    tr_h = torch.from_numpy(hists[idx[:n_tr]]).float()
    tr_c = torch.from_numpy(ctxs[idx[:n_tr]]).float()
    tr_t = torch.from_numpy(targets[idx[:n_tr]]).float()
    val_h = torch.from_numpy(hists[idx[n_tr:]]).float()
    val_c = torch.from_numpy(ctxs[idx[n_tr:]]).float()
    val_t = torch.from_numpy(targets[idx[n_tr:]]).float()
    print(f'  Train: {n_tr},  Val: {n - n_tr}')

    # Train adapter
    print(f'\n[2/3] Training adapter ({EPOCHS} epochs)...')
    adapter = ContextAdapterV2(input_dim=6, context_len=CTX_LEN,
                               d_model=model.d_model, hidden=128).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_val_fde = float('inf')
    best_state = None

    for ep in range(EPOCHS):
        adapter.train()
        perm = np.random.permutation(n_tr)
        ep_losses = []
        for b in range(0, n_tr, BATCH_SIZE):
            bidx = perm[b:b + BATCH_SIZE]
            hb, cb, tb = tr_h[bidx].to(device), tr_c[bidx].to(device), tr_t[bidx].to(device)
            ctx_feat = adapter(cb)
            pred = model(hb, force_predict=True, context_injection=ctx_feat)['predictions']
            loss = F.mse_loss(pred, tb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        scheduler.step()

        # Validation
        adapter.eval()
        val_fdes = []
        for b in range(0, len(val_h), BATCH_SIZE):
            hb, cb, tb = val_h[b:b + BATCH_SIZE].to(device), \
                         val_c[b:b + BATCH_SIZE].to(device), \
                         val_t[b:b + BATCH_SIZE]
            with torch.no_grad():
                ctx_feat = adapter(cb)
                pred = model(hb, force_predict=True,
                            context_injection=ctx_feat)['predictions'].cpu()
            val_fdes.append(torch.norm(pred[:, -1, :] - tb[:, -1, :], dim=-1))
        val_fde = torch.cat(val_fdes).mean().item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  Epoch {ep + 1:2d}/{EPOCHS}: train_loss={np.mean(ep_losses):.5f}  '
                  f'val_fde={val_fde:.3f}m')

        if val_fde < best_val_fde:
            best_val_fde = val_fde
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}

    # Save
    print(f'\n[3/3] Saving (val FDE={best_val_fde:.3f}m)...')
    adapter.load_state_dict(best_state)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT)

    # Restore gate
    model.ua_pgd.physics_gate.forward = orig_forward

    # Quick eval
    adapter.eval()
    val_fde_base, val_fde_adpt = [], []
    for b in range(0, len(val_h), BATCH_SIZE):
        hb = val_h[b:b + BATCH_SIZE].to(device)
        cb = val_c[b:b + BATCH_SIZE].to(device)
        tb = val_t[b:b + BATCH_SIZE]
        with torch.no_grad():
            # Base (with gate scale)
            model.ua_pgd.physics_gate.forward = scaled_forward
            pred_base = model(hb, force_predict=True)['predictions'].cpu()
            # Adapter
            ctx_feat = adapter(cb)
            pred_adpt = model(hb, force_predict=True,
                             context_injection=ctx_feat)['predictions'].cpu()
            model.ua_pgd.physics_gate.forward = orig_forward
        val_fde_base.append(torch.norm(pred_base[:, -1, :] - tb[:, -1, :], dim=-1))
        val_fde_adpt.append(torch.norm(pred_adpt[:, -1, :] - tb[:, -1, :], dim=-1))

    fde_base = torch.cat(val_fde_base).mean().item()
    fde_adpt = torch.cat(val_fde_adpt).mean().item()
    print(f'\n  A+C FDE:     {fde_base:.3f}m')
    print(f'  A+C+Adapter: {fde_adpt:.3f}m  ({(fde_base-fde_adpt)/fde_base*100:+.1f}%)')
    print(f'  Saved: {OUT} ({OUT.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
