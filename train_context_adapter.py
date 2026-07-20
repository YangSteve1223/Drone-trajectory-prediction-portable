#!/usr/bin/env python3
"""
Train ContextAdapterV2 on real UAV-Flow long trajectories.

The adapter takes a 60-frame (12s) context window and produces a d_model=128
feature vector that enriches the model's last encoded feature. This gives the
4-second decoder window access to 12 seconds of context — critical for detecting
upcoming turns that the 20-frame history alone cannot see.

Training:
  - Data: all windows from trajectories >= 150 frames
  - Input: 60-frame context preceding the 20-frame history window
  - Output: d_model vector added to feat_compress output
  - Loss: MSE between augmented prediction and ground truth
  - Base model: frozen
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
OUT = WEIGHTS_DIR / 'context_adapter_long.pth'

HIST_LEN, PRED_LEN, CTX_LEN = 20, 20, 60
MIN_FRAMES = 150                   # only long trajectories
DIRERR_MAX = 90.0                  # filter catastrophic windows

# Training config
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5


def collect_training_data():
    """Extract (history, context, target) from all long trajectory windows."""
    all_hist, all_ctx, all_target = [], [], []

    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        traj = d['traj']
        n = traj.shape[0]
        if n < MIN_FRAMES:
            continue

        # Slide: context (60 frames) + history (20 frames) + future (20 frames)
        total_needed = CTX_LEN + HIST_LEN + PRED_LEN  # 100 frames
        if n < total_needed:
            continue

        for start in range(0, n - total_needed + 1, 5):  # stride=5 for manageable dataset
            ctx_end = start + CTX_LEN
            hist_end = ctx_end + HIST_LEN
            fut_end = hist_end + PRED_LEN

            ctx_win = traj[start:ctx_end].copy()           # (60, 6)
            hist_win = traj[ctx_end:hist_end].copy()       # (20, 6)
            fut_abs = traj[hist_end:fut_end, :3]
            target = fut_abs - traj[hist_end - 1, :3]      # (20, 3) displacement

            all_hist.append(hist_win)
            all_ctx.append(ctx_win)
            all_target.append(target)

    print(f'  Collected {len(all_hist)} windows from long trajectories')
    return (np.array(all_hist, dtype=np.float32),
            np.array(all_ctx, dtype=np.float32),
            np.array(all_target, dtype=np.float32))


def main():
    print('=' * 80)
    print('ContextAdapterV2 Training on Long UAV-Flow Trajectories')
    print(f'  Context: {CTX_LEN} frames (12s),  History: {HIST_LEN} frames (4s)')
    print(f'  Min trajectory: {MIN_FRAMES} frames,  DirErr filter: <{DIRERR_MAX}°')
    print('=' * 80)

    # Load model
    p = DronePredictor()
    model = p.low; model.eval(); device = p.device
    # Freeze base model
    for param in model.parameters():
        param.requires_grad_(False)

    # Collect data
    print('\n[1/3] Collecting training data...')
    hists, ctxs, targets = collect_training_data()
    if len(hists) < 100:
        print(f'  ERROR: Only {len(hists)} windows! Need >=100.')
        return

    # Filter catastrophic windows
    print('\n  Filtering catastrophic windows...')
    all_h = torch.from_numpy(hists).float()
    all_t = torch.from_numpy(targets).float()
    base_preds = []
    for b in range(0, len(all_h), 128):
        hb = all_h[b:b + 128].to(device)
        with torch.no_grad():
            base_preds.append(model(hb, force_predict=True)['predictions'].cpu())
    base_preds = torch.cat(base_preds, dim=0)

    fde = torch.norm(base_preds[:, -1, :] - all_t[:, -1, :], dim=-1)
    # Keep windows with reasonable base predictions
    good = fde < 10.0  # FDE < 10m (generous filter, keep most)
    hists, ctxs, targets = hists[good.numpy()], ctxs[good.numpy()], targets[good.numpy()]
    print(f'  After filtering: {len(hists)} windows (removed {int((~good).sum())} catastrophic)')

    # Train/val split
    n = len(hists)
    n_tr = int(n * 0.8)
    idx = np.random.RandomState(42).permutation(n)
    tr_idx, val_idx = idx[:n_tr], idx[n_tr:]

    tr_h = torch.from_numpy(hists[tr_idx]).float()
    tr_c = torch.from_numpy(ctxs[tr_idx]).float()
    tr_t = torch.from_numpy(targets[tr_idx]).float()
    val_h = torch.from_numpy(hists[val_idx]).float()
    val_c = torch.from_numpy(ctxs[val_idx]).float()
    val_t = torch.from_numpy(targets[val_idx]).float()

    print(f'  Train: {n_tr},  Val: {len(val_idx)}')

    # Create adapter
    print(f'\n[2/3] Training ContextAdapterV2 ({EPOCHS} epochs)...')
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
            hb = tr_h[bidx].to(device)
            cb = tr_c[bidx].to(device)
            tb = tr_t[bidx].to(device)

            ctx_feat = adapter(cb)  # (B, 128) context feature
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
        for b in range(0, len(val_idx), BATCH_SIZE):
            hb = val_h[b:b + BATCH_SIZE].to(device)
            cb = val_c[b:b + BATCH_SIZE].to(device)
            tb = val_t[b:b + BATCH_SIZE]
            with torch.no_grad():
                ctx_feat = adapter(cb)
                pred = model(hb, force_predict=True, context_injection=ctx_feat)['predictions'].cpu()
            val_fdes.append(torch.norm(pred[:, -1, :] - tb[:, -1, :], dim=-1))
        val_fde = torch.cat(val_fdes).mean().item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  Epoch {ep + 1:2d}/{EPOCHS}:  train_loss={np.mean(ep_losses):.5f}  '
                  f'val_fde={val_fde:.3f}m')

        if val_fde < best_val_fde:
            best_val_fde = val_fde
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}

    # Save
    print(f'\n[3/3] Saving best model (val FDE={best_val_fde:.3f}m)...')
    adapter.load_state_dict(best_state)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT)

    # Quick eval: compare base vs adapter-augmented
    print('\nQuick Comparison (validation set):')
    adapter.eval()
    val_fde_base, val_fde_adapter = [], []
    for b in range(0, len(val_idx), BATCH_SIZE):
        hb = val_h[b:b + BATCH_SIZE].to(device)
        cb = val_c[b:b + BATCH_SIZE].to(device)
        tb = val_t[b:b + BATCH_SIZE]
        with torch.no_grad():
            pred_base = model(hb, force_predict=True)['predictions'].cpu()
            ctx_feat = adapter(cb)
            pred_adpt = model(hb, force_predict=True, context_injection=ctx_feat)['predictions'].cpu()
        val_fde_base.append(torch.norm(pred_base[:, -1, :] - tb[:, -1, :], dim=-1))
        val_fde_adapter.append(torch.norm(pred_adpt[:, -1, :] - tb[:, -1, :], dim=-1))

    fde_base = torch.cat(val_fde_base).mean().item()
    fde_adpt = torch.cat(val_fde_adapter).mean().item()
    gain = (fde_base - fde_adpt) / fde_base * 100
    print(f'  Base FDE:    {fde_base:.3f}m')
    print(f'  Adapter FDE: {fde_adpt:.3f}m  ({gain:+.1f}%)')
    print(f'\n  Weights saved: {OUT}')
    print(f'  File size: {OUT.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
