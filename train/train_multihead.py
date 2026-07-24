#!/usr/bin/env python3
"""
Multi-Hypothesis Decoder Training with Winner-Takes-All (WTA) Loss.

Features:
  - tqdm progress bars for real-time monitoring
  - Checkpoint resume: auto-saves each epoch, --resume to continue
  - Per-epoch validation with per-intent breakdown
  - Training log saved as JSON for external monitoring

Usage:
  python train_multihead.py --model high --K 5 --epochs 10 --batch_size 64
  python train_multihead.py --model high --K 5 --epochs 10 --resume   # continue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, argparse, json, time, warnings, os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder
from utils.fast_data_loader import FastWindowDataset

# tqdm with fallback
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    p = argparse.ArgumentParser(description='Train Multi-Hypothesis Decoder')
    p.add_argument('--model', default='high', choices=['low', 'high'])
    p.add_argument('--K', type=int, default=5, help='Number of hypotheses')
    p.add_argument('--epochs', type=int, default=10, help='Total epochs (including resumed)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--data_root', default='../SimCruise', help='Training data root')
    p.add_argument('--device', default='cuda')
    p.add_argument('--noise_std', type=float, default=0.02,
                   help='Noise std for head initialization')
    p.add_argument('--resume', action='store_true',
                   help='Resume from latest checkpoint')
    p.add_argument('--checkpoint_dir', default='weights/checkpoints',
                   help='Checkpoint save directory')
    p.add_argument('--val_batches', type=int, default=500,
                   help='Number of batches for quick validation (0 = full val set)')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Training & Evaluation
# ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, epoch, pbar_position=0):
    """Train one epoch with WTA loss + tqdm progress bar."""
    model.train()
    total_loss = 0.0; total_disp = 0.0; total_conf = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}', unit='batch',
                leave=False, position=pbar_position, disable=not HAS_TQDM)

    for hist_batch, target_batch, intent_batch in pbar:
        hist_batch = hist_batch.to(device)
        target_batch = target_batch.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            h_norm = model._normalize(hist_batch)
            enc = model.emam_se(h_norm)
            dtp_out = model.ia_dtp(enc, historical_trajectory=h_norm)

        mh_out = model.ua_pgd.forward_multi_head(
            encoded_feat=enc,
            global_anchor=dtp_out['global_anchor'],
            historical_trajectory=h_norm,
            intent_weights=dtp_out['intent_weights'],
        )

        last_pos = hist_batch[:, -1:, :3]
        target_rel = target_batch[:, :, :3] - last_pos
        scale_pos = model._get_scale_pos()
        preds_norm = mh_out['all_predictions'] / scale_pos
        target_norm = target_rel / scale_pos

        loss_dict = MultiHeadNeuralDecoder.compute_wta_loss(
            preds_norm, mh_out['all_logvars'],
            mh_out['confidences'], target_norm
        )

        loss = loss_dict['total_wta_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.ua_pgd.neural_decoder.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_disp += loss_dict['wta_disp_loss'].item()
        total_conf += loss_dict['conf_loss'].item()
        n_batches += 1

        # Update tqdm postfix every 50 batches
        if n_batches % 50 == 0 and HAS_TQDM:
            pbar.set_postfix({
                'loss': f'{total_loss/n_batches:.4f}',
                'disp': f'{total_disp/n_batches:.4f}',
                'conf': f'{total_conf/n_batches:.4f}',
            })

    return {
        'loss': total_loss / max(n_batches, 1),
        'disp_loss': total_disp / max(n_batches, 1),
        'conf_loss': total_conf / max(n_batches, 1),
        'n_batches': n_batches,
    }


@torch.no_grad()
def evaluate_mh(model, loader, device, max_batches=0):
    """Evaluate minADE_K / minFDE_K. max_batches=0 means full validation set."""
    model.eval()
    all_min_ade = []; all_min_fde = []
    all_single_ade = []; all_single_fde = []
    per_intent = defaultdict(lambda: {'min_ade': [], 'min_fde': [], 'count': 0})
    n_batches = 0

    pbar = tqdm(loader, desc='Validating', unit='batch',
                leave=False, disable=not HAS_TQDM)

    for hist_batch, target_batch, intent_batch in pbar:
        hist_batch = hist_batch.to(device)
        target_batch = target_batch.to(device)

        h_norm = model._normalize(hist_batch)
        enc = model.emam_se(h_norm)
        dtp_out = model.ia_dtp(enc, historical_trajectory=h_norm)

        mh_out = model.ua_pgd.forward_multi_head(
            encoded_feat=enc,
            global_anchor=dtp_out['global_anchor'],
            historical_trajectory=h_norm,
            intent_weights=dtp_out['intent_weights'],
        )

        all_preds = mh_out['all_predictions']
        best_pred = mh_out['predictions']

        last_pos = hist_batch[:, -1:, :3]
        target_rel = target_batch[:, :, :3] - last_pos

        metrics = MultiHeadNeuralDecoder.compute_minade_fde(all_preds, target_rel)
        all_min_ade.extend(metrics['min_ade'].cpu().tolist())
        all_min_fde.extend(metrics['min_fde'].cpu().tolist())

        step_errs_single = torch.norm(best_pred - target_rel, dim=-1)
        all_single_ade.extend(step_errs_single.mean(dim=1).cpu().tolist())
        all_single_fde.extend(step_errs_single[:, -1].cpu().tolist())

        for i in range(hist_batch.shape[0]):
            intent = intent_batch[i].item()
            per_intent[intent]['min_ade'].append(metrics['min_ade'][i].item())
            per_intent[intent]['min_fde'].append(metrics['min_fde'][i].item())
            per_intent[intent]['count'] += 1

        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return {
        'min_ade': np.array(all_min_ade),
        'min_fde': np.array(all_min_fde),
        'single_ade': np.array(all_single_ade),
        'single_fde': np.array(all_single_fde),
        'per_intent': dict(per_intent),
        'n_samples': len(all_min_ade),
    }


# ──────────────────────────────────────────────────────────────────────
#  Checkpoint
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, best_fde, history, args, ckpt_dir):
    """Save full training state for resume."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f'{args.model}_K{args.K}_epoch{epoch:03d}.pth'
    torch.save({
        'epoch': epoch,
        'multi_decoder_state': model.ua_pgd.neural_decoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_min_fde': best_fde,
        'history': history,
        'args': vars(args),
    }, ckpt_path)

    # Also save as "latest" for easy resume
    latest_path = ckpt_dir / f'{args.model}_K{args.K}_latest.pth'
    torch.save({
        'epoch': epoch,
        'multi_decoder_state': model.ua_pgd.neural_decoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_min_fde': best_fde,
        'history': history,
        'args': vars(args),
    }, latest_path)

    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, args, ckpt_dir):
    """Load latest checkpoint. Returns (start_epoch, best_fde, history) or (0, inf, [])."""
    latest_path = Path(ckpt_dir) / f'{args.model}_K{args.K}_latest.pth'
    if not latest_path.exists():
        print('  No checkpoint found, starting fresh.')
        return 0, float('inf'), []

    print(f'  Loading checkpoint: {latest_path}')
    ckpt = torch.load(latest_path, map_location=args.device, weights_only=False)

    # Load multi-decoder weights
    model.ua_pgd.neural_decoder.load_state_dict(ckpt['multi_decoder_state'])

    # Load optimizer & scheduler
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])

    start_epoch = ckpt['epoch'] + 1  # next epoch
    best_fde = ckpt.get('best_min_fde', float('inf'))
    history = ckpt.get('history', [])

    print(f'  Resumed from epoch {ckpt["epoch"]}, best minFDE={best_fde:.4f}m')
    return start_epoch, best_fde, history


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

INTENT_NAMES_HIGH = {0: 'STRAIGHT', 1: 'TURN_L', 2: 'TURN_R', 3: 'DESCEND'}
INTENT_NAMES_LOW = {0: 'STRAIGHT', 1: 'TURN_L', 2: 'TURN_R', 3: 'ASCEND', 4: 'DESC', 5: 'HOVER'}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    intent_names = INTENT_NAMES_LOW if args.model == 'low' else INTENT_NAMES_HIGH
    n_classes = 6 if args.model == 'low' else 4
    label_remap = None if args.model == 'low' else {4: 3}
    data_root = '../UAV-Flow-pure' if args.model == 'low' else args.data_root
    weight_path = Path(__file__).resolve().parents[1] / 'weights' / (
        'low_speed_6class.pth' if args.model == 'low' else 'high_speed_4class.pth')

    print(f'{"="*60}')
    print(f'  Multi-Hypothesis Training — {args.model.upper()} model, K={args.K}')
    print(f'  Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}')
    print(f'  Resume: {args.resume}  |  tqdm: {HAS_TQDM}')
    print(f'{"="*60}')

    # ── Build model ───────────────────────────────────────────────
    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=n_classes,
        use_trigger=True, trigger_mode='simple',
    ).to(device)

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    model._norm_input = False
    model._get_scale_pos = lambda: 100.0

    def _normalize(hist):
        scale = hist.new_tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        return hist / scale.unsqueeze(0).unsqueeze(0)
    model._normalize = _normalize

    # ── Replace decoder ───────────────────────────────────────────
    multi_decoder = model.ua_pgd.replace_with_multi_head(K=args.K, noise_std=args.noise_std)
    multi_decoder = multi_decoder.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = 'neural_decoder' in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

    # ── Optimizer & Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume or start fresh ─────────────────────────────────────
    ckpt_dir = Path(__file__).resolve().parents[1] / args.checkpoint_dir

    if args.resume:
        start_epoch, best_min_fde, history = load_checkpoint(
            model, optimizer, scheduler, args, ckpt_dir)
    else:
        start_epoch = 0
        best_min_fde = float('inf')
        history = []

    # ── Data ──────────────────────────────────────────────────────
    print(f'\nLoading data from {data_root}...')
    t0 = time.time()
    train_ds = FastWindowDataset(data_root, split='train', label_remap=label_remap)
    val_ds = FastWindowDataset(data_root, split='val', label_remap=label_remap)
    print(f'  Data loaded in {time.time()-t0:.0f}s')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    # ── Baseline eval (only if starting fresh) ────────────────────
    if start_epoch == 0:
        print('\n=== Baseline (before training) ===')
        baseline = evaluate_mh(model, val_loader, device, max_batches=args.val_batches)
        print(f'  Oracle minADE_{args.K}: {baseline["min_ade"].mean():.4f}m  '
              f'minFDE_{args.K}: {baseline["min_fde"].mean():.4f}m')
        print(f'  Single-best ADE: {baseline["single_ade"].mean():.4f}m  '
              f'FDE: {baseline["single_fde"].mean():.4f}m')
        best_min_fde = baseline['min_fde'].mean()
        history.append({
            'epoch': 0, 'stage': 'baseline',
            'min_ade': float(baseline['min_ade'].mean()),
            'min_fde': float(baseline['min_fde'].mean()),
            'single_ade': float(baseline['single_ade'].mean()),
            'single_fde': float(baseline['single_fde'].mean()),
        })

    # ── Training ──────────────────────────────────────────────────
    print(f'\n=== Training epoch {start_epoch+1}~{args.epochs} ===')
    print(f'  Best minFDE so far: {best_min_fde:.4f}m')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        scheduler.step()

        # Validate every epoch
        val_metrics = evaluate_mh(model, val_loader, device, max_batches=args.val_batches)
        min_fde_mean = val_metrics['min_fde'].mean()
        improved = min_fde_mean < best_min_fde
        if improved:
            best_min_fde = min_fde_mean

        elapsed = time.time() - t0

        # ── Epoch summary ─────────────────────────────────────────
        print(f'\n  Epoch {epoch+1:2d}/{args.epochs} | '
              f'Loss: {train_metrics["loss"]:.4f} '
              f'(disp={train_metrics["disp_loss"]:.4f} conf={train_metrics["conf_loss"]:.4f}) | '
              f'minFDE_{args.K}: {min_fde_mean:.4f}m {"*" if improved else ""} | '
              f'{elapsed:.0f}s')

        # Per-intent breakdown
        pi = val_metrics['per_intent']
        for intent_id in sorted(pi.keys()):
            d = pi[intent_id]
            name = intent_names.get(intent_id, f'C{intent_id}')
            print(f'    {name:12s}: minFDE={np.mean(d["min_fde"]):.4f}m  '
                  f'(n={d["count"]})')

        # Record history
        history.append({
            'epoch': epoch + 1,
            'loss': float(train_metrics['loss']),
            'disp_loss': float(train_metrics['disp_loss']),
            'conf_loss': float(train_metrics['conf_loss']),
            'min_ade': float(val_metrics['min_ade'].mean()),
            'min_fde': float(min_fde_mean),
            'single_ade': float(val_metrics['single_ade'].mean()),
            'single_fde': float(val_metrics['single_fde'].mean()),
            'best_min_fde': float(best_min_fde),
            'per_intent': {
                intent_names.get(k, f'C{k}'): {
                    'min_fde': float(np.mean(v['min_fde'])),
                    'count': v['count'],
                }
                for k, v in pi.items()
            },
            'elapsed_s': int(elapsed),
            'timestamp': datetime.now().isoformat(),
        })

        # ── Save checkpoint ───────────────────────────────────────
        ckpt_path = save_checkpoint(
            model, optimizer, scheduler, epoch, best_min_fde, history, args, ckpt_dir)
        print(f'  Checkpoint: {ckpt_path.name}')

        # Save history JSON for external monitoring
        history_path = ckpt_dir / f'{args.model}_K{args.K}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    # ── Final ─────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Training Complete')
    if history:
        base = history[0]
        final = history[-1]
        imp = (base['min_fde'] - final['min_fde']) / base['min_fde'] * 100
        print(f'  minADE_{args.K}: {base["min_ade"]:.4f} -> {final["min_ade"]:.4f}m '
              f'({(base["min_ade"]-final["min_ade"])/base["min_ade"]*100:+.1f}%)')
        print(f'  minFDE_{args.K}: {base["min_fde"]:.4f} -> {final["min_fde"]:.4f}m '
              f'({imp:+.1f}%)')
        print(f'  Best minFDE_{args.K}: {best_min_fde:.4f}m')
    print(f'{"="*60}')

    # Save final weights
    out_dir = Path(__file__).resolve().parents[1] / 'weights'
    out_path = out_dir / f'{args.model}_multihead_K{args.K}.pth'
    torch.save({
        'multi_decoder_state': model.ua_pgd.neural_decoder.state_dict(),
        'K': args.K,
        'model_type': args.model,
        'min_fde': best_min_fde,
        'history': history,
    }, out_path)
    print(f'\nFinal weights: {out_path}')
    print(f'History log:  {history_path}')
    print('Done!')


if __name__ == '__main__':
    main()
