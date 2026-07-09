"""Training script: multi-dataset, resumable, checkpoint export."""

# Must be set before importing torch to avoid VRAM fragmentation OOM.
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,garbage_collection_threshold:0.7')

import sys
from pathlib import Path
# Allow imports from parent directory (emam_model, utils)
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from emam_model import TrajectoryPredictor
from utils.fast_data_loader import get_dataloader
from utils.logger import TrainingLogger
from utils.metrics import (
    full_evaluation, compute_displacement_error,
    compute_distance_accuracy, compute_direction_accuracy,
    maneuver_classification
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train EMam-SE trajectory prediction')
    # Data
    parser.add_argument('--dataset', type=str, default='uav_delivery',
                        choices=['uav_delivery', 'uav_flow_sim'])
    parser.add_argument('--data_root', type=str,
                        default='./SimCruise')
    parser.add_argument('--hist_len', type=int, default=20)
    parser.add_argument('--pred_len', type=int, default=20)
    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--emam_n_layers', type=int, default=2)
    parser.add_argument('--use_trigger', type=int, default=1)
    parser.add_argument('--trigger_mode', type=str, default='funnel',
                        choices=['simple', 'funnel', 'learned'],
                        help='Trigger mode: simple=always, funnel=PPT multi-stage, learned=3-factor')
    parser.add_argument('--num_intent_classes', type=int, default=5)
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    # Loss weights
    parser.add_argument('--loss_disp_weight', type=float, default=1.0)
    parser.add_argument('--loss_intent_weight', type=float, default=0.1)
    parser.add_argument('--loss_unc_weight', type=float, default=0.05)
    # Output
    parser.add_argument('--exp_name', type=str, default='emam_se_default')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='Limit batches per epoch (0 = no limit)')
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    # Pretrained weights (partial load, allows different num classes)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint for partial weight init')
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (simulate larger batch)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile')
    return parser.parse_args()


def build_model(args):
    model = TrajectoryPredictor(
        input_dim=6,
        history_len=args.hist_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        emam_n_layers=args.emam_n_layers,
        num_intent_classes=args.num_intent_classes,
        use_trigger=bool(args.use_trigger),
        trigger_mode=getattr(args, 'trigger_mode', 'funnel'),
        loss_weights={
            'displacement': args.loss_disp_weight,
            'intent': args.loss_intent_weight,
            'uncertainty': args.loss_unc_weight
        }
    )
    return model


def load_pretrained_weights(model, checkpoint_path, device):
    """Partially load pretrained weights: load matching-shape layers, skip mismatches.

    Used for transfer from a checkpoint with different num_intent_classes.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict']
    model_state = model.state_dict()

    loaded = []
    skipped = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded.append(k)
        else:
            reason = 'missing' if k not in model_state else f'shape {tuple(v.shape)} → {tuple(model_state[k].shape)}'
            skipped.append((k, reason))

    model.load_state_dict(model_state)

    print(f'[Pretrained] Loaded {len(loaded)}/{len(state)} layers from {checkpoint_path}')
    if skipped:
        print(f'[Pretrained] Skipped {len(skipped)} layers (will train from scratch):')
        for k, reason in skipped[:8]:
            print(f'  - {k}: {reason}')
        if len(skipped) > 8:
            print(f'  ... and {len(skipped)-8} more')
    return model


def build_optimizer(model, args):
    # fused AdamW: ~15-20% faster than standard AdamW on CUDA
    use_fused = torch.cuda.is_available()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_fused,
    )
    if use_fused:
        print("Fused AdamW enabled")
    return optimizer


def build_scheduler(optimizer, args, total_steps):
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=len(optimizer.param_groups[0]['params']) * 10, gamma=0.5
        )
    else:
        scheduler = None
    return scheduler


def train_epoch(model, train_loader, optimizer, device, epoch, args, writer=None, scaler=None):
    """Return (loss_breakdown, avg_train_loss, status_dict).
    status_dict has 'healthy' (bool) and 'nan_ratio' (float) to decide whether to save.
    """
    model.train()
    total_loss = 0
    total_samples = 0
    total_batches = 0  # valid batch count
    loss_breakdown = {'displacement': 0.0, 'intent': 0.0, 'uncertainty': 0.0, 'physics': 0.0}

    use_amp = scaler is not None
    grad_accum = getattr(args, 'grad_accum', 1)
    optimizer.zero_grad()  # grad accumulation: zero at cycle start

    # Free VRAM fragments before each epoch (prevents epoch 1+ crash)
    if device.type == 'cuda' and epoch > 0:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Free CUDA IPC shared memory left by DataLoader workers
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

    # NaN counters (detect gradient explosion trend)
    nan_skip_count = 0
    consecutive_nan = 0
    _diverged = False  # whether model has diverged
    _nan_batches_after_divergence = 0

    # Epoch 0 gradient diagnostic: capture decoder grad norm after first update
    _diag_grad_sum = 0.0
    _diag_grad_count = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (hist, pred, intent_labels) in enumerate(pbar):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        hist = hist.to(device)        # (B, hist_len, 6)
        pred = pred.to(device)        # (B, pred_len, 3)
        intent_labels = intent_labels.to(device)  # (B,)

        # Forward: keep FP32 to avoid overflow (SSM scan / attention sensitive)
        out = model(hist, intent_labels=intent_labels, return_all=False)

        predictions = out['predictions']       # (B, pred_len, 3)
        uncertainty = out['uncertainty']     # (B, pred_len, 3)
        intent_logits = out['intent_logits']  # (B, num_classes)
        intent_weights = out.get('intent_weights', None)
        physics_trajectory = out.get('physics_trajectory', None)
        gate_inertia = out.get('gate_inertia', None)

        # Detect NaN/Inf activations early (avoid corrupting the model)
        if not torch.isfinite(predictions).all():
            nan_skip_count += 1
            consecutive_nan += 1
            print(f"  WARN: NaN/Inf in predictions at batch {batch_idx} "
                  f"(consecutive={consecutive_nan}), skip")
            torch.cuda.empty_cache()
            # Too many consecutive NaN -> model has crashed
            if consecutive_nan >= 10:
                if not _diverged:
                    print(f"  FATAL: {consecutive_nan} consecutive NaN batches — model diverged!")
                    _diverged = True
                _nan_batches_after_divergence += 1
                # After divergence, run 20 more batches to see if it recovers, else abort
                if _nan_batches_after_divergence > 20:
                    print(f"  FATAL: model unrecoverable. Aborting epoch.")
                    break
            continue
        else:
            consecutive_nan = 0
            if _diverged:
                _nan_batches_after_divergence = max(0, _nan_batches_after_divergence - 1)
        total_batches += 1

        # Displacement target (last history pos -> future pos delta)
        targets = pred[..., :3] - hist[:, -1:, :3]

        # Loss under AMP for speed + memory (loss magnitude is safe)
        with torch.amp.autocast('cuda', enabled=use_amp):
            losses = model.compute_loss(
                predictions, uncertainty, targets,
                intent_logits, intent_labels,
                intent_weights=intent_weights,
                physics_trajectory=physics_trajectory,
                gate_inertia=gate_inertia,
            )
        loss = losses['total_loss'] / grad_accum

        # Skip NaN/Inf batch
        if not torch.isfinite(loss):
            nan_skip_count += 1
            consecutive_nan += 1
            print(f"  WARN: non-finite loss at batch {batch_idx} "
                  f"(consecutive={consecutive_nan}), skip")
            torch.cuda.empty_cache()
            if consecutive_nan >= 10:
                print(f"  FATAL: {consecutive_nan} consecutive NaN losses — aborting epoch.")
                _diverged = True
                break
            continue

        # Backward (grad accumulation: accumulate, update every grad_accum steps)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Epoch 0: capture decoder grad norm (before optimizer.step, grads not zeroed)
        if epoch == 0 and _diag_grad_count < 5:
            _pgd_params = list(model.ua_pgd.neural_decoder.parameters())
            _pgd_grad = sum(p.grad.norm().item() for p in _pgd_params if p.grad is not None)
            _diag_grad_sum += _pgd_grad
            _diag_grad_count += 1

        is_update_step = (batch_idx + 1) % grad_accum == 0
        if is_update_step:
            if use_amp:
                scaler.unscale_(optimizer)
                # Check gradients before clipping
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Grad explosion: large pre-clip norm -> skip update to avoid corruption
                if total_norm > 100 and nan_skip_count > 0:
                    print(f"  WARN: large grad norm ({total_norm:.1f}) + prior NaN — "
                          f"skipping optimizer step to prevent weight corruption")
                    optimizer.zero_grad()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if total_norm > 100 and nan_skip_count > 0:
                    print(f"  WARN: large grad norm ({total_norm:.1f}) + prior NaN — skipping step")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
            optimizer.zero_grad()

        # Stats (use original loss)
        B = hist.shape[0]
        total_loss += loss.item() * grad_accum * B
        total_samples += B
        for k, v in loss_breakdown.items():
            loss_breakdown[k] += losses.get(f'loss_{k}', 0.0) * B

        pbar.set_postfix({'loss': f'{loss.item() * grad_accum:.4f}'})

        # Free GPU cache fragments every 25 batches (6GB VRAM protection)
        if batch_idx > 0 and batch_idx % 25 == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save decoder grad diagnostic for external use
    args._diag_decoder_grad_norm = _diag_grad_sum / max(_diag_grad_count, 1)

    nan_ratio = nan_skip_count / max(total_batches + nan_skip_count, 1)
    status = {
        'healthy': not _diverged and nan_ratio < 0.5,  # >50% NaN batches = unhealthy
        'nan_ratio': nan_ratio,
        'diverged': _diverged,
        'total_batches': total_batches,
        'nan_skipped': nan_skip_count,
    }
    if total_samples == 0:
        print("  ERROR: All batches in this epoch produced NaN — cannot continue.")
        status['healthy'] = False
        return loss_breakdown, float('inf'), status

    avg_loss = total_loss / total_samples
    return {k: v / total_samples for k, v in loss_breakdown.items()}, avg_loss, status


@torch.no_grad()
def validate(model, val_loader, device, epoch, args):
    model.eval()
    all_predictions = []
    all_targets = []
    all_intents_pred = []
    all_intents_true = []

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (hist, pred, intent_labels) in enumerate(tqdm(val_loader, desc='Validating')):
        if args.max_batches > 0 and batch_idx >= max(1, args.max_batches // 4):
            break
        hist = hist.to(device)
        pred = pred.to(device)
        intent_labels = intent_labels.to(device)

        out = model(hist, force_predict=True)  # FP32 forward for numerical stability
        predictions = out['predictions']
        intent_logits = out['intent_logits']
        uncertainty = out.get('uncertainty', None)

        B = hist.shape[0]
        # Target: displacement relative to history[-1]
        targets = pred[..., :3].to(device) - hist[:, -1:, :3].to(device)

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())
        all_intents_pred.append(intent_logits.argmax(dim=-1).cpu())
        all_intents_true.append(intent_labels.cpu())

        # Compute validation loss
        if uncertainty is not None:
            losses = model.compute_loss(
                predictions, uncertainty, targets,
                intent_logits, intent_labels,
            )
            total_loss += losses['total_loss'].item() * B
            total_samples += B

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_intents_pred = torch.cat(all_intents_pred, dim=0)
    all_intents_true = torch.cat(all_intents_true, dim=0)

    # Evaluate
    results = full_evaluation(all_predictions, all_targets)
    intent_acc = (all_intents_pred == all_intents_true).float().mean().item()
    results['intent_accuracy'] = intent_acc
    if total_samples > 0:
        results['val_loss'] = total_loss / total_samples

    return results


def save_checkpoint(model, optimizer, scheduler, epoch, results, args, is_best=False,
                    train_loss=None, force=False):
    """Save model checkpoint.

    Saves latest.pth and epoch_{N}.pth every epoch; best.pth if is_best.
    """
    ckpt_dir = Path(args.checkpoint_dir) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'args': vars(args)
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if train_loss is not None:
        ckpt['train_loss'] = train_loss

    # Save latest.pth every epoch (crash recovery)
    latest_path = ckpt_dir / 'latest.pth'
    torch.save(ckpt, latest_path)

    # Save epoch_{N}.pth every epoch (keep training trajectory)
    epoch_path = ckpt_dir / f'epoch_{epoch}.pth'
    torch.save(ckpt, epoch_path)

    # Best model
    if is_best:
        best_path = ckpt_dir / 'best.pth'
        torch.save(ckpt, best_path)

    # Save/update config
    config_path = ckpt_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)

    return str(latest_path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # === Performance tuning (6GB VRAM safe config) ===
    if device.type == 'cuda':
        # cudnn.benchmark=True picks fastest algos but may use more VRAM;
        # on a 6GB card this causes second-epoch OOM. Off = ~5% slower, no crash.
        torch.backends.cudnn.benchmark = False
        # Limit cuDNN workspace allocation
        torch.backends.cudnn.deterministic = False
        # TF32 is safe to enable (no extra VRAM)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Cap cuDNN workspace size (default unlimited)
        if hasattr(torch.backends.cudnn, 'set_conv_workspace_limit'):
            torch.backends.cudnn.set_conv_workspace_limit(32 * 1024 * 1024)  # 32MB max
        print("CUDA config: cudnn.benchmark=OFF (6GB safe), TF32=enabled, workspace_limit=32MB")

    # Data
    print(f"Loading dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")

    # Label remap: DESCEND (4->3) for num_intent_classes=4
    label_remap = None
    if args.num_intent_classes == 4:
        label_remap = {4: 3}  # DESCEND(4) -> 3; ASCEND(3) absent in data
        print("Label remap: {4:3} (DESCEND→3), num_intent_classes=4")

    train_loader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='train',
        label_remap=label_remap,
    )
    val_loader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='val',
        label_remap=label_remap,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    batches_per_epoch = len(train_loader)
    if args.max_batches > 0:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)
    print(f"Batches/epoch: {batches_per_epoch} (bs={args.batch_size})")
    # Rough estimate: d_model=128 ~125ms/batch, d_model=256 ~400ms/batch
    ms_per_batch = 125 if args.d_model <= 128 else 400
    est_min = batches_per_epoch * ms_per_batch / 1000 / 60
    print(f"Estimated ~{ms_per_batch}ms/batch → ~{est_min:.1f} min/epoch")

    # Model
    model = build_model(args).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Partial load pretrained weights (allows different class count, e.g. 4->6)
    if args.pretrained and os.path.exists(args.pretrained):
        model = load_pretrained_weights(model, args.pretrained, device)

    # torch.compile (PyTorch 2.0+, Linux only — Windows lacks Triton)
    import platform as _platform
    _can_compile = (
        hasattr(torch, 'compile') and device.type == 'cuda'
        and _platform.system() != 'Windows'  # Windows lacks Triton
        and not args.no_compile
    )
    if _can_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"torch.compile failed: {e}")
    else:
        if _platform.system() == 'Windows' and not args.no_compile:
            print("torch.compile skipped (Windows — Triton not available)")
        elif args.no_compile:
            print("torch.compile disabled (--no_compile)")

    # Optimizer
    optimizer = build_optimizer(model, args)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, args, total_steps)

    # Resume
    start_epoch = 0
    best_metric = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'results' in ckpt and ckpt['results'] is not None and 'RMSE' in ckpt['results']:
            best_metric = ckpt['results']['RMSE']

    # TensorBoard (lazy import to avoid Windows spawn import-chain crash)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'./runs/{args.exp_name}')

    # Training log & loss curves
    logger = TrainingLogger(
        log_dir=f'./runs/{args.exp_name}',
        patience=15,
        min_delta=0.001,
    )

    # AMP mixed precision (~40% VRAM saving, ~1.5x speedup)
    use_amp = (device.type == 'cuda') and (not args.no_amp)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("AMP enabled (FP16 mixed precision)")
    else:
        print("AMP disabled")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training started: {args.exp_name}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"Trigger: {getattr(args, 'trigger_mode', 'funnel')}")
    print(f"={'='*60}")

    # Startup diagnostic: check trigger filter rate (avoid FunnelTrigger silently blocking decoder)
    if getattr(args, 'trigger_mode', 'funnel') == 'funnel':
        print("\n[Diagnostic] Checking FunnelTrigger activation rate...")
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            hist, pred, intent_labels = sample_batch
            hist = hist[:32].to(device)
            _out = model(hist, intent_labels=intent_labels[:32].to(device) if intent_labels is not None else None)
            trigger_rate = _out['trigger_decision'].float().mean().item()
            print(f"  Trigger activation rate: {trigger_rate:.1%}")
            if trigger_rate < 0.3:
                print(f"  [WARN] Only {trigger_rate:.1%} samples trigger -> decoder receives "
                      f"very little gradient! Training may be ineffective.")
                print(f"  [WARN] Consider using --trigger_mode simple for stable training.")
        model.train()
        print(f"{'='*60}\n")

    last_good_ckpt = None  # checkpoint path of last healthy epoch

    for epoch in range(start_epoch, args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        current_lr = optimizer.param_groups[0]['lr']

        loss_breakdown, train_loss, train_status = train_epoch(
            model, train_loader, optimizer, device, epoch, args, writer, scaler
        )

        # Force-free fragments after each epoch (key protection on 6GB card)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

        # Unhealthy epoch: skip saving, try auto-rollback
        if not train_status['healthy']:
            print(f"  [WARN] Epoch {epoch} unhealthy (nan_ratio={train_status['nan_ratio']:.1%}, "
                  f"diverged={train_status['diverged']}). Skipping checkpoint save.")
            if train_status['diverged'] and last_good_ckpt is not None:
                print(f"  Rolling back to last good checkpoint: {last_good_ckpt}")
                ckpt = torch.load(last_good_ckpt, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                # Reduce LR to 1/3 to avoid another explosion
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] / 3
                print(f"  ↻ LR reduced to {optimizer.param_groups[0]['lr']:.2e}")
                continue  # skip this epoch, restart with rolled-back weights

            # Log but don't save
            logger.log_epoch(epoch, train_loss, loss_breakdown, None, lr=current_lr)
            continue

        # Mark as healthy epoch
        last_good_ckpt = None  # updated after save_checkpoint
        val_results = None
        is_best = False

        # Validate (every epoch)
        if True:
            val_results = validate(model, val_loader, device, epoch, args)

            # Check if best
            is_best = val_results['RMSE'] < best_metric
            if is_best:
                best_metric = val_results['RMSE']

            print(f"\nEpoch {epoch} Val Results:")
            print(f"  RMSE: {val_results['RMSE']:.4f}  |  Distance Acc: {val_results['Distance_Accuracy']:.4f}")
            print(f"  Direction Acc: {val_results['Direction_Accuracy']:.4f}  |  Intent Acc: {val_results.get('intent_accuracy', 0):.4f}")
            print(f"  Mean Jerk: {val_results['Mean_Jerk']:.4f}")

            # TensorBoard
            for k, v in val_results.items():
                writer.add_scalar(f'val/{k}', v, epoch)

        # Save checkpoint every epoch (recover from latest.pth after crash)
        ckpt_path = save_checkpoint(
            model, optimizer, scheduler, epoch, val_results, args,
            is_best=is_best, train_loss=train_loss
        )
        last_good_ckpt = ckpt_path  # record latest healthy checkpoint for rollback
        print(f"Checkpoint: {ckpt_path}")

        # JSON log
        logger.log_epoch(epoch, train_loss, loss_breakdown, val_results, lr=current_lr)

        # Plot curves + convergence check every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            logger.plot_curves()
            conv = logger.check_convergence()
            print(f"\n[Convergence] {conv['message']}")
            print(f"[Best] epoch={conv['best_epoch']}, RMSE={conv['best_rmse']:.4f}")
            if conv.get('suggestion'):
                print(f"[Suggestion] {conv['suggestion']}")
            if conv['converged']:
                print(f"\n*** Early stopping suggested! Best model at epoch {conv['best_epoch']} ***")

        # TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        for k, v in loss_breakdown.items():
            writer.add_scalar(f'train/loss_{k}', v, epoch)

        # Epoch 0 diagnostic: check decoder received gradients
        # Note: grads already zeroed by optimizer.zero_grad(), tracked within epoch
        _grad_norm_ok = getattr(train_loader.dataset, '_grad_diagnostic_done', True)
        if epoch == 0 and train_status['healthy']:
            # grad norms passed back from train_epoch via args
            _pgd_grad = getattr(args, '_diag_decoder_grad_norm', -1.0)
            if _pgd_grad < 1e-8:
                print(f"  [WARN] GradCheck: decoder grad norm = {_pgd_grad:.2e} (ZERO!)")
                print(f"  [WARN] The trigger is likely blocking all samples. Use --trigger_mode simple.")
            elif _pgd_grad > 0:
                print(f"  [OK] GradCheck: decoder grad norm = {_pgd_grad:.4f} (healthy)")

        print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | "
              f"Disp: {loss_breakdown['displacement']:.1f} | "
              f"Intent: {loss_breakdown['intent']:.4f} | "
              f"Unc: {loss_breakdown['uncertainty']:.4f} | "
              f"Phys: {loss_breakdown['physics']:.4f}")

    # Final summary
    writer.close()
    logger.plot_curves()
    summary = logger.get_summary()
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total epochs: {summary['total_epochs']}")
    print(f"  Best epoch: {summary['best_epoch']} (RMSE={summary['best_val_rmse']:.4f})")
    print(f"  Final train loss: {summary['current_train_loss']:.4f}")
    conv = summary['convergence']
    print(f"  Convergence: {conv['message']}")
    print(f"  Logs: ./runs/{args.exp_name}/training_log.json")
    print(f"  Curves: ./runs/{args.exp_name}/training_curves.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
