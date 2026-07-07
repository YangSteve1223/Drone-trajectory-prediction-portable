#!/usr/bin/env python3
"""
Multi-Hypothesis Decoder Training with Winner-Takes-All (WTA) Loss.

Approach:
  1. Load pre-trained model, freeze encoder + intent classifier
  2. Replace single NeuralDecoder with MultiHeadNeuralDecoder (K heads)
  3. Initialize K heads from original weights + small Gaussian noise
  4. Train only the multi-head decoder with WTA loss
  5. Evaluate minADE_K / minFDE_K (oracle metrics)

Usage:
  python train_multi_head.py --model high --K 5 --epochs 10 --batch_size 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, argparse, json, time, warnings
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder
from utils.fast_data_loader import FastWindowDataset, get_dataloader


def parse_args():
    p = argparse.ArgumentParser(description='Train Multi-Hypothesis Decoder')
    p.add_argument('--model', default='high', choices=['low', 'high'],
                   help='Which base model to use')
    p.add_argument('--K', type=int, default=5, help='Number of hypotheses')
    p.add_argument('--epochs', type=int, default=8, help='Training epochs')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--data_root', default='../SimCruise', help='Training data root')
    p.add_argument('--device', default='cuda', help='Device')
    p.add_argument('--noise_std', type=float, default=0.02,
                   help='Noise std for head initialization diversity')
    return p.parse_args()


def init_multi_heads_from_single(single_decoder, multi_decoder, noise_std=0.02):
    """Initialize K heads from the single pre-trained head with noise for diversity."""
    orig_delta_weight = single_decoder.delta_head.weight.data.clone()
    orig_delta_bias = single_decoder.delta_head.bias.data.clone()

    for k in range(multi_decoder.K):
        # Delta head: copy + noise
        noise_w = torch.randn_like(orig_delta_weight) * noise_std * orig_delta_weight.std()
        noise_b = torch.randn_like(orig_delta_bias) * noise_std * orig_delta_bias.std()
        multi_decoder.delta_heads[k].weight.data.copy_(orig_delta_weight + noise_w)
        multi_decoder.delta_heads[k].bias.data.copy_(orig_delta_bias + noise_b)

        # Var head: copy from single
        for p_single, p_multi in zip(single_decoder.var_head.parameters(),
                                      multi_decoder.var_heads[k].parameters()):
            if p_single.shape == p_multi.shape:
                noise = torch.randn_like(p_single) * noise_std * p_single.std()
                p_multi.data.copy_(p_single.data + noise)

    # Copy projection layer
    for p_single, p_multi in zip(single_decoder.proj.parameters(),
                                  multi_decoder.proj.parameters()):
        if p_single.shape == p_multi.shape:
            p_multi.data.copy_(p_single.data)

    print(f'  Initialized {multi_decoder.K} heads from single decoder (noise_std={noise_std})')


def train_epoch(model, loader, optimizer, device):
    """Train one epoch with WTA loss using full multi-head pipeline."""
    model.train()
    total_loss = 0.0; total_disp = 0.0; total_conf = 0.0
    n_batches = 0

    for hist_batch, target_batch, intent_batch in loader:
        hist_batch = hist_batch.to(device)
        target_batch = target_batch.to(device)
        B = hist_batch.shape[0]
        optimizer.zero_grad()

        # Forward through frozen encoder + intent classifier
        with torch.no_grad():
            h_norm = model._normalize(hist_batch)
            enc = model.emam_se(h_norm)
            dtp_out = model.ia_dtp(enc, historical_trajectory=h_norm)

        # Multi-head forward through UA-PGD
        mh_out = model.ua_pgd.forward_multi_head(
            encoded_feat=enc,
            global_anchor=dtp_out['global_anchor'],
            historical_trajectory=h_norm,
            intent_weights=dtp_out['intent_weights'],
        )

        # Ground truth relative displacement (in meters)
        last_pos = hist_batch[:, -1:, :3]
        target_rel = target_batch[:, :, :3] - last_pos

        # forward_multi_head returns predictions in meters (already denormalized)
        # For WTA loss, normalize both predictions and targets for stable training
        scale_pos = model._get_scale_pos()
        preds_norm = mh_out['all_predictions'] / scale_pos  # (K, B, P, 3) normalized
        target_norm = target_rel / scale_pos

        # WTA loss on the K hypotheses (in normalized space for numerical stability)
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

    return {
        'loss': total_loss / n_batches,
        'disp_loss': total_disp / n_batches,
        'conf_loss': total_conf / n_batches,
    }


@torch.no_grad()
def evaluate_mh(model, loader, device):
    """Evaluate minADE_K and minFDE_K using full multi-head pipeline."""
    model.eval()
    all_min_ade = []; all_min_fde = []
    all_single_ade = []; all_single_fde = []
    per_intent = defaultdict(lambda: {'min_ade': [], 'min_fde': [], 'count': 0})

    for hist_batch, target_batch, intent_batch in loader:
        hist_batch = hist_batch.to(device)
        target_batch = target_batch.to(device)

        # Full multi-head forward
        h_norm = model._normalize(hist_batch)
        enc = model.emam_se(h_norm)
        dtp_out = model.ia_dtp(enc, historical_trajectory=h_norm)

        mh_out = model.ua_pgd.forward_multi_head(
            encoded_feat=enc,
            global_anchor=dtp_out['global_anchor'],
            historical_trajectory=h_norm,
            intent_weights=dtp_out['intent_weights'],
        )

        # All predictions are already in meters (scaled by ua_pgd forward)
        all_preds = mh_out['all_predictions']  # (K, B, P, 3)
        best_pred = mh_out['predictions']      # (B, P, 3)

        last_pos = hist_batch[:, -1:, :3]
        target_rel = target_batch[:, :, :3] - last_pos

        # minADE/minFDE across K hypotheses
        metrics = MultiHeadNeuralDecoder.compute_minade_fde(all_preds, target_rel)
        all_min_ade.extend(metrics['min_ade'].cpu().tolist())
        all_min_fde.extend(metrics['min_fde'].cpu().tolist())

        # Single-best (highest confidence)
        step_errs_single = torch.norm(best_pred - target_rel, dim=-1)
        all_single_ade.extend(step_errs_single.mean(dim=1).cpu().tolist())
        all_single_fde.extend(step_errs_single[:, -1].cpu().tolist())

        for i in range(hist_batch.shape[0]):
            intent = intent_batch[i].item()
            per_intent[intent]['min_ade'].append(metrics['min_ade'][i].item())
            per_intent[intent]['min_fde'].append(metrics['min_fde'][i].item())
            per_intent[intent]['count'] += 1

    return {
        'min_ade': np.array(all_min_ade),
        'min_fde': np.array(all_min_fde),
        'single_ade': np.array(all_single_ade),
        'single_fde': np.array(all_single_fde),
        'per_intent': dict(per_intent),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Training {args.K}-hypothesis decoder on {args.model.upper()} model')

    # --- Load pre-trained base model ---
    weight_dir = Path(__file__).parent / 'weights'
    if args.model == 'high':
        weight_path = weight_dir / 'high_speed_4class.pth'
        n_classes = 4
        data_root = args.data_root
        label_remap = {4: 3}
    else:
        weight_path = weight_dir / 'low_speed_6class.pth'
        n_classes = 6
        data_root = '../UAV-Flow-pure'
        label_remap = None

    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=n_classes,
        use_trigger=True, trigger_mode='simple',
    ).to(device)

    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # --- Add helper methods for training ---
    # Disable internal normalization (we'll normalize manually for sub-module calls)
    model._norm_input = False
    model._get_scale_pos = lambda: 100.0

    def _normalize(hist):
        scale = hist.new_tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        return hist / scale.unsqueeze(0).unsqueeze(0)
    model._normalize = _normalize

    # --- Replace decoder with multi-head version ---
    multi_decoder = model.ua_pgd.replace_with_multi_head(K=args.K, noise_std=args.noise_std)
    multi_decoder = multi_decoder.to(device)
    print(f'  Replaced NeuralDecoder with MultiHeadNeuralDecoder (K={args.K})')

    # --- Freeze everything except multi-head decoder ---
    for name, param in model.named_parameters():
        param.requires_grad = 'neural_decoder' in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)')

    # --- Data ---
    print(f'\nLoading data from {data_root}...')
    train_ds = FastWindowDataset(data_root, split='train', label_remap=label_remap)
    val_ds = FastWindowDataset(data_root, split='val', label_remap=label_remap)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Baseline eval ---
    print('\n=== Baseline (before training) ===')
    baseline = evaluate_mh(model, val_loader, device)
    print(f'  Oracle minADE_K: {baseline["min_ade"].mean():.4f}m  '
          f'minFDE_K: {baseline["min_fde"].mean():.4f}m')
    print(f'  Single-best ADE: {baseline["single_ade"].mean():.4f}m  '
          f'FDE: {baseline["single_fde"].mean():.4f}m')

    intent_names = {0: 'STRAIGHT', 1: 'TURN_L', 2: 'TURN_R', 3: 'DESCEND'}
    if args.model == 'low':
        intent_names = {0: 'STRAIGHT', 1: 'TURN_L', 2: 'TURN_R', 3: 'ASCEND', 4: 'DESC', 5: 'HOVER'}

    # --- Training ---
    print(f'\n=== Training {args.epochs} epochs ===')
    best_min_fde = baseline['min_fde'].mean()

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Quick eval every 2 epochs
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate_mh(model, val_loader, device)
            min_fde_mean = val_metrics['min_fde'].mean()
            improved = min_fde_mean < best_min_fde
            if improved:
                best_min_fde = min_fde_mean

            print(f'  Epoch {epoch+1:2d}/{args.epochs} | '
                  f'Loss: {train_metrics["loss"]:.4f} '
                  f'(disp={train_metrics["disp_loss"]:.4f} conf={train_metrics["conf_loss"]:.4f}) | '
                  f'minFDE: {min_fde_mean:.4f}m {"*" if improved else ""} | '
                  f'{time.time()-t0:.0f}s')

            # Per-intent breakdown on last epoch
            if epoch == args.epochs - 1:
                pi = val_metrics['per_intent']
                for intent_id in sorted(pi.keys()):
                    d = pi[intent_id]
                    name = intent_names.get(intent_id, f'C{intent_id}')
                    print(f'    {name:12s}: minFDE={np.mean(d["min_fde"]):.4f}m  '
                          f'n={d["count"]}')
        else:
            print(f'  Epoch {epoch+1:2d}/{args.epochs} | '
                  f'Loss: {train_metrics["loss"]:.4f} | {time.time()-t0:.0f}s')

    # --- Final evaluation ---
    print(f'\n=== Final Evaluation ===')
    final = evaluate_mh(model, val_loader, device)
    print(f'  Oracle minADE_{args.K}: {final["min_ade"].mean():.4f}m  '
          f'(baseline: {baseline["min_ade"].mean():.4f}m)')
    print(f'  Oracle minFDE_{args.K}: {final["min_fde"].mean():.4f}m  '
          f'(baseline: {baseline["min_fde"].mean():.4f}m)')
    print(f'  Single-best ADE: {final["single_ade"].mean():.4f}m  '
          f'FDE: {final["single_fde"].mean():.4f}m')

    improvement = (baseline['min_fde'].mean() - final['min_fde'].mean()) / baseline['min_fde'].mean() * 100
    print(f'  minFDE improvement: {improvement:+.1f}%')

    # --- Save ---
    out_dir = Path(__file__).parent / 'weights'
    out_path = out_dir / f'{args.model}_multihead_K{args.K}.pth'
    torch.save({
        'multi_decoder_state': multi_decoder.state_dict(),
        'K': args.K,
        'model_type': args.model,
        'min_fde': final['min_fde'].mean(),
    }, out_path)
    print(f'\nSaved: {out_path}')

    # --- Also save metrics JSON ---
    metrics = {
        'K': args.K,
        'model': args.model,
        'baseline_min_ade': float(baseline['min_ade'].mean()),
        'baseline_min_fde': float(baseline['min_fde'].mean()),
        'final_min_ade': float(final['min_ade'].mean()),
        'final_min_fde': float(final['min_fde'].mean()),
        'final_single_ade': float(final['single_ade'].mean()),
        'final_single_fde': float(final['single_fde'].mean()),
        'improvement_pct': float(improvement),
        'per_intent': {},
    }
    for intent_id, d in final['per_intent'].items():
        name = intent_names.get(intent_id, f'C{intent_id}')
        metrics['per_intent'][name] = {
            'min_ade': float(np.mean(d['min_ade'])),
            'min_fde': float(np.mean(d['min_fde'])),
            'count': d['count'],
        }

    json_path = out_dir / f'{args.model}_multihead_K{args.K}_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved: {json_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
