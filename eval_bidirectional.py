#!/usr/bin/env python3
"""
Bidirectional Mamba evaluation for LOW model.
Trains BidirectionalEnhancer on LOW data and compares before/after metrics.
"""

import torch, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from emam_model import TrajectoryPredictor
from emam_model.bidirectional_mamba import BidirectionalMambaEncoder
from utils.fast_data_loader import FastWindowDataset


class LowBidirectionalEnhancer(torch.nn.Module):
    """Bidirectional enhancer adapted for LOW model (works directly with emam_se)."""

    def __init__(self, d_model=128, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = torch.nn.Linear(6, d_model)
        self.bi_encoder = BidirectionalMambaEncoder(
            d_model=d_model, d_state=d_state, expand=expand,
        )
        self.output_scale = torch.nn.Parameter(torch.tensor(0.01))

        # Init near-zero
        for name, param in self.bi_encoder.named_parameters():
            if 'out_proj' in name or 'residual_proj' in name:
                torch.nn.init.normal_(param, std=0.001)

    def forward(self, x):
        """x: (B, T, 6) raw input → (B, T, d_model) residual"""
        projected = self.input_proj(x)
        _, _, fused = self.bi_encoder(projected)
        return fused * self.output_scale

    def enable_training(self):
        self.output_scale.data.fill_(1.0)


def evaluate_low_model(model, dataset, device, desc='Eval', max_samples=None):
    """Evaluate ADE/FDE/direction error for LOW model."""
    model.eval()
    all_ade, all_fde, all_dir_err = [], [], []
    per_intent_fde = defaultdict(list)
    intent_names = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESCEND', 'HOVER']

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in tqdm(range(n), desc=desc):
        hist, target, intent = dataset[i]
        hist = hist.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(hist, force_predict=True)

        pred = out['predictions']
        lp = hist[0, -1, :3]
        pa = lp + pred[0]
        ga = target[0, :, :3]

        diff = pa - ga
        l2 = torch.norm(diff, dim=1)
        ade = l2.mean().item()
        fde = l2[-1].item()
        all_ade.append(ade)
        all_fde.append(fde)

        pred_dir = pa[-1] - pa[0]
        true_dir = ga[-1] - ga[0]
        cos_sim = torch.nn.functional.cosine_similarity(pred_dir.unsqueeze(0), true_dir.unsqueeze(0))
        dir_err = torch.acos(torch.clamp(cos_sim, -1, 1)).item() * 180 / np.pi
        all_dir_err.append(dir_err)

        intent_idx = intent.item()
        if intent_idx < 6:
            per_intent_fde[intent_names[intent_idx]].append(fde)

    ade_arr = np.array(all_ade)
    fde_arr = np.array(all_fde)
    dir_arr = np.array(all_dir_err)

    return {
        'ade_mean': ade_arr.mean(), 'ade_median': np.median(ade_arr),
        'fde_mean': fde_arr.mean(), 'fde_median': np.median(fde_arr),
        'fde_p95': np.percentile(fde_arr, 95),
        'dir_mean': dir_arr.mean(), 'dir_median': np.median(dir_arr),
        'catastrophic_pct': (dir_arr > 90).mean() * 100,
        'per_intent': {k: np.mean(v) for k, v in per_intent_fde.items()},
        'n': len(all_ade),
    }


def main():
    device = torch.device('cuda')
    data_path = Path('../UAV-Flow-pure')
    weights_path = Path('weights')

    print('=' * 60)
    print('  Bidirectional Mamba — LOW Model Evaluation')
    print('=' * 60)

    # ── Load LOW model ──
    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(device).eval()
    ckpt = torch.load(weights_path / 'low_speed_6class.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # ── Create enhancer ──
    enhancer = LowBidirectionalEnhancer(d_model=128, d_state=16, expand=2).to(device)
    n_params = sum(p.numel() for p in enhancer.parameters())
    print(f'  Enhancer params: {n_params:,}')

    # ── Hook into encoder ──
    original_forward = model.emam_se.forward

    def hooked_forward(x):
        base = original_forward(x)
        bi = enhancer(x)
        return base + bi

    # ── Load data ──
    print('\nLoading data...')
    train_ds = FastWindowDataset(str(data_path), split='train')
    test_ds = FastWindowDataset(str(data_path), split='test')
    print(f'  Train: {len(train_ds):,}  Test: {len(test_ds):,}')

    # ── Baseline (no enhancer) ──
    print('\n[1] Baseline evaluation (no bidirectional)...')
    model.emam_se.forward = original_forward
    baseline = evaluate_low_model(model, test_ds, device, desc='Baseline', max_samples=5000)
    print(f'  ADE: {baseline["ade_mean"]:.3f}m  FDE: {baseline["fde_mean"]:.3f}m  '
          f'Dir: {baseline["dir_mean"]:.1f}°  Catastrophic: {baseline["catastrophic_pct"]:.1f}%')

    # ── Untrained enhancer (should be near-identical) ──
    print('\n[2] Untrained bidirectional (output_scale=0.01, should match baseline)...')
    model.emam_se.forward = hooked_forward
    untrained = evaluate_low_model(model, test_ds, device, desc='Untrained', max_samples=5000)
    print(f'  ADE: {untrained["ade_mean"]:.3f}m  FDE: {untrained["fde_mean"]:.3f}m  '
          f'Dir: {untrained["dir_mean"]:.1f}°  Catastrophic: {untrained["catastrophic_pct"]:.1f}%')

    # ── Train enhancer ──
    print('\n[3] Training bidirectional enhancer (3 epochs, 50K samples)...')
    enhancer.enable_training()
    enhancer.train()

    # Freeze base model
    for param in model.parameters():
        param.requires_grad_(False)

    model.emam_se.forward = hooked_forward

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(enhancer.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    for epoch in range(3):
        total_loss = 0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/3')

        for hist, pred, intent in pbar:
            if n_batches >= 800:  # ~50K samples
                break

            hist = hist.to(device)
            pred = pred.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = model(hist, force_predict=True)
                # Target: relative displacement
                last_pos = hist[:, -1:, :3]
                target_rel = pred[:, :, :3] - last_pos
                loss = torch.nn.functional.mse_loss(out['predictions'], target_rel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(enhancer.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 100 == 0:
                pbar.set_postfix(loss=f'{total_loss/n_batches:.4f}')

        avg_loss = total_loss / max(n_batches, 1)
        print(f'  Epoch {epoch+1}: avg_loss={avg_loss:.4f}  batches={n_batches}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(enhancer.state_dict(), 'weights/bidir_low_enhancer.pth')
            print(f'  -> Best saved')

    # ── Evaluate trained enhancer ──
    print('\n[4] Trained bidirectional evaluation...')
    enhancer.eval()
    model.emam_se.forward = hooked_forward
    trained = evaluate_low_model(model, test_ds, device, desc='Trained', max_samples=5000)

    # ── Comparison ──
    print('\n' + '=' * 60)
    print('  RESULTS: Baseline vs Bidirectional Mamba')
    print('=' * 60)
    print(f'  {"Metric":25s}  {"Baseline":>10s}  {"Bidir":>10s}  {"Change":>10s}')
    print(f'  {"-"*55}')

    metrics_show = [
        ('ADE mean', 'ade_mean', 'm', False),
        ('ADE median', 'ade_median', 'm', False),
        ('FDE mean', 'fde_mean', 'm', False),
        ('FDE median', 'fde_median', 'm', False),
        ('FDE P95', 'fde_p95', 'm', False),
        ('Direction error mean', 'dir_mean', '°', True),
        ('Direction error median', 'dir_median', '°', True),
        ('Catastrophic (>90°)', 'catastrophic_pct', '%', True),
    ]

    for label, key, unit, lower_better in metrics_show:
        base_val = baseline[key]
        new_val = trained[key]
        if lower_better:
            change = (base_val - new_val) / (base_val + 1e-8) * 100
            arrow = '+' if change > 0 else ''
        else:
            change = (new_val - base_val) / (base_val + 1e-8) * 100
            arrow = '+' if change > 0 else ''
        print(f'  {label:25s}  {base_val:10.3f}{unit}  {new_val:10.3f}{unit}  {arrow}{change:+.1f}%')

    print(f'\n  Per-intent FDE comparison:')
    for name in ['STRAIGHT', 'TURN_L', 'TURN_R', 'DESCEND', 'HOVER']:
        b = baseline['per_intent'].get(name, 0)
        t = trained['per_intent'].get(name, 0)
        if b > 0:
            change = (b - t) / b * 100
            print(f'    {name:12s}: {b:.3f}m -> {t:.3f}m ({change:+.1f}%)')

    # Cleanup hook
    model.emam_se.forward = original_forward
    print('\nDone!')


if __name__ == '__main__':
    main()
