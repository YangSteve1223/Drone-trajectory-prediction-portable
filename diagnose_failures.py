#!/usr/bin/env python3
"""
Deep diagnostic: find worst-performing samples and analyze root causes.
Extracts ALL internal model states for frame-by-frame diagnosis.

Diagnostic dimensions:
  1. Intent probability distribution (is the model confused about intent?)
  2. Gate inertia / anchor (physics vs neural contribution)
  3. Physics trajectory vs neural delta (which component is wrong?)
  4. Per-dimension error (X, Y, Z breakdown)
  5. Speed estimation (near threshold issues?)
  6. Uncertainty calibration (is model confident but wrong?)
  7. Trigger decision
"""

import torch, numpy as np, sys
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 7})

INTENT_NAMES_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
INTENT_NAMES_4 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'DESCEND']
DT_LOW = 0.2   # 5Hz
DT_HIGH = 1.0  # 1Hz

# Colors
C_BLUE = '#2196F3'; C_RED = '#F44336'; C_GREEN = '#4CAF50'
C_ORANGE = '#FF9800'; C_PURPLE = '#9C27B0'; C_TEAL = '#009688'


def run_model_with_internals(model, hist, device):
    """Run model and extract ALL internal states."""
    with torch.no_grad():
        out = model(hist.unsqueeze(0).to(device), force_predict=True, return_all=True)
    # gate_anchor is not in model output; use gate_inertia as rough proxy for anchor
    gi = out.get('gate_inertia', torch.zeros(1, 20, device=device))
    return {
        'predictions': out['predictions'][0].cpu(),
        'intent_logits': out['intent_logits'][0].cpu(),
        'intent_weights': out['intent_weights'][0].cpu(),
        'trigger_decision': out['trigger_decision'][0].cpu().item(),
        'uncertainty': out['uncertainty'][0].cpu(),
        'physics_trajectory': out.get('physics_trajectory', torch.zeros(1, 20, 3, device=device))[0].cpu(),
        'gate_inertia': gi[0].cpu(),
        'gate_anchor': torch.zeros(20),  # not exposed by model
        'encoded_features': out.get('encoded_features', None),
        'global_anchor': out.get('global_anchor', None),
    }


def collect_all_samples(ds, model, model_label, device, n_worst=10):
    """Run all test samples in batches, collect errors and internal states."""
    from torch.utils.data import DataLoader

    results = []
    n = len(ds)
    print(f'  Running {n} {model_label} samples...')

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    idx_offset = 0

    for batch_idx, (hist_batch, target_batch, intent_batch) in enumerate(loader):
        hist_batch = hist_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            out = model(hist_batch, force_predict=True, return_all=True)

        preds = out['predictions'].cpu()          # (B, 20, 3)
        intent_logits = out['intent_logits'].cpu()  # (B, n_class)
        intent_weights = out['intent_weights'].cpu()
        uncertainty = out['uncertainty'].cpu()     # (B, 20, 3)
        physics = out.get('physics_trajectory', torch.zeros_like(preds)).cpu()
        gate_inertia = out.get('gate_inertia', torch.zeros(preds.shape[0], preds.shape[1])).cpu()

        # Apply Z correction for HIGH model (same logic as predictor.py)
        if model_label == 'HIGH':
            high_intent_prob = torch.softmax(intent_logits, dim=-1)
            descend_prob = high_intent_prob[:, 3]

            dampen = torch.ones(descend_prob.shape[0])
            strong_mask = descend_prob < 0.05
            weak_mask = (descend_prob >= 0.05) & (descend_prob < 0.20)

            if strong_mask.any():
                dampen[strong_mask] = 0.05
            if weak_mask.any():
                t = (descend_prob[weak_mask] - 0.05) / 0.15
                dampen[weak_mask] = 0.30 + 0.70 * t

            apply_mask = dampen < 1.0
            if apply_mask.any():
                preds[apply_mask, :, 2] *= dampen[apply_mask].view(-1, 1)

        # Compute targets (on GPU first, then move to CPU)
        last_pos = hist_batch[:, -1:, :3]
        target_rel = (target_batch[:, :, :3] - last_pos).cpu()  # (B, 20, 3)

        # Per-step errors
        step_errs = torch.norm(preds - target_rel, dim=-1)  # (B, 20)
        endpoint_errs = step_errs[:, -1]  # (B,)
        avg_errs = step_errs.mean(dim=1)  # (B,)

        # Per-dimension errors
        err_x = (preds[:, :, 0] - target_rel[:, :, 0]).abs().mean(dim=1)
        err_y = (preds[:, :, 1] - target_rel[:, :, 1]).abs().mean(dim=1)
        err_z = (preds[:, :, 2] - target_rel[:, :, 2]).abs().mean(dim=1)

        # Intent
        intent_probs = torch.softmax(intent_logits, dim=-1)
        top_intents = intent_probs.argmax(dim=-1)
        top_probs = intent_probs.max(dim=-1).values
        intent_entropies = -(intent_probs * torch.log(intent_probs + 1e-8)).sum(dim=-1)

        # Speed
        speeds = DronePredictor.compute_speed(hist_batch).cpu()

        # Gate stats
        gate_inertia_means = gate_inertia.mean(dim=1)

        # Physics norm
        phys_norms = physics.norm(dim=-1).mean(dim=1)
        pred_norms = preds.norm(dim=-1).mean(dim=1)

        # Uncertainty
        uncertainty_means = uncertainty.mean(dim=(1, 2))

        # GT curvature
        gt_vel = torch.diff(target_rel, dim=1)
        gt_dir = torch.atan2(gt_vel[:, :, 1], gt_vel[:, :, 0])
        gt_turns = torch.abs(torch.diff(gt_dir, dim=1)).mean(dim=1)
        # Handle NaN from zero velocity
        gt_turns = torch.nan_to_num(gt_turns, nan=0.0)

        B = preds.shape[0]
        for i in range(B):
            idx = idx_offset + i
            intent_label = intent_batch[i].item() if isinstance(intent_batch[i], torch.Tensor) else int(intent_batch[i])

            results.append({
                'idx': idx,
                'intent_label': intent_label,
                'top_intent': top_intents[i].item(),
                'top_prob': top_probs[i].item(),
                'intent_entropy': intent_entropies[i].item(),
                'intent_probs': intent_probs[i],
                'speed': speeds[i].item(),
                'endpoint_err': endpoint_errs[i].item(),
                'avg_err': avg_errs[i].item(),
                'step_err': step_errs[i],
                'err_x': err_x[i].item(), 'err_y': err_y[i].item(), 'err_z': err_z[i].item(),
                'gate_inertia_mean': gate_inertia_means[i].item(),
                'gate_anchor_mean': 0.0,
                'phys_norm': phys_norms[i].item(),
                'pred_norm': pred_norms[i].item(),
                'uncertainty_mean': uncertainty_means[i].item(),
                'gt_turn': gt_turns[i].item(),
                'hist': hist_batch[i].cpu(),
                'target': target_rel[i],
                'internals': {
                    'predictions': preds[i],
                    'intent_logits': intent_logits[i],
                    'intent_weights': intent_weights[i],
                    'trigger_decision': True,
                    'uncertainty': uncertainty[i],
                    'physics_trajectory': physics[i],
                    'gate_inertia': gate_inertia[i],
                    'gate_anchor': torch.zeros(20),
                    'encoded_features': None,
                    'global_anchor': None,
                },
            })

        idx_offset += B
        if (batch_idx + 1) % 20 == 0:
            print(f'    {idx_offset}/{n} done...')

    # Sort by endpoint error (worst first)
    results.sort(key=lambda r: r['endpoint_err'], reverse=True)
    return results


def diagnose_sample(ax_grid, r, model_label, dt, intent_names, rank):
    """Generate a comprehensive diagnostic panel for one sample."""
    pred = r['internals']['predictions']
    target = r['target']
    hist = r['hist']
    step_err = r['step_err']
    intent_probs = r['intent_probs']
    gate_inertia = r['internals']['gate_inertia']
    gate_anchor = r['internals']['gate_anchor']
    physics = r['internals']['physics_trajectory']
    uncertainty = r['internals']['uncertainty']
    n_steps = len(pred)

    time_axis = np.arange(1, n_steps + 1) * dt
    hist_time = np.arange(-19, 1) * dt

    last_pos = hist[-1, :3].numpy()

    # --- Row 1: 3D trajectory ---
    ax = ax_grid[0]
    hp = hist[:, :3].numpy()
    pa = pred.numpy() + last_pos
    ta = target.numpy() + last_pos
    ax.plot(hp[:, 0], hp[:, 1], color=C_BLUE, lw=1.5, label='History')
    ax.plot(pa[:, 0], pa[:, 1], color=C_RED, lw=1.5, ls='--', label='Pred')
    ax.plot(ta[:, 0], ta[:, 1], color=C_GREEN, lw=1.5, ls='--', label='Truth')
    ax.scatter(hp[-1, 0], hp[-1, 1], c=C_BLUE, s=40, marker='s', zorder=5)
    ax.scatter(pa[:, 0], pa[:, 1], c=C_RED, s=5, alpha=0.5, zorder=3)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'#{rank} idx={r["idx"]} end_err={r["endpoint_err"]:.2f}m speed={r["speed"]:.1f}m/s',
                 fontweight='bold', fontsize=7)
    ax.set_box_aspect(1); ax.grid(True, alpha=0.3); ax.legend(fontsize=5, loc='upper right')

    # --- Row 2: Per-step error ---
    ax = ax_grid[1]
    ax.bar(time_axis, step_err.numpy(), width=dt*0.7, color=C_RED, alpha=0.7, label='Step err')
    ax.axhline(y=step_err.mean().item(), color=C_RED, ls=':', lw=1, label=f'Mean={step_err.mean():.2f}')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Error (m)')
    ax.set_title('Per-Step L2 Error'); ax.grid(True, alpha=0.3); ax.legend(fontsize=5)

    # --- Row 3: Per-dimension error ---
    ax = ax_grid[2]
    err_x = (pred[:, 0] - target[:, 0]).abs().numpy()
    err_y = (pred[:, 1] - target[:, 1]).abs().numpy()
    err_z = (pred[:, 2] - target[:, 2]).abs().numpy()
    ax.plot(time_axis, err_x, 'o-', color='#E91E63', ms=2, lw=0.8, label='X')
    ax.plot(time_axis, err_y, 's-', color='#2196F3', ms=2, lw=0.8, label='Y')
    ax.plot(time_axis, err_z, '^-', color='#4CAF50', ms=2, lw=0.8, label='Z')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Abs Error (m)')
    ax.set_title('Per-Dimension Error'); ax.grid(True, alpha=0.3); ax.legend(fontsize=5)

    # --- Row 4: Intent probabilities ---
    ax = ax_grid[3]
    n_intents = len(intent_probs)
    bars = ax.bar(range(n_intents), intent_probs.numpy(), color=plt.cm.Set2(np.linspace(0, 1, n_intents)))
    for i, (bar, val) in enumerate(zip(bars, intent_probs.numpy())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=5, fontweight='bold')
    ax.set_xticks(range(n_intents))
    ax.set_xticklabels(intent_names[:n_intents], rotation=30, ha='right', fontsize=6)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Probability'); ax.set_title(
        f'Intent: true={intent_names[r["intent_label"]]}, pred={intent_names[r["top_intent"]]} '
        f'(entropy={r["intent_entropy"]:.3f})')
    ax.grid(True, alpha=0.2, axis='y')

    # --- Row 5: Gate dynamics ---
    ax = ax_grid[4]
    ax.plot(time_axis, gate_inertia.numpy(), 'o-', color=C_TEAL, ms=3, lw=1, label='Inertia (physics weight)')
    ax.plot(time_axis, gate_anchor.numpy(), 's-', color=C_PURPLE, ms=3, lw=1, label='Anchor pull')
    ax.fill_between(time_axis, 0, 1, alpha=0.05, color='gray')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Gate Value')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Gates: inertia_mean={r["gate_inertia_mean"]:.3f} anchor_mean={r["gate_anchor_mean"]:.3f}')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=5)

    # --- Row 6: Physics vs Neural vs Final ---
    ax = ax_grid[5]
    phys_norm = physics.norm(dim=-1).numpy()
    pred_norm_step = pred.norm(dim=-1).numpy()
    target_norm = target.norm(dim=-1).numpy()
    ax.plot(time_axis, phys_norm, 's-', color=C_TEAL, ms=3, lw=1, label='Physics (kinematic)')
    ax.plot(time_axis, pred_norm_step, 'o-', color=C_RED, ms=3, lw=1, label='Prediction')
    ax.plot(time_axis, target_norm, 'D-', color=C_GREEN, ms=3, lw=1, label='Truth')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Displacement Norm (m)')
    ax.set_title(f'Physics={phys_norm.mean():.1f}m vs Pred={pred_norm_step.mean():.1f}m vs Truth={target_norm.mean():.1f}m')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=5)

    # --- Row 7: Uncertainty ---
    ax = ax_grid[6]
    unc = uncertainty.numpy()
    ax.plot(time_axis, unc[:, 0], 'o-', color='#E91E63', ms=2, lw=0.8, label='var X')
    ax.plot(time_axis, unc[:, 1], 's-', color='#2196F3', ms=2, lw=0.8, label='var Y')
    ax.plot(time_axis, unc[:, 2], '^-', color='#4CAF50', ms=2, lw=0.8, label='var Z')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Log-Variance')
    ax.set_title(f'Uncertainty (mean={r["uncertainty_mean"]:.3f}) — higher = less confident')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=5)

    # --- Row 8: Z-axis detailed ---
    ax = ax_grid[7]
    hist_z = hist[:, 2].numpy()
    pred_z_abs = pred[:, 2].numpy() + last_pos[2]
    true_z_abs = target[:, 2].numpy() + last_pos[2]
    ax.plot(hist_time, hist_z, color=C_BLUE, lw=1.5, label='Hist Z')
    ax.plot(time_axis, pred_z_abs, color=C_RED, lw=1.5, ls='--', label='Pred Z')
    ax.plot(time_axis, true_z_abs, color=C_GREEN, lw=1.5, ls='--', label='True Z')
    ax.axvline(x=0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Z (m)')
    ax.set_title(f'Z-axis: pred range={np.ptp(pred_z_abs):.3f}m true range={np.ptp(true_z_abs):.3f}m')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=5)


def make_summary_figure(worst, model_label, intent_names, out_path):
    """Create summary figure: scatter plots showing error correlations."""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{model_label} — Error Pattern Analysis (n={len(worst)} worst samples)',
                 fontsize=13, fontweight='bold')

    errors = np.array([r['endpoint_err'] for r in worst])
    speeds = np.array([r['speed'] for r in worst])
    entropies = np.array([r['intent_entropy'] for r in worst])
    top_probs = np.array([r['top_prob'] for r in worst])
    inertias = np.array([r['gate_inertia_mean'] for r in worst])
    anchors = np.array([r['gate_anchor_mean'] for r in worst])
    uncertainties = np.array([r['uncertainty_mean'] for r in worst])
    gt_turns = np.array([r['gt_turn'] for r in worst])
    intent_labels = np.array([r['intent_label'] for r in worst])
    top_intents = np.array([r['top_intent'] for r in worst])

    # 1. Error vs Speed
    ax = fig.add_subplot(2, 4, 1)
    ax.scatter(speeds, errors, c=C_RED, alpha=0.6, s=20)
    ax.set_xlabel('Speed (m/s)'); ax.set_ylabel('Endpoint Error (m)')
    ax.set_title('Error vs Speed'); ax.grid(True, alpha=0.3)

    # 2. Error vs Intent Entropy
    ax = fig.add_subplot(2, 4, 2)
    ax.scatter(entropies, errors, c=C_PURPLE, alpha=0.6, s=20)
    ax.set_xlabel('Intent Entropy'); ax.set_ylabel('Endpoint Error (m)')
    ax.set_title('Error vs Intent Confusion'); ax.grid(True, alpha=0.3)

    # 3. Error vs Gate Inertia
    ax = fig.add_subplot(2, 4, 3)
    ax.scatter(inertias, errors, c=C_TEAL, alpha=0.6, s=20)
    ax.set_xlabel('Gate Inertia Mean'); ax.set_ylabel('Endpoint Error (m)')
    ax.set_title('Error vs Physics Reliance'); ax.grid(True, alpha=0.3)

    # 4. Error vs Uncertainty
    ax = fig.add_subplot(2, 4, 4)
    ax.scatter(uncertainties, errors, c=C_ORANGE, alpha=0.6, s=20)
    ax.set_xlabel('Uncertainty Mean'); ax.set_ylabel('Endpoint Error (m)')
    ax.set_title('Error vs Model Confidence'); ax.grid(True, alpha=0.3)

    # 5. Error vs GT Turn Rate
    ax = fig.add_subplot(2, 4, 5)
    ax.scatter(gt_turns, errors, c='#E91E63', alpha=0.6, s=20)
    ax.set_xlabel('GT Turn Rate (rad)'); ax.set_ylabel('Endpoint Error (m)')
    ax.set_title('Error vs Trajectory Curvature'); ax.grid(True, alpha=0.3)

    # 6. Intent Confusion Matrix (true vs predicted)
    ax = fig.add_subplot(2, 4, 6)
    n_intents = len(intent_names)
    cm = np.zeros((n_intents, n_intents))
    for r in worst:
        cm[r['intent_label'], r['top_intent']] += 1
    im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_intents)); ax.set_xticklabels(intent_names, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(n_intents)); ax.set_yticklabels(intent_names, fontsize=6)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(n_intents):
        for j in range(n_intents):
            if cm[i, j] > 0:
                ax.text(j, i, f'{int(cm[i,j])}', ha='center', va='center',
                        fontsize=6, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_title('Intent Confusion (worst samples)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 7. Per-dimension error share
    ax = fig.add_subplot(2, 4, 7)
    x_errs = np.array([r['err_x'] for r in worst])
    y_errs = np.array([r['err_y'] for r in worst])
    z_errs = np.array([r['err_z'] for r in worst])
    ax.bar(['X', 'Y', 'Z'], [x_errs.mean(), y_errs.mean(), z_errs.mean()],
           color=['#E91E63', '#2196F3', '#4CAF50'], alpha=0.7)
    ax.set_ylabel('Mean Abs Error (m)')
    ax.set_title('Error by Dimension')
    for i, v in enumerate([x_errs.mean(), y_errs.mean(), z_errs.mean()]):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=8)

    # 8. Top-10 error bar chart
    ax = fig.add_subplot(2, 4, 8)
    top10 = worst[:10]
    idx_labels = [f'#{r["idx"]}' for r in top10]
    err_vals = [r['endpoint_err'] for r in top10]
    colors = [C_RED if r['intent_label'] != r['top_intent'] else C_TEAL for r in top10]
    ax.barh(range(10), err_vals[::-1], color=colors[::-1], alpha=0.7)
    ax.set_yticks(range(10))
    ax.set_yticklabels(idx_labels[::-1], fontsize=6)
    ax.set_xlabel('Endpoint Error (m)')
    ax.set_title('Top 10 Worst (red=intent mismatch)')
    ax.grid(True, alpha=0.3, axis='x')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def make_detail_figure(worst, model_label, dt, intent_names, out_path, n_show=6):
    """Create detailed diagnostic panels for the worst N samples."""
    n_show = min(n_show, len(worst))

    fig = plt.figure(figsize=(28, 5 * n_show))
    fig.suptitle(f'{model_label} — Deep Diagnostic: {n_show} Worst Samples\n'
                 f'[Row 1: XY | Row 2: StepErr | Row 3: DimErr | Row 4: Intent | '
                 f'Row 5: Gates | Row 6: Physics | Row 7: Uncertainty | Row 8: Z-axis]',
                 fontsize=12, fontweight='bold', y=1.002)

    for i in range(n_show):
        r = worst[i]
        # 8 rows x 1 col per sample
        axes = []
        for j in range(8):
            ax = fig.add_subplot(n_show, 8, i * 8 + j + 1)
            axes.append(ax)
        diagnose_sample(axes, r, model_label, dt, intent_names, i + 1)

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def analyze_patterns(worst, model_label, intent_names):
    """Print a structured analysis of error patterns."""
    print(f'\n{"="*80}')
    print(f'PATTERN ANALYSIS: {model_label} — Worst {min(20, len(worst))} Samples')
    print(f'{"="*80}')

    # 1. Intent mismatch rate
    mismatches = [r for r in worst if r['intent_label'] != r['top_intent']]
    mm_rate = len(mismatches) / len(worst) * 100
    print(f'\n[1] Intent Mismatch Rate: {len(mismatches)}/{len(worst)} = {mm_rate:.1f}%')

    if mismatches:
        print(f'    Breakdown:')
        for r in mismatches[:10]:
            true_name = intent_names[r['intent_label']] if r['intent_label'] < len(intent_names) else '?'
            pred_name = intent_names[r['top_intent']] if r['top_intent'] < len(intent_names) else '?'
            print(f'    idx={r["idx"]:5d}: true={true_name:10s} pred={pred_name:10s} '
                  f'err={r["endpoint_err"]:.2f}m entropy={r["intent_entropy"]:.3f}')

    # 2. High entropy samples (confused intent)
    high_entropy = [r for r in worst if r['intent_entropy'] > 1.0]
    print(f'\n[2] High Intent Entropy (>1.0): {len(high_entropy)}/{len(worst)} '
          f'(mean entropy of worst: {np.mean([r["intent_entropy"] for r in worst]):.3f})')

    # 3. Gate analysis
    high_inertia = [r for r in worst if r['gate_inertia_mean'] > 0.5]
    low_inertia = [r for r in worst if r['gate_inertia_mean'] < 0.2]
    print(f'\n[3] Gate Inertia Distribution:')
    print(f'    High inertia (>0.5, physics-dominated): {len(high_inertia)} samples')
    print(f'    Low inertia  (<0.2, neural-dominated):  {len(low_inertia)} samples')
    print(f'    Mean inertia: {np.mean([r["gate_inertia_mean"] for r in worst]):.3f}')

    # For high-error + high-inertia → physics model is wrong
    hi_hi = [r for r in worst if r['gate_inertia_mean'] > 0.5 and r['endpoint_err'] > np.median([x['endpoint_err'] for x in worst])]
    if hi_hi:
        print(f'    [!!] High-error + High-inertia: {len(hi_hi)} samples (physics model may be misleading)')

    # 4. Z-axis issues
    z_bad = [r for r in worst if r['err_z'] > r['err_x'] and r['err_z'] > r['err_y']]
    print(f'\n[4] Z-dominated Errors: {len(z_bad)}/{len(worst)} '
          f'(avg Z err: {np.mean([r["err_z"] for r in worst]):.3f}m)')

    # 5. Uncertainty calibration
    print(f'\n[5] Uncertainty Calibration:')
    print(f'    Mean logvar: {np.mean([r["uncertainty_mean"] for r in worst]):.3f}')
    print(f'    Mean error:  {np.mean([r["endpoint_err"] for r in worst]):.3f}m')
    # If low uncertainty + high error = overconfident
    overconfident = [r for r in worst if r['uncertainty_mean'] < -2 and r['endpoint_err'] > 1.0]
    if overconfident:
        print(f'    [!!] Overconfident (low unc, high err): {len(overconfident)} samples')

    # 6. Speed threshold proximity
    near_thresh = [r for r in worst if 4.0 < r['speed'] < 6.0]
    print(f'\n[6] Near Speed Threshold (4-6 m/s): {len(near_thresh)}/{len(worst)}')
    if near_thresh:
        for r in near_thresh[:5]:
            print(f'    idx={r["idx"]:5d}: speed={r["speed"]:.1f}m/s err={r["endpoint_err"]:.2f}m')

    # 7. Turn rate correlation
    high_turn = [r for r in worst if r['gt_turn'] > 0.3]
    print(f'\n[7] High Curvature (>0.3 rad): {len(high_turn)}/{len(worst)} '
          f'(mean err: {np.mean([r["endpoint_err"] for r in high_turn]):.2f}m)' if high_turn else '')

    # 8. Top root cause candidates
    print(f'\n[8] ROOT CAUSE CANDIDATES:')
    causes = {'Intent Confusion': mm_rate,
              'Physics Over-reliance': float(len(hi_hi)) if hi_hi else 0.0,
              'Z-axis Drift': float(len(z_bad)),
              'Overconfident': float(len(overconfident)) if overconfident else 0.0,
              'Threshold Proximity': float(len(near_thresh))}
    max_val = max(causes.values()) if causes.values() else 1.0
    for cause, count in sorted(causes.items(), key=lambda x: x[1], reverse=True):
        bar = '#' * min(40, int(count / max_val * 40)) if max_val > 0 else ''
        print(f'    {cause:25s}: {count:3.0f} samples  {bar}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor
    print('Loading predictor...')
    p = DronePredictor(device=device)

    # ============================================================
    # LOW MODEL: UAV-Flow test set
    # ============================================================
    print('\n' + '='*80)
    print('LOW-SPEED MODEL DIAGNOSTICS (UAV-Flow, 5Hz, 6-class)')
    print('='*80)

    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    n_low = len(ds_low)
    low_results = collect_all_samples(ds_low, p.low, 'LOW', device, n_worst=min(30, n_low))

    worst_low = low_results[:min(30, len(low_results))]
    analyze_patterns(worst_low, 'LOW (UAV-Flow)', INTENT_NAMES_6)

    make_summary_figure(worst_low, 'LOW Model (UAV-Flow, 5Hz)',
                        INTENT_NAMES_6, out_dir / 'diag_low_summary.png')
    make_detail_figure(worst_low, 'LOW (UAV-Flow, 5Hz)', DT_LOW,
                       INTENT_NAMES_6, out_dir / 'diag_low_detail.png', n_show=6)

    # ============================================================
    # HIGH MODEL: SimCruise test set
    # ============================================================
    print('\n' + '='*80)
    print('HIGH-SPEED MODEL DIAGNOSTICS (SimCruise, 1Hz, 4-class)')
    print('='*80)

    ds_high = FastWindowDataset('../SimCruise', split='test', label_remap={4: 3})
    n_high = len(ds_high)
    high_results = collect_all_samples(ds_high, p.high, 'HIGH', device, n_worst=min(30, n_high))

    worst_high = high_results[:min(30, len(high_results))]
    analyze_patterns(worst_high, 'HIGH (SimCruise)', INTENT_NAMES_4)

    make_summary_figure(worst_high, 'HIGH Model (SimCruise, 1Hz)',
                        INTENT_NAMES_4, out_dir / 'diag_high_summary.png')
    make_detail_figure(worst_high, 'HIGH (SimCruise, 1Hz)', DT_HIGH,
                       INTENT_NAMES_4, out_dir / 'diag_high_detail.png', n_show=6)

    # ============================================================
    # Cross-model comparison
    # ============================================================
    print('\n' + '='*80)
    print('CROSS-MODEL COMPARISON')
    print('='*80)

    low_errs = [r['endpoint_err'] for r in low_results]
    high_errs = [r['endpoint_err'] for r in high_results]

    print(f'\nLOW  — Mean err: {np.mean(low_errs):.3f}m  '
          f'Median: {np.median(low_errs):.3f}m  '
          f'P95: {np.percentile(low_errs, 95):.3f}m  '
          f'Max: {np.max(low_errs):.3f}m')
    print(f'HIGH — Mean err: {np.mean(high_errs):.3f}m  '
          f'Median: {np.median(high_errs):.3f}m  '
          f'P95: {np.percentile(high_errs, 95):.3f}m  '
          f'Max: {np.max(high_errs):.3f}m')

    # Histogram comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Error Distribution: LOW vs HIGH', fontsize=13, fontweight='bold')

    for ax, errs, label, color in [
        (axes[0], low_errs, 'LOW (UAV-Flow)', C_BLUE),
        (axes[1], high_errs, 'HIGH (SimCruise)', C_RED),
    ]:
        ax.hist(errs, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(x=np.median(errs), color='black', ls='--', lw=1.5, label=f'Median={np.median(errs):.3f}')
        ax.axvline(x=np.percentile(errs, 95), color='red', ls=':', lw=1.5, label=f'P95={np.percentile(errs, 95):.3f}')
        ax.set_xlabel('Endpoint Error (m)'); ax.set_ylabel('Count')
        ax.set_title(f'{label} (n={len(errs)})'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / 'diag_error_distribution.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir / "diag_error_distribution.png"}')
    plt.close()

    # Save worst samples indices for reference
    print(f'\nWorst LOW sample indices:  {[r["idx"] for r in worst_low[:10]]}')
    print(f'Worst HIGH sample indices: {[r["idx"] for r in worst_high[:10]]}')

    print('\nDone! All diagnostic charts saved to pic-results/')


if __name__ == '__main__':
    main()
