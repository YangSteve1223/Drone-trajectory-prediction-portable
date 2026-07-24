#!/usr/bin/env python3
"""Evaluation suite: ADE, FDE, per-intent breakdown, direction error, speed profile, uncertainty calibration.
Runs on both LOW (UAV-Flow) and HIGH (SimCruise) test sets."""

import torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor import DronePredictor
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 8})

INTENT_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
INTENT_4 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'DESCEND']
DT_LOW = 0.2; DT_HIGH = 1.0


def angle_between(v1, v2):
    """Angle in radians between two vectors."""
    dot = (v1 * v2).sum(dim=-1)
    norm = v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8
    return torch.acos((dot / norm).clamp(-1, 1))


def evaluate_model(ds, model, model_label, intent_names, dt, device):
    """Run comprehensive evaluation on a dataset."""
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    # Accumulators
    all_ade = []; all_fde = []
    per_intent = defaultdict(lambda: {'ade': [], 'fde': [], 'count': 0})
    dir_errs = []; speed_errs = []
    unc_bins = defaultdict(lambda: {'errors': [], 'variances': []})

    idx_offset = 0

    for hist_batch, target_batch, intent_batch in loader:
        hist_batch = hist_batch.to(device)
        target_batch = target_batch.to(device)
        B = hist_batch.shape[0]

        with torch.no_grad():
            out = model(hist_batch, force_predict=True)

        preds = out['predictions']  # (B, 20, 3) — on GPU

        # Apply Z correction for HIGH model
        if model_label == 'HIGH':
            intent_prob = torch.softmax(out['intent_logits'], dim=-1)
            descend_prob = intent_prob[:, 3]
            dampen = torch.ones(B, device=device)
            strong = descend_prob < 0.05
            weak = (descend_prob >= 0.05) & (descend_prob < 0.20)
            if strong.any(): dampen[strong] = 0.05
            if weak.any():
                t = (descend_prob[weak] - 0.05) / 0.15
                dampen[weak] = 0.30 + 0.70 * t
            apply = dampen < 1.0
            if apply.any():
                preds[apply, :, 2] *= dampen[apply].view(-1, 1)

        # Ground truth relative displacement
        last_pos = hist_batch[:, -1:, :3]
        target_rel = target_batch[:, :, :3] - last_pos  # (B, 20, 3)

        # ---- ADE / FDE ----
        step_errs = torch.norm(preds - target_rel, dim=-1)  # (B, 20)
        ade = step_errs.mean(dim=1).cpu()   # (B,)
        fde = step_errs[:, -1].cpu()         # (B,)

        all_ade.extend(ade.tolist()); all_fde.extend(fde.tolist())

        # ---- Per-Intent ----
        for i in range(B):
            intent = intent_batch[i].item() if isinstance(intent_batch[i], torch.Tensor) else int(intent_batch[i])
            intent_name = intent_names[intent] if intent < len(intent_names) else '?'
            per_intent[intent_name]['ade'].append(ade[i].item())
            per_intent[intent_name]['fde'].append(fde[i].item())
            per_intent[intent_name]['count'] += 1

        # ---- Direction Error ----
        # Predicted heading at final step vs true heading
        pred_vec = preds[:, -1, :2].cpu()    # XY at final step
        true_vec = target_rel[:, -1, :2].cpu()
        pred_norm = pred_vec.norm(dim=-1)
        true_norm = true_vec.norm(dim=-1)
        valid = (pred_norm > 0.01) & (true_norm > 0.01)
        if valid.any():
            ang = angle_between(pred_vec[valid], true_vec[valid])
            dir_errs.extend(np.degrees(ang.numpy()).tolist())

        # ---- Speed Profile Error ----
        # Speed magnitude over time
        pred_speed = preds.norm(dim=-1).cpu() / dt       # m/s (approximate)
        true_speed = target_rel.norm(dim=-1).cpu() / dt
        speed_rmse = ((pred_speed - true_speed) ** 2).mean(dim=1).sqrt()
        speed_errs.extend(speed_rmse.tolist())

        # ---- Uncertainty Calibration ----
        logvar = out.get('uncertainty', torch.zeros_like(preds))
        if logvar is not None:
            pred_var = torch.exp(logvar.clamp(-10, 10)).mean(dim=(1, 2)).cpu()  # (B,) mean variance
            for i in range(B):
                # Bin by log10(variance)
                var_val = max(-4.0, min(4.0, np.log10(pred_var[i].item() + 1e-8)))
                bin_key = round(var_val * 2) / 2  # 0.5-width bins
                unc_bins[bin_key]['errors'].append(fde[i].item())
                unc_bins[bin_key]['variances'].append(pred_var[i].item())

        idx_offset += B

    # ---- Summarize ----
    ade_arr = np.array(all_ade); fde_arr = np.array(all_fde)
    n_total = len(ade_arr)

    print(f'\n{"="*70}')
    print(f'{model_label} MODEL — Comprehensive Evaluation (n={n_total})')
    print(f'{"="*70}')

    print(f'\n  ADE:  mean={ade_arr.mean():.4f}m  median={np.median(ade_arr):.4f}m  '
          f'P95={np.percentile(ade_arr, 95):.4f}m  max={ade_arr.max():.4f}m')
    print(f'  FDE:  mean={fde_arr.mean():.4f}m  median={np.median(fde_arr):.4f}m  '
          f'P95={np.percentile(fde_arr, 95):.4f}m  max={fde_arr.max():.4f}m')

    # Per-intent table
    print(f'\n  {"Intent":12s} {"Count":>6s} {"ADE":>8s} {"FDE":>8s} {"% of data":>10s}')
    print(f'  {"-"*44}')
    for name in intent_names:
        if name not in per_intent: continue
        d = per_intent[name]
        if d['count'] == 0: continue
        m_ade = np.mean(d['ade']); m_fde = np.mean(d['fde'])
        print(f'  {name:12s} {d["count"]:6d} {m_ade:8.4f} {m_fde:8.4f} {d["count"]/n_total*100:9.1f}%')

    # Direction error
    dir_arr = np.array(dir_errs) if dir_errs else np.array([0])
    print(f'\n  Direction Error: mean={dir_arr.mean():.1f}deg  median={np.median(dir_arr):.1f}deg  '
          f'P95={np.percentile(dir_arr, 95):.1f}deg')

    # Speed profile error
    spd_arr = np.array(speed_errs) if speed_errs else np.array([0])
    print(f'  Speed Profile RMSE: mean={spd_arr.mean():.4f}m/s  median={np.median(spd_arr):.4f}m/s')

    return {
        'ade': ade_arr, 'fde': fde_arr, 'per_intent': per_intent,
        'dir_err': dir_arr, 'speed_err': spd_arr, 'unc_bins': unc_bins,
        'n_total': n_total, 'model_label': model_label, 'intent_names': intent_names,
    }


def plot_evaluation(results_low, results_high, out_dir):
    """Generate evaluation charts."""
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('Comprehensive Evaluation: LOW (UAV-Flow) vs HIGH (SimCruise)',
                 fontsize=14, fontweight='bold')

    # 1. ADE histogram overlay
    ax = fig.add_subplot(2, 4, 1)
    for res, color, label in [(results_low, '#2196F3', 'LOW'), (results_high, '#F44336', 'HIGH')]:
        ade = res['ade']
        ax.hist(np.clip(ade, 0, 10), bins=80, color=color, alpha=0.5, label=label, density=True)
        ax.axvline(x=np.median(ade), color=color, ls='--', lw=1.5)
    ax.set_xlabel('ADE (m)'); ax.set_ylabel('Density')
    ax.set_title('ADE Distribution'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 2. FDE histogram overlay
    ax = fig.add_subplot(2, 4, 2)
    for res, color, label in [(results_low, '#2196F3', 'LOW'), (results_high, '#F44336', 'HIGH')]:
        fde = res['fde']
        ax.hist(np.clip(fde, 0, 20), bins=80, color=color, alpha=0.5, label=label, density=True)
        ax.axvline(x=np.median(fde), color=color, ls='--', lw=1.5)
    ax.set_xlabel('FDE (m)'); ax.set_ylabel('Density')
    ax.set_title('FDE Distribution'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. Per-intent ADE bar chart — LOW
    ax = fig.add_subplot(2, 4, 3)
    pi = results_low['per_intent']
    names = [n for n in INTENT_6 if n in pi and pi[n]['count'] > 0]
    ade_means = [np.mean(pi[n]['ade']) for n in names]
    fde_means = [np.mean(pi[n]['fde']) for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, ade_means, w, color='#2196F3', alpha=0.7, label='ADE')
    ax.bar(x + w/2, fde_means, w, color='#1565C0', alpha=0.7, label='FDE')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Error (m)'); ax.set_title('LOW: Per-Intent Error')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # 4. Per-intent ADE bar chart — HIGH
    ax = fig.add_subplot(2, 4, 4)
    pi_h = results_high['per_intent']
    names_h = [n for n in INTENT_4 if n in pi_h and pi_h[n]['count'] > 0]
    ade_h = [np.mean(pi_h[n]['ade']) for n in names_h]
    fde_h = [np.mean(pi_h[n]['fde']) for n in names_h]
    x_h = np.arange(len(names_h))
    ax.bar(x_h - w/2, ade_h, w, color='#F44336', alpha=0.7, label='ADE')
    ax.bar(x_h + w/2, fde_h, w, color='#B71C1C', alpha=0.7, label='FDE')
    ax.set_xticks(x_h); ax.set_xticklabels(names_h, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Error (m)'); ax.set_title('HIGH: Per-Intent Error')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # 5. Direction error CDF
    ax = fig.add_subplot(2, 4, 5)
    for res, color, label in [(results_low, '#2196F3', 'LOW'), (results_high, '#F44336', 'HIGH')]:
        de = np.sort(res['dir_err'])
        y = np.linspace(0, 1, len(de))
        ax.plot(de, y, color=color, lw=2, label=f'{label} (med={np.median(de):.1f}deg)')
    ax.set_xlabel('Direction Error (deg)'); ax.set_ylabel('CDF')
    ax.set_xlim(0, min(180, np.percentile(results_high['dir_err'], 99)))
    ax.set_title('Direction Error CDF'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 6. Speed profile error over time
    ax = fig.add_subplot(2, 4, 6)
    for res, color, label, dt in [(results_low, '#2196F3', 'LOW', DT_LOW), (results_high, '#F44336', 'HIGH', DT_HIGH)]:
        spd = res['speed_err']
        ax.hist(np.clip(spd, 0, 20), bins=80, color=color, alpha=0.5, label=f'{label} (mean={spd.mean():.2f})', density=True)
    ax.set_xlabel('Speed RMSE (m/s)'); ax.set_ylabel('Density')
    ax.set_title('Speed Profile Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 7. Uncertainty calibration — LOW
    ax = fig.add_subplot(2, 4, 7)
    for res, color, label in [(results_low, '#2196F3', 'LOW'), (results_high, '#F44336', 'HIGH')]:
        bins = res['unc_bins']
        if not bins: continue
        bin_keys = sorted(bins.keys())
        bin_mean_err = [np.mean(bins[k]['errors']) for k in bin_keys]
        bin_mean_var = [np.mean(bins[k]['variances']) for k in bin_keys]
        ax.scatter(bin_mean_var, bin_mean_err, c=color, s=30, alpha=0.7, label=label)
    # Diagonal: perfect calibration
    all_vars = []
    for res in [results_low, results_high]:
        for k, v in res['unc_bins'].items():
            all_vars.extend(v['variances'])
    if all_vars:
        max_val = max(max(all_vars), 0.1)
        ax.plot([0, max_val], [0, np.sqrt(max_val)], 'k--', lw=1, alpha=0.5, label='Perfect (std)')
    ax.set_xlabel('Predicted Variance'); ax.set_ylabel('Actual FDE (m)')
    ax.set_title('Uncertainty Calibration'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # 8. Summary metrics table
    ax = fig.add_subplot(2, 4, 8)
    ax.axis('off')
    rows = ['ADE mean', 'ADE median', 'ADE P95', 'FDE mean', 'FDE median', 'FDE P95',
            'Dir Err mean', 'Dir Err median', 'Speed RMSE']
    low_vals = [f'{results_low["ade"].mean():.3f}', f'{np.median(results_low["ade"]):.3f}',
                f'{np.percentile(results_low["ade"], 95):.3f}',
                f'{results_low["fde"].mean():.3f}', f'{np.median(results_low["fde"]):.3f}',
                f'{np.percentile(results_low["fde"], 95):.3f}',
                f'{results_low["dir_err"].mean():.1f}deg', f'{np.median(results_low["dir_err"]):.1f}deg',
                f'{results_low["speed_err"].mean():.3f}']
    high_vals = [f'{results_high["ade"].mean():.3f}', f'{np.median(results_high["ade"]):.3f}',
                 f'{np.percentile(results_high["ade"], 95):.3f}',
                 f'{results_high["fde"].mean():.3f}', f'{np.median(results_high["fde"]):.3f}',
                 f'{np.percentile(results_high["fde"], 95):.3f}',
                 f'{results_high["dir_err"].mean():.1f}deg', f'{np.median(results_high["dir_err"]):.1f}deg',
                 f'{results_high["speed_err"].mean():.3f}']
    tbl = ax.table(cellText=list(zip(low_vals, high_vals)),
                   rowLabels=rows, colLabels=['LOW', 'HIGH'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for j in range(2):
        tbl[(0, j)].set_facecolor('#EEEEEE'); tbl[(0, j)].get_text().set_fontweight('bold')
    ax.set_title('Summary Metrics', fontweight='bold', fontsize=10)

    out_path = out_dir / 'eval_comprehensive.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out_path}')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(__file__).resolve().parents[1] / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    p = DronePredictor(device=device)

    # ---- LOW ----
    print('Loading LOW test set...')
    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    results_low = evaluate_model(ds_low, p.low, 'LOW', INTENT_6, DT_LOW, device)

    # ---- HIGH ----
    print('\nLoading HIGH test set...')
    ds_high = FastWindowDataset('../SimCruise', split='test', label_remap={4: 3})
    results_high = evaluate_model(ds_high, p.high, 'HIGH', INTENT_4, DT_HIGH, device)

    # ---- Plot ----
    plot_evaluation(results_low, results_high, out_dir)

    # ---- Save raw metrics as JSON for reference ----
    import json
    metrics = {
        'LOW': {
            'ade_mean': float(results_low['ade'].mean()),
            'ade_median': float(np.median(results_low['ade'])),
            'ade_p95': float(np.percentile(results_low['ade'], 95)),
            'fde_mean': float(results_low['fde'].mean()),
            'fde_median': float(np.median(results_low['fde'])),
            'fde_p95': float(np.percentile(results_low['fde'], 95)),
            'dir_err_mean': float(results_low['dir_err'].mean()),
            'speed_rmse_mean': float(results_low['speed_err'].mean()),
            'per_intent': {k: {'count': v['count'], 'ade_mean': float(np.mean(v['ade'])),
                               'fde_mean': float(np.mean(v['fde']))}
                          for k, v in results_low['per_intent'].items()},
        },
        'HIGH': {
            'ade_mean': float(results_high['ade'].mean()),
            'ade_median': float(np.median(results_high['ade'])),
            'ade_p95': float(np.percentile(results_high['ade'], 95)),
            'fde_mean': float(results_high['fde'].mean()),
            'fde_median': float(np.median(results_high['fde'])),
            'fde_p95': float(np.percentile(results_high['fde'], 95)),
            'dir_err_mean': float(results_high['dir_err'].mean()),
            'speed_rmse_mean': float(results_high['speed_err'].mean()),
            'per_intent': {k: {'count': v['count'], 'ade_mean': float(np.mean(v['ade'])),
                               'fde_mean': float(np.mean(v['fde']))}
                          for k, v in results_high['per_intent'].items()},
        },
    }
    json_path = out_dir / 'eval_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved: {json_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
