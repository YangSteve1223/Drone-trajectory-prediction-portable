#!/usr/bin/env python3
"""
Multi-Hypothesis Final Evaluation — test set, full metrics, comparison charts.
"""
import torch, numpy as np, sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.constrained_layout.use': True, 'font.size': 8})

INTENT_4 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'DESCEND']
INTENT_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
OUT_DIR = Path(__file__).parent / 'pic-results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_single_model():
    """Load original single-head HIGH model."""
    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=4,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load('weights/high_speed_4class.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def load_multi_model():
    """Load multi-hypothesis HIGH model."""
    model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=4,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load('weights/high_speed_4class.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model._norm_input = False
    model._get_scale_pos = lambda: 100.0

    def _normalize(hist):
        scale = hist.new_tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        return hist / scale.unsqueeze(0).unsqueeze(0)
    model._normalize = _normalize

    # Replace with multi-head
    multi_dec = model.ua_pgd.replace_with_multi_head(K=5, noise_std=0.0)
    mh_ckpt = torch.load('weights/high_multihead_K5.pth', map_location=DEVICE, weights_only=False)
    multi_dec.load_state_dict(mh_ckpt['multi_decoder_state'])
    multi_dec = multi_dec.to(DEVICE)
    return model


@torch.no_grad()
def eval_single(model, loader):
    """Evaluate single-head model."""
    all_ade, all_fde = [], []
    per_intent = defaultdict(lambda: {'ade': [], 'fde': [], 'count': 0})

    for hist, target, intent in loader:
        hist = hist.to(DEVICE); target = target.to(DEVICE)
        out = model(hist, force_predict=True)
        preds = out['predictions']
        # Z correction
        intent_prob = torch.softmax(out['intent_logits'], dim=-1)
        descend_prob = intent_prob[:, 3]
        dampen = torch.ones(hist.shape[0], device=DEVICE)
        strong = descend_prob < 0.05; weak = (descend_prob >= 0.05) & (descend_prob < 0.20)
        if strong.any(): dampen[strong] = 0.05
        if weak.any():
            t = (descend_prob[weak] - 0.05) / 0.15
            dampen[weak] = 0.30 + 0.70 * t
        apply = dampen < 1.0
        if apply.any(): preds[apply, :, 2] *= dampen[apply].view(-1, 1)

        last_pos = hist[:, -1:, :3]
        target_rel = target[:, :, :3] - last_pos
        step_errs = torch.norm(preds - target_rel, dim=-1)
        ade = step_errs.mean(dim=1).cpu().tolist()
        fde = step_errs[:, -1].cpu().tolist()
        all_ade.extend(ade); all_fde.extend(fde)
        for i in range(hist.shape[0]):
            name = INTENT_4[intent[i].item()] if intent[i].item() < 4 else '?'
            per_intent[name]['ade'].append(ade[i])
            per_intent[name]['fde'].append(fde[i])
            per_intent[name]['count'] += 1

    return {
        'ade': np.array(all_ade), 'fde': np.array(all_fde),
        'per_intent': dict(per_intent),
    }


@torch.no_grad()
def eval_multi(model, loader):
    """Evaluate multi-head model with minADE_5/minFDE_5."""
    all_min_ade, all_min_fde = [], []
    all_single_ade, all_single_fde = [], []
    per_intent = defaultdict(lambda: {'min_ade': [], 'min_fde': [], 'single_fde': [], 'count': 0})

    for hist, target, intent in loader:
        hist = hist.to(DEVICE); target = target.to(DEVICE)
        hn = model._normalize(hist)
        enc = model.emam_se(hn)
        dtp = model.ia_dtp(enc, historical_trajectory=hn)
        mh = model.ua_pgd.forward_multi_head(
            encoded_feat=enc, global_anchor=dtp['global_anchor'],
            historical_trajectory=hn, intent_weights=dtp['intent_weights'])

        all_preds = mh['all_predictions']  # (K, B, P, 3)
        best_pred = mh['predictions']       # (B, P, 3)
        last_pos = hist[:, -1:, :3]
        target_rel = target[:, :, :3] - last_pos

        metrics = MultiHeadNeuralDecoder.compute_minade_fde(all_preds, target_rel)
        all_min_ade.extend(metrics['min_ade'].cpu().tolist())
        all_min_fde.extend(metrics['min_fde'].cpu().tolist())

        step_single = torch.norm(best_pred - target_rel, dim=-1)
        all_single_ade.extend(step_single.mean(dim=1).cpu().tolist())
        all_single_fde.extend(step_single[:, -1].cpu().tolist())

        for i in range(hist.shape[0]):
            name = INTENT_4[intent[i].item()] if intent[i].item() < 4 else '?'
            per_intent[name]['min_ade'].append(metrics['min_ade'][i].item())
            per_intent[name]['min_fde'].append(metrics['min_fde'][i].item())
            per_intent[name]['single_fde'].append(step_single[:, -1][i].item())
            per_intent[name]['count'] += 1

    return {
        'min_ade': np.array(all_min_ade), 'min_fde': np.array(all_min_fde),
        'single_ade': np.array(all_single_ade), 'single_fde': np.array(all_single_fde),
        'per_intent': dict(per_intent),
    }


def plot_results(single, multi, out_dir):
    """Generate comparison charts."""
    single_fde_mean = single['fde'].mean()
    multi_fde_mean = multi['min_fde'].mean()
    single_desc_fde = np.mean(single['per_intent']['DESCEND']['fde'])
    multi_desc_fde = np.mean(multi['per_intent']['DESCEND']['min_fde'])

    # ── Chart 1: FDE Distribution Overlay ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Multi-Hypothesis (K=5) vs Single Model — HIGH (SimCruise) Test Set',
                 fontsize=13, fontweight='bold')

    # 1a. FDE histogram
    ax = axes[0, 0]
    ax.hist(np.clip(single['fde'], 0, 30), bins=100, color='#F44336', alpha=0.5,
            density=True, label=f'Single (mean={single_fde_mean:.2f}m)')
    ax.hist(np.clip(multi['min_fde'], 0, 30), bins=100, color='#4CAF50', alpha=0.5,
            density=True, label=f'Multi K=5 (mean={multi_fde_mean:.2f}m)')
    ax.axvline(x=np.median(single['fde']), color='#F44336', ls='--', lw=1.5)
    ax.axvline(x=np.median(multi['min_fde']), color='#4CAF50', ls='--', lw=1.5)
    ax.set_xlabel('FDE (m)'); ax.set_ylabel('Density')
    ax.set_title('FDE Distribution: Single vs Multi-Hypothesis')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 1b. ADE histogram
    ax = axes[0, 1]
    ax.hist(np.clip(single['ade'], 0, 8), bins=100, color='#F44336', alpha=0.5,
            density=True, label=f'Single (mean={single["ade"].mean():.2f}m)')
    ax.hist(np.clip(multi['min_ade'], 0, 8), bins=100, color='#4CAF50', alpha=0.5,
            density=True, label=f'Multi K=5 (mean={multi["min_ade"].mean():.2f}m)')
    ax.axvline(x=np.median(single['ade']), color='#F44336', ls='--', lw=1.5)
    ax.axvline(x=np.median(multi['min_ade']), color='#4CAF50', ls='--', lw=1.5)
    ax.set_xlabel('ADE (m)'); ax.set_ylabel('Density')
    ax.set_title('ADE Distribution: Single vs Multi-Hypothesis')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 1c. Per-intent FDE comparison bar chart
    ax = axes[0, 2]
    names = INTENT_4
    single_vals = [np.mean(single['per_intent'][n]['fde']) for n in names]
    multi_vals = [np.mean(multi['per_intent'][n]['min_fde']) for n in names]
    x = np.arange(len(names)); w = 0.35
    bars1 = ax.bar(x - w/2, single_vals, w, color='#F44336', alpha=0.7, label='Single')
    bars2 = ax.bar(x + w/2, multi_vals, w, color='#4CAF50', alpha=0.7, label='Multi K=5')
    for i, (s, m) in enumerate(zip(single_vals, multi_vals)):
        imp = (s - m) / s * 100
        ax.text(i, max(s, m) + 0.3, f'-{imp:.0f}%', ha='center', fontsize=8, fontweight='bold', color='#2E7D32')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('FDE (m)'); ax.set_title('Per-Intent FDE Comparison')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # 2a. DESCEND FDE distribution
    ax = axes[1, 0]
    desc_single = np.clip(np.array(single['per_intent']['DESCEND']['fde']), 0, 60)
    desc_multi = np.clip(np.array(multi['per_intent']['DESCEND']['min_fde']), 0, 60)
    ax.hist(desc_single, bins=80, color='#F44336', alpha=0.5, density=True,
            label=f'Single (mean={single_desc_fde:.2f}m)')
    ax.hist(desc_multi, bins=80, color='#4CAF50', alpha=0.5, density=True,
            label=f'Multi K=5 (mean={multi_desc_fde:.2f}m)')
    ax.axvline(x=np.median(desc_single), color='#F44336', ls='--', lw=1.5)
    ax.axvline(x=np.median(desc_multi), color='#4CAF50', ls='--', lw=1.5)
    ax.set_xlabel('FDE (m)'); ax.set_ylabel('Density')
    ax.set_title(f'DESCEND FDE Distribution (improvement: {(single_desc_fde-multi_desc_fde)/single_desc_fde*100:.0f}%)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 2b. Improvement scatter: each sample's FDE before vs after
    ax = axes[1, 1]
    # Sample a subset for scatter
    n_sample = min(5000, len(single['fde']))
    idx = np.random.choice(len(single['fde']), n_sample, replace=False)
    s_fde = single['fde'][idx]; m_fde = multi['min_fde'][idx]
    max_val = max(np.percentile(s_fde, 99), np.percentile(m_fde, 99))
    ax.scatter(s_fde, m_fde, c='#2196F3', s=3, alpha=0.3, edgecolors='none')
    ax.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label='No improvement')
    ax.fill_between([0, max_val], [0, max_val], 0, alpha=0.1, color='#4CAF50',
                     label='Improved region')
    ax.set_xlabel('Single FDE (m)'); ax.set_ylabel('Multi K=5 minFDE (m)')
    ax.set_title(f'Per-Sample FDE: Single vs Multi-Hypothesis (n={n_sample})')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)

    # 2c. Summary table
    ax = axes[1, 2]
    ax.axis('off')
    rows = ['ADE mean', 'ADE median', 'ADE P95',
            'FDE mean', 'FDE median', 'FDE P95',
            'STRAIGHT FDE', 'TURN_L FDE', 'TURN_R FDE', 'DESCEND FDE']
    s_vals = [
        f'{single["ade"].mean():.3f}', f'{np.median(single["ade"]):.3f}',
        f'{np.percentile(single["ade"], 95):.3f}',
        f'{single["fde"].mean():.3f}', f'{np.median(single["fde"]):.3f}',
        f'{np.percentile(single["fde"], 95):.3f}',
        f'{np.mean(single["per_intent"]["STRAIGHT"]["fde"]):.3f}',
        f'{np.mean(single["per_intent"]["TURN_L"]["fde"]):.3f}',
        f'{np.mean(single["per_intent"]["TURN_R"]["fde"]):.3f}',
        f'{single_desc_fde:.3f}',
    ]
    m_vals = [
        f'{multi["min_ade"].mean():.3f}', f'{np.median(multi["min_ade"]):.3f}',
        f'{np.percentile(multi["min_ade"], 95):.3f}',
        f'{multi["min_fde"].mean():.3f}', f'{np.median(multi["min_fde"]):.3f}',
        f'{np.percentile(multi["min_fde"], 95):.3f}',
        f'{np.mean(multi["per_intent"]["STRAIGHT"]["min_fde"]):.3f}',
        f'{np.mean(multi["per_intent"]["TURN_L"]["min_fde"]):.3f}',
        f'{np.mean(multi["per_intent"]["TURN_R"]["min_fde"]):.3f}',
        f'{multi_desc_fde:.3f}',
    ]

    # Add improvement column
    imp_vals = []
    s_nums = [single["ade"].mean(), np.median(single["ade"]), np.percentile(single["ade"], 95),
              single["fde"].mean(), np.median(single["fde"]), np.percentile(single["fde"], 95),
              np.mean(single["per_intent"]["STRAIGHT"]["fde"]),
              np.mean(single["per_intent"]["TURN_L"]["fde"]),
              np.mean(single["per_intent"]["TURN_R"]["fde"]),
              single_desc_fde]
    m_nums = [multi["min_ade"].mean(), np.median(multi["min_ade"]), np.percentile(multi["min_ade"], 95),
              multi["min_fde"].mean(), np.median(multi["min_fde"]), np.percentile(multi["min_fde"], 95),
              np.mean(multi["per_intent"]["STRAIGHT"]["min_fde"]),
              np.mean(multi["per_intent"]["TURN_L"]["min_fde"]),
              np.mean(multi["per_intent"]["TURN_R"]["min_fde"]),
              multi_desc_fde]
    for s, m in zip(s_nums, m_nums):
        imp_vals.append(f'{(s-m)/s*100:+.1f}%')

    cell_text = []
    for sv, mv, iv in zip(s_vals, m_vals, imp_vals):
        cell_text.append([sv, mv, iv])
    tbl = ax.table(cellText=cell_text, rowLabels=rows,
                   colLabels=['Single', 'Multi K=5', 'Improve'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
    for j in range(3):
        tbl[(0, j)].set_facecolor('#EEEEEE')
        tbl[(0, j)].get_text().set_fontweight('bold')
    ax.set_title('Full Metrics Comparison', fontweight='bold', fontsize=10)

    out_path = out_dir / 'eval_multihypothesis.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    # ── Chart 2: Per-intent error growth over time ─────────────────
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    fig2.suptitle('Error Growth Over Prediction Horizon — Single vs Multi-Hypothesis',
                  fontsize=12, fontweight='bold')

    # We'll compute this from raw predictions
    for ax_i, intent_name in enumerate(INTENT_4):
        ax = axes2[ax_i]
        # Placeholder data from our evaluation
        s_count = single['per_intent'][intent_name]['count']
        m_count = multi['per_intent'][intent_name]['count']
        ax.text(0.5, 0.5, f'{intent_name}\nSingle n={s_count}\nMulti n={m_count}',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title(intent_name)

    out_path2 = out_dir / 'eval_multihypothesis_error_growth.png'
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path2}')
    plt.close()

    return {
        'single_ade_mean': float(single['ade'].mean()),
        'single_fde_mean': float(single_fde_mean),
        'multi_min_ade': float(multi['min_ade'].mean()),
        'multi_min_fde': float(multi_fde_mean),
        'single_desc_fde': float(single_desc_fde),
        'multi_desc_fde': float(multi_desc_fde),
    }


def main():
    print(f'Device: {DEVICE}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test data ──────────────────────────────────────────
    print('Loading test set...')
    ds = FastWindowDataset('../SimCruise', split='test', label_remap={4: 3})
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    print(f'  {len(ds):,} test samples')

    # ── Single model eval ───────────────────────────────────────
    print('\nEvaluating single model...')
    single_model = load_single_model()
    single_results = eval_single(single_model, loader)

    print(f'  Single ADE: {single_results["ade"].mean():.4f}m  '
          f'FDE: {single_results["fde"].mean():.4f}m  '
          f'median: {np.median(single_results["fde"]):.4f}m')
    for name in INTENT_4:
        d = single_results['per_intent'][name]
        print(f'    {name:12s}: FDE={np.mean(d["fde"]):.4f}m  (n={d["count"]})')

    # ── Multi-head model eval ───────────────────────────────────
    print('\nEvaluating multi-hypothesis model (K=5)...')
    multi_model = load_multi_model()
    multi_results = eval_multi(multi_model, loader)

    print(f'  Multi minADE_5: {multi_results["min_ade"].mean():.4f}m  '
          f'minFDE_5: {multi_results["min_fde"].mean():.4f}m  '
          f'median: {np.median(multi_results["min_fde"]):.4f}m')
    print(f'  Multi single-best ADE: {multi_results["single_ade"].mean():.4f}m  '
          f'FDE: {multi_results["single_fde"].mean():.4f}m')
    for name in INTENT_4:
        d = multi_results['per_intent'][name]
        s_fde = np.mean(single_results['per_intent'][name]['fde'])
        m_fde = np.mean(d['min_fde'])
        imp = (s_fde - m_fde) / s_fde * 100
        print(f'    {name:12s}: minFDE={m_fde:.4f}m  '
              f'(single={s_fde:.4f}m, {imp:+.1f}%)  (n={d["count"]})')

    # ── Overall improvement ─────────────────────────────────────
    s_ade = single_results['ade'].mean()
    s_fde = single_results['fde'].mean()
    m_ade = multi_results['min_ade'].mean()
    m_fde = multi_results['min_fde'].mean()
    print(f'\n{"="*60}')
    print(f'  ADE: {s_ade:.4f} -> {m_ade:.4f}m ({(s_ade-m_ade)/s_ade*100:+.1f}%)')
    print(f'  FDE: {s_fde:.4f} -> {m_fde:.4f}m ({(s_fde-m_fde)/s_fde*100:+.1f}%)')
    s_desc = np.mean(single_results['per_intent']['DESCEND']['fde'])
    m_desc = np.mean(multi_results['per_intent']['DESCEND']['min_fde'])
    print(f'  DESCEND FDE: {s_desc:.2f} -> {m_desc:.2f}m ({(s_desc-m_desc)/s_desc*100:+.1f}%)')
    print(f'{"="*60}')

    # ── Plots ───────────────────────────────────────────────────
    print('\nGenerating charts...')
    metrics = plot_results(single_results, multi_results, OUT_DIR)

    # ── Save metrics JSON ───────────────────────────────────────
    eval_metrics = {
        'model': 'HIGH (SimCruise)',
        'K': 5,
        'test_samples': len(ds),
        'single': {
            'ade_mean': float(single_results['ade'].mean()),
            'ade_median': float(np.median(single_results['ade'])),
            'ade_p95': float(np.percentile(single_results['ade'], 95)),
            'fde_mean': float(single_results['fde'].mean()),
            'fde_median': float(np.median(single_results['fde'])),
            'fde_p95': float(np.percentile(single_results['fde'], 95)),
            'per_intent': {n: {'fde_mean': float(np.mean(single_results['per_intent'][n]['fde'])),
                               'count': single_results['per_intent'][n]['count']}
                          for n in INTENT_4},
        },
        'multi_k5': {
            'min_ade_mean': float(multi_results['min_ade'].mean()),
            'min_ade_median': float(np.median(multi_results['min_ade'])),
            'min_ade_p95': float(np.percentile(multi_results['min_ade'], 95)),
            'min_fde_mean': float(multi_results['min_fde'].mean()),
            'min_fde_median': float(np.median(multi_results['min_fde'])),
            'min_fde_p95': float(np.percentile(multi_results['min_fde'], 95)),
            'single_best_fde_mean': float(multi_results['single_fde'].mean()),
            'per_intent': {n: {'min_fde_mean': float(np.mean(multi_results['per_intent'][n]['min_fde'])),
                               'count': multi_results['per_intent'][n]['count']}
                          for n in INTENT_4},
        },
        'improvement': {
            'ade_pct': float((s_ade - m_ade) / s_ade * 100),
            'fde_pct': float((s_fde - m_fde) / s_fde * 100),
            'descend_fde_pct': float((s_desc - m_desc) / s_desc * 100),
        },
    }
    json_path = OUT_DIR / 'eval_multihypothesis.json'
    with open(json_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f'Saved: {json_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
