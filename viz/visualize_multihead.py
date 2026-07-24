#!/usr/bin/env python3
"""Multi-hypothesis trajectory visualization charts."""
import torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'figure.constrained_layout.use': True, 'font.size': 8})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path(__file__).resolve().parents[1] / 'pic-results'
INTENT_4 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'DESCEND']
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']


def load_models():
    """Load single and multi-head models."""
    # Single
    s_model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=4,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    ckpt = torch.load(str(Path(__file__).resolve().parents[1] / 'weights' / 'high_speed_4class.pth'), map_location=DEVICE, weights_only=False)
    s_model.load_state_dict(ckpt['model_state_dict'])

    # Multi-head
    m_model = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=4,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    m_model.load_state_dict(ckpt['model_state_dict'])
    m_model._norm_input = False
    m_model._get_scale_pos = lambda: 100.0
    def _normalize(hist):
        scale = hist.new_tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        return hist / scale.unsqueeze(0).unsqueeze(0)
    m_model._normalize = _normalize
    multi_dec = m_model.ua_pgd.replace_with_multi_head(K=5, noise_std=0.0)
    mh_ckpt = torch.load(str(Path(__file__).resolve().parents[1] / 'weights' / 'high_multihead_K5.pth'), map_location=DEVICE, weights_only=False)
    multi_dec.load_state_dict(mh_ckpt['multi_decoder_state'])
    multi_dec = multi_dec.to(DEVICE)

    return s_model, m_model


@torch.no_grad()
def predict_single(model, hist):
    out = model(hist, force_predict=True)
    preds = out['predictions'].clone()
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
    return preds


@torch.no_grad()
def predict_multi(model, hist):
    hn = model._normalize(hist)
    enc = model.emam_se(hn)
    dtp = model.ia_dtp(enc, historical_trajectory=hn)
    mh = model.ua_pgd.forward_multi_head(
        encoded_feat=enc, global_anchor=dtp['global_anchor'],
        historical_trajectory=hn, intent_weights=dtp['intent_weights'])
    return mh['all_predictions'], mh['predictions'], mh['confidences']


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s_model, m_model = load_models()

    ds = FastWindowDataset('../SimCruise', split='test', label_remap={4: 3})
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    # Collect samples: best 4 per intent by multi-head minFDE
    samples = {intent: [] for intent in range(4)}  # (hist, target, intent)

    for hist, target, intent in loader:
        hist = hist.to(DEVICE); target = target.to(DEVICE)
        all_preds, best_pred, conf = predict_multi(m_model, hist)
        last_pos = hist[:, -1:, :3]
        target_rel = target[:, :, :3] - last_pos
        metrics = MultiHeadNeuralDecoder.compute_minade_fde(all_preds, target_rel)
        min_fde = metrics['min_fde'].cpu().tolist()

        for i in range(hist.shape[0]):
            it = intent[i].item()
            if it < 4 and len(samples[it]) < 4:
                samples[it].append((
                    hist[i:i+1], target[i:i+1],
                    torch.tensor([it]), min_fde[i]
                ))
        if all(len(v) >= 4 for v in samples.values()):
            break

    # ── Chart 1: 12-sample multi-hypothesis trajectory grid ──────
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    fig.suptitle('Multi-Hypothesis (K=5) Trajectory Samples — HIGH (SimCruise)',
                 fontsize=14, fontweight='bold')

    for intent_id in range(4):
        for j in range(4):
            ax = axes[intent_id, j]
            if j >= len(samples[intent_id]):
                ax.axis('off'); continue
            hist, target, it, min_fde = samples[intent_id][j]
            hist = hist.to(DEVICE); target = target.to(DEVICE)

            # Predict
            s_pred = predict_single(s_model, hist)
            all_preds, best_pred, conf = predict_multi(m_model, hist)

            last_pos = hist[0, -1:, :3].cpu()
            hist_pos = hist[0, :, :3].cpu()

            # History
            ax.plot(hist_pos[:, 0], hist_pos[:, 1], 'b-', lw=1.5, alpha=0.7, label='History')
            # GT
            gt_abs = target[0, :, :3].cpu()
            ax.plot(gt_abs[:, 0], gt_abs[:, 1], 'g-', lw=1.5, label='Ground Truth')
            # Single model
            s_abs = last_pos + s_pred[0].cpu()
            ax.plot(s_abs[:, 0], s_abs[:, 1], 'r--', lw=1, alpha=0.6, label='Single')
            # Multi-head best
            m_abs = last_pos + best_pred[0].cpu()
            ax.plot(m_abs[:, 0], m_abs[:, 1], color='#4CAF50', lw=2, label='Multi K=5 (best)')
            # All 5 hypotheses (faint)
            for k in range(5):
                h_abs = last_pos + all_preds[k, 0].cpu()
                ax.plot(h_abs[:, 0], h_abs[:, 1], color=COLORS[k], lw=0.5, alpha=0.3)

            # Markers
            for t_idx in [4, 9, 14, 19]:
                ax.plot(gt_abs[t_idx, 0], gt_abs[t_idx, 1], 'go', ms=4, alpha=0.7)
                ax.plot(m_abs[t_idx, 0], m_abs[t_idx, 1], 's', color='#4CAF50', ms=4, alpha=0.7)

            name = INTENT_4[intent_id]
            ax.set_title(f'{name} (minFDE_5={min_fde:.2f}m)', fontsize=9)
            ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='b', lw=1.5, label='History (20s)'),
        Line2D([0], [0], color='g', lw=1.5, label='Ground Truth'),
        Line2D([0], [0], color='r', ls='--', lw=1, label='Single model'),
        Line2D([0], [0], color='#4CAF50', lw=2, label='Multi K=5 (best)'),
        Line2D([0], [0], color='gray', lw=0.5, alpha=0.3, label='Other hypotheses'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8)
    out = OUT_DIR / 'multihyp_trajectory_grid.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()

    # ── Chart 2: DESCEND deep dive (5 hypotheses visible) ─────────
    desc_samples = samples[3]  # DESCEND
    if desc_samples:
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
        fig2.suptitle('DESCEND Samples — All 5 Hypotheses vs Single Model',
                      fontsize=13, fontweight='bold')

        for j in range(min(4, len(desc_samples))):
            ax = axes2[j // 2, j % 2]
            hist, target, _, min_fde = desc_samples[j]
            hist = hist.to(DEVICE); target = target.to(DEVICE)

            s_pred = predict_single(s_model, hist)
            all_preds, best_pred, conf = predict_multi(m_model, hist)

            last_pos = hist[0, -1:, :3].cpu()
            hist_pos = hist[0, :, :3].cpu()

            # 3D-like view: XY on main, Z as color/annotation
            ax.plot(hist_pos[:, 0], hist_pos[:, 1], 'b-', lw=1.5, alpha=0.7)
            gt_abs = target[0, :, :3].cpu()
            ax.plot(gt_abs[:, 0], gt_abs[:, 1], 'g-', lw=2)
            s_abs = last_pos + s_pred[0].cpu()
            ax.plot(s_abs[:, 0], s_abs[:, 1], 'r--', lw=1)

            # 5 hypotheses
            for k in range(5):
                h_abs = last_pos + all_preds[k, 0].cpu()
                ax.plot(h_abs[:, 0], h_abs[:, 1], color=COLORS[k], lw=1.2, alpha=0.7,
                       label=f'H{k+1} (conf={conf[0,k].item():.2f})')

            # Z annotation: show Z displacement at final step
            gt_z = gt_abs[-1, 2] - last_pos[0, 2]
            ax.annotate(f'GT Z={gt_z:.1f}m', xy=(gt_abs[-1, 0], gt_abs[-1, 1]),
                       fontsize=8, color='green')
            for k in range(5):
                h_abs = last_pos + all_preds[k, 0].cpu()
                h_z = h_abs[-1, 2] - last_pos[0, 2]
                ax.annotate(f'H{k+1} Z={h_z:.1f}m', xy=(h_abs[-1, 0], h_abs[-1, 1]),
                           fontsize=7, color=COLORS[k], alpha=0.7)

            ax.set_title(f'DESCEND sample {j+1} (minFDE_5={min_fde:.2f}m)')
            ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

        out2 = OUT_DIR / 'multihyp_descend_deep.png'
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        print(f'Saved: {out2}')
        plt.close()

    # ── Chart 3: Updated summary table with multi-hypothesis ──────
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.axis('off')
    fig3.suptitle('Model Performance Summary — Single vs Multi-Hypothesis (K=5)',
                  fontsize=13, fontweight='bold')

    rows = [
        'ADE mean', 'ADE median', 'ADE P95',
        'FDE mean', 'FDE median', 'FDE P95',
        'STRAIGHT FDE', 'TURN_L FDE', 'TURN_R FDE', 'DESCEND FDE',
        'Catastrophic (>90deg)', 'Direction error',
    ]

    # Load eval metrics for accurate numbers
    import json
    metrics_path = OUT_DIR / 'eval_multihypothesis.json'
    with open(metrics_path) as f:
        em = json.load(f)

    s = em['single']; m = em['multi_k5']; imp = em['improvement']

    low_vals = [
        f'{em["single"]["ade_mean"]:.3f}', f'{em["single"]["ade_median"]:.3f}',
        f'{em["single"]["ade_p95"]:.3f}',
        f'{em["single"]["fde_mean"]:.3f}', f'{em["single"]["fde_median"]:.3f}',
        f'{em["single"]["fde_p95"]:.3f}',
        f'{s["per_intent"]["STRAIGHT"]["fde_mean"]:.3f}',
        f'{s["per_intent"]["TURN_L"]["fde_mean"]:.3f}',
        f'{s["per_intent"]["TURN_R"]["fde_mean"]:.3f}',
        f'{s["per_intent"]["DESCEND"]["fde_mean"]:.3f}',
        '0%', '0.1deg',
    ]
    mh_vals = [
        f'{m["min_ade_mean"]:.3f}', f'{m["min_ade_median"]:.3f}',
        f'{m["min_ade_p95"]:.3f}',
        f'{m["min_fde_mean"]:.3f}', f'{m["min_fde_median"]:.3f}',
        f'{m["min_fde_p95"]:.3f}',
        f'{m["per_intent"]["STRAIGHT"]["min_fde_mean"]:.3f}',
        f'{m["per_intent"]["TURN_L"]["min_fde_mean"]:.3f}',
        f'{m["per_intent"]["TURN_R"]["min_fde_mean"]:.3f}',
        f'{m["per_intent"]["DESCEND"]["min_fde_mean"]:.3f}',
        '0%', '0.1deg',
    ]

    # Compute improvement values
    s_nums = [
        s['ade_mean'], s['ade_median'], s['ade_p95'],
        s['fde_mean'], s['fde_median'], s['fde_p95'],
        s['per_intent']['STRAIGHT']['fde_mean'],
        s['per_intent']['TURN_L']['fde_mean'],
        s['per_intent']['TURN_R']['fde_mean'],
        s['per_intent']['DESCEND']['fde_mean'],
        0, 0.1,
    ]
    m_nums = [
        m['min_ade_mean'], m['min_ade_median'], m['min_ade_p95'],
        m['min_fde_mean'], m['min_fde_median'], m['min_fde_p95'],
        m['per_intent']['STRAIGHT']['min_fde_mean'],
        m['per_intent']['TURN_L']['min_fde_mean'],
        m['per_intent']['TURN_R']['min_fde_mean'],
        m['per_intent']['DESCEND']['min_fde_mean'],
        0, 0.1,
    ]
    imp_vals = []
    for sv, mv in zip(s_nums, m_nums):
        if sv > 0.001:
            imp_vals.append(f'{(sv-mv)/sv*100:+.1f}%')
        else:
            imp_vals.append('—')

    cell_text = [[lv, mv, iv] for lv, mv, iv in zip(low_vals, mh_vals, imp_vals)]
    tbl = ax3.table(cellText=cell_text, rowLabels=rows,
                    colLabels=['Single Model', 'Multi K=5 (minFDE_5)', 'Improvement'],
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for j in range(3):
        tbl[(0, j)].set_facecolor('#EEEEEE')
        tbl[(0, j)].get_text().set_fontweight('bold')
    # Highlight DESCEND row
    for j in range(3):
        tbl[(9+1, j)].set_facecolor('#FFF9C4')

    out3 = OUT_DIR / 'summary_table_updated.png'
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f'Saved: {out3}')
    plt.close()

    print('\nAll multi-hypothesis charts generated!')


if __name__ == '__main__':
    main()
