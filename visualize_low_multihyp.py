#!/usr/bin/env python3
"""
LOW Multi-Hypothesis Trajectory Visualization — 24 Best Samples.
Always filters dead heads. Splits 24 samples across 3 pages (8 each) for readability.
"""
import torch, numpy as np, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from emam_model import TrajectoryPredictor
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path(__file__).parent / 'pic-results'; OUT.mkdir(parents=True, exist_ok=True)

INTENT_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
MH_C = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
HC, GC, BC = '#37474F', '#00C853', '#00E676'
LSEG = [(0, 5, '0-1s'), (5, 10, '1-2s'), (10, 15, '2-3s'), (15, 20, '3-4s')]
CONF_THRESHOLD = 0.01  # dead head threshold


# ═══════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════
def load_low_multi():
    m = TrajectoryPredictor(
        input_dim=6, history_len=20, pred_len=20,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE).eval()
    c = torch.load('weights/low_speed_6class.pth', map_location=DEVICE, weights_only=False)
    m.load_state_dict(c['model_state_dict'])
    m._norm_input = False
    m._get_scale_pos = lambda: 100.0
    def _n(h):
        s = h.new_tensor([100., 100., 100., 10., 10., 10.])
        return h / s.unsqueeze(0).unsqueeze(0)
    m._normalize = _n
    m.ua_pgd.replace_with_multi_head(K=5, noise_std=0.0)
    mc = torch.load('weights/low_multihead_K5.pth', map_location=DEVICE, weights_only=False)
    m.ua_pgd.neural_decoder.load_state_dict(mc['multi_decoder_state'])
    m.ua_pgd.neural_decoder = m.ua_pgd.neural_decoder.to(DEVICE)
    return m


@torch.no_grad()
def pred_low_multi(m, h):
    hn = m._normalize(h)
    enc = m.emam_se(hn)
    dtp = m.ia_dtp(enc, historical_trajectory=hn)
    mh = m.ua_pgd.forward_multi_head(
        encoded_feat=enc, global_anchor=dtp['global_anchor'],
        historical_trajectory=hn, intent_weights=dtp['intent_weights'])
    all_preds = mh['all_predictions']   # (K, B, T, 3)
    conf = mh['confidences']             # (B, K)
    # Filter dead heads
    cp = torch.softmax(conf[0:1], dim=1)[0]
    alive = cp > CONF_THRESHOLD
    if alive.sum() < 1:
        alive[0] = True
    all_preds = all_preds[alive]
    conf = conf[:, alive]
    best_idx = conf.argmax(dim=1)
    best_pred = all_preds[best_idx, torch.arange(h.shape[0], device=DEVICE)]
    cp_out = torch.softmax(conf, dim=1)
    return all_preds, best_pred, conf, cp_out


# ═══════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════
def smoothness(t):
    if t.shape[0] < 3: return 100.0
    v = t[1:] - t[:-1]; n = np.linalg.norm(v, axis=1) + 1e-8
    vn = v / n[:, None]
    d = np.clip(np.sum(vn[1:] * vn[:-1], axis=1), -1, 1)
    return float(np.degrees(np.mean(np.abs(np.arccos(d)))))


def score_sample(hist, target, all_preds, best_pred, conf, intent):
    lp = hist[-1, :3]; ga = target[:, :3]
    K = all_preds.shape[0]
    mf = float('inf')
    for k in range(K):
        e = np.linalg.norm(lp + all_preds[k, :, :3][-1] - ga[-1])
        mf = min(mf, e)
    ba = lp + best_pred[:, :3]
    bade = float(np.mean(np.linalg.norm(ba - ga, axis=1)))
    gs = smoothness(ga)
    ext = float(np.linalg.norm(ga[-1] - ga[0]))
    cp = torch.softmax(conf, dim=1)[0].cpu().numpy()
    cs = float(np.std(cp))
    spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
    s = 0.0
    s += max(0, 5.0 - mf) * 5.0
    s += max(0, 3.0 - bade) * 3.0
    s += max(0, 25.0 - gs) * 0.3
    s += min(ext, 6.0) * 2.0
    s += cs * 10.0
    s += min(spd, 2.5) * 2.0
    s -= abs(mf - bade) * 0.5
    if intent in (3, 4): s += 5.0
    return s, mf


def seg_errs(pa, ga, segs):
    return [(lbl, np.linalg.norm(pa[s:e] - ga[s:e], axis=1)) for s, e, lbl in segs]


# ═══════════════════════════════════════════════════════════════════════
#  Collect
# ═══════════════════════════════════════════════════════════════════════
def collect(n_tgt=200):
    print(f'Collecting samples (target {n_tgt})...')
    m = load_low_multi()
    ds = FastWindowDataset('../UAV-Flow-pure', split='test')
    ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    scored = []; seen = 0
    for hb, tb, ib in ld:
        hb = hb.to(DEVICE)
        ap, bp, cf, cp = pred_low_multi(m, hb)
        for i in range(hb.shape[0]):
            hist = hb[i].cpu().numpy(); tgt = tb[i].cpu().numpy()
            a = ap[:, i].cpu().numpy(); b = bp[i].cpu().numpy()
            c = cf[i:i+1].cpu(); it = ib[i].item()
            if it >= 6: continue
            s, mf = score_sample(hist, tgt, a, b, c, it)
            scored.append((s, mf, hist, tgt, a, b, c, it))
            seen += 1
        if seen >= n_tgt * 3: break

    scored.sort(key=lambda x: x[0], reverse=True)
    sel = []; ic = defaultdict(int)
    for s, mf, hi, ta, a, b, c, it in scored:
        if ic[it] < 8: sel.append((hi, ta, a, b, c, it, mf)); ic[it] += 1
        if len(sel) >= 24: break
    while len(sel) < 24 and len(sel) < len(scored):
        e = scored[len(sel)]
        sel.append((e[2], e[3], e[4], e[5], e[6], e[7], e[1]))
    print(f'  Evaluated {seen}, selected {len(sel)}')
    for idx, (hi, ta, a, b, c, it, mf) in enumerate(sel):
        cp_arr = torch.softmax(c, dim=1)[0].numpy()
        print(f'  #{idx+1:2d}: {INTENT_6[it]:12s} minFDE={mf:.2f}m  conf={np.array2string(cp_arr, precision=2)}')
    return sel


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Best Trajectory 3D (24 samples → 3 pages × 8)
# ═══════════════════════════════════════════════════════════════════════
def chart_best_3d(samples):
    n_per_page = 8
    n_pages = (len(samples) + n_per_page - 1) // n_per_page
    for page in range(n_pages):
        chunk = samples[page * n_per_page:(page + 1) * n_per_page]
        rows = 2; cols = 4
        fig = plt.figure(figsize=(28, 16))
        fig.suptitle(f'LOW Model (UAV-Flow) — Best Confidence Trajectory 3D (Page {page+1}/{n_pages})\n'
                     'Dark=History(4s)  Green=Ground Truth  Green--=Best Prediction  '
                     'o=GT marker  s=Pred marker',
                     fontsize=12, fontweight='bold', y=0.99)
        for idx, (hist, target, all_preds, best_pred, conf, intent, min_fde) in enumerate(chunk):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            lp = hist[-1, :3]; hp = hist[:, :3]; ga = target[:, :3]; ba = lp + best_pred[:, :3]
            ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=HC, lw=1.5, alpha=0.7, label='History')
            ax.plot(ga[:, 0], ga[:, 1], ga[:, 2], color=GC, lw=2.5, label='GT')
            ax.plot(ba[:, 0], ba[:, 1], ba[:, 2], color=BC, lw=2, ls='--', label='Best')
            for s, e, lbl in LSEG:
                ax.scatter(*ga[e-1], c=GC, s=20, marker='o', alpha=0.9, zorder=5)
                ax.scatter(*ba[e-1], c=BC, s=20, marker='s', alpha=0.9, zorder=5)
            se = seg_errs(ba, ga, LSEG)
            es = ' | '.join([f'{l}:{np.mean(e):.2f}m' for l, e in se])
            z_errs = [np.mean(np.abs(ba[s:e, 2] - ga[s:e, 2])) for s, e, _ in LSEG]
            zs = 'Z: ' + ' | '.join([f'{lbl}:{ze:.2f}m' for ze, (_, _, lbl) in zip(z_errs, LSEG)])
            spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
            global_num = page * n_per_page + idx + 1
            ax.set_title(f'#{global_num} {INTENT_6[intent]}  spd={spd:.1f}m/s  minFDE={min_fde:.2f}m\n{es}  {zs}',
                         fontsize=6.5, family='monospace')
            # Suppress exaggerated Z-axis visual: match Z limits to XY scale
            z_all = np.concatenate([hp[:, 2], ga[:, 2], ba[:, 2]])
            xy_range = max(ga[:, 0].max() - ga[:, 0].min(), ga[:, 1].max() - ga[:, 1].min(), 0.5)
            z_mid = (z_all.max() + z_all.min()) / 2
            ax.set_zlim(z_mid - xy_range * 0.5, z_mid + xy_range * 0.5)
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w'); ax.yaxis.pane.set_edgecolor('w'); ax.zaxis.pane.set_edgecolor('w')
            ax.tick_params(labelsize=5)
            if idx == 0: ax.legend(fontsize=6, loc='upper left')
        p = OUT / f'low_24_best_3d_p{page+1}.png'
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Best Trajectory XY (24 samples → 3 pages × 8)
# ═══════════════════════════════════════════════════════════════════════
def chart_best_xy(samples):
    n_per_page = 8
    n_pages = (len(samples) + n_per_page - 1) // n_per_page
    for page in range(n_pages):
        chunk = samples[page * n_per_page:(page + 1) * n_per_page]
        fig, axs = plt.subplots(2, 4, figsize=(28, 16))
        fig.suptitle(f'LOW Model — Best Confidence Trajectory XY (Page {page+1}/{n_pages})\n'
                     'Dark=History  Green=GT  Green--=Best  o=GT marker  s=Pred marker',
                     fontsize=12, fontweight='bold')
        for idx, (hist, target, all_preds, best_pred, conf, intent, min_fde) in enumerate(chunk):
            ax = axs[idx // 4, idx % 4]
            lp = hist[-1, :3]; hp = hist[:, :3]; ga = target[:, :3]; ba = lp + best_pred[:, :3]
            all_x = np.concatenate([hp[:, 0], ga[:, 0], ba[:, 0]])
            all_y = np.concatenate([hp[:, 1], ga[:, 1], ba[:, 1]])
            r = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.5)
            xm = (all_x.max() + all_x.min()) / 2; ym = (all_y.max() + all_y.min()) / 2
            h = r * 0.65; ax.set_xlim(xm - h, xm + h); ax.set_ylim(ym - h, ym + h)
            ax.plot(hp[:, 0], hp[:, 1], color=HC, lw=2, alpha=0.7, label='History')
            ax.plot(ga[:, 0], ga[:, 1], color=GC, lw=2.5, label='GT')
            ax.plot(ba[:, 0], ba[:, 1], color=BC, lw=2, ls='--', label='Best')
            for s, e, lbl in LSEG:
                ax.plot(ga[e-1, 0], ga[e-1, 1], 'o', color=GC, ms=8, alpha=0.9)
                ax.plot(ba[e-1, 0], ba[e-1, 1], 's', color=BC, ms=8, alpha=0.9)
            se = seg_errs(ba, ga, LSEG)
            es = ' | '.join([f'{l}:{np.mean(e):.2f}m' for l, e in se])
            z_errs = [np.mean(np.abs(ba[s:e, 2] - ga[s:e, 2])) for s, e, _ in LSEG]
            zs = 'Z: ' + ' | '.join([f'{lbl}:{ze:.2f}m' for ze, (_, _, lbl) in zip(z_errs, LSEG)])
            global_num = page * n_per_page + idx + 1
            ax.set_title(f'#{global_num} {INTENT_6[intent]}  minFDE={min_fde:.2f}m\n{es}  {zs}',
                         fontsize=6.5, family='monospace')
            ax.grid(True, alpha=0.3)
            if idx == 0: ax.legend(fontsize=7, loc='best')
        p = OUT / f'low_24_best_xy_p{page+1}.png'
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Multi-Hypothesis 3D (top 6 samples, 2×3)
# ═══════════════════════════════════════════════════════════════════════
def chart_multihyp_3d(samples):
    top6 = samples[:6]
    K_alive = top6[0][2].shape[0]
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f'LOW Model (UAV-Flow) — Multi-Hypothesis K={K_alive} (3D, Top 6 Samples)\n'
                 'Dark=History(4s)  Green=GT  Colored=Hypotheses (thicker=higher confidence)  '
                 'o=GT marker  s=BestPred marker',
                 fontsize=12, fontweight='bold', y=0.99)
    for idx, (hist, target, all_preds, best_pred, conf, intent, min_fde) in enumerate(top6):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        lp = hist[-1, :3]; hp = hist[:, :3]; ga = target[:, :3]; ba = lp + best_pred[:, :3]
        K = all_preds.shape[0]
        ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=HC, lw=1.5, alpha=0.5, label='History')
        ax.plot(ga[:, 0], ga[:, 1], ga[:, 2], color=GC, lw=2.5, label='GT')
        cp_arr = torch.softmax(conf, dim=1)[0].cpu().numpy()
        for k in range(K):
            ha = lp + all_preds[k, :, :3]
            alpha = 0.3 + 0.7 * cp_arr[k]
            lw = 1.0 + 2.5 * cp_arr[k]
            ax.plot(ha[:, 0], ha[:, 1], ha[:, 2], color=MH_C[k], lw=lw,
                    alpha=max(0.3, alpha), label=f'H{k+1} ({cp_arr[k]:.0%})')
        for s, e, lbl in LSEG:
            ax.scatter(*ga[e-1], c=GC, s=25, marker='o', alpha=0.9, zorder=5)
            ax.scatter(*ba[e-1], c=BC, s=25, marker='s', alpha=0.9, zorder=5)
        se = seg_errs(ba, ga, LSEG)
        es = ' | '.join([f'{l}:{np.mean(e):.2f}m' for l, e in se])
        spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
        ax.set_title(f'#{idx+1} {INTENT_6[intent]}  spd={spd:.1f}m/s  minFDE={min_fde:.2f}m\n{es}',
                     fontsize=7.5, family='monospace')
        # Suppress exaggerated Z-axis visual
        z_all = np.concatenate([hp[:, 2], ga[:, 2], ba[:, 2]])
        xy_range = max(ga[:, 0].max() - ga[:, 0].min(), ga[:, 1].max() - ga[:, 1].min(), 0.5)
        z_mid = (z_all.max() + z_all.min()) / 2
        ax.set_zlim(z_mid - xy_range * 0.5, z_mid + xy_range * 0.5)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w'); ax.yaxis.pane.set_edgecolor('w'); ax.zaxis.pane.set_edgecolor('w')
        ax.tick_params(labelsize=6)
        if idx == 0: ax.legend(fontsize=5.5, loc='upper left', ncol=2)
    p = OUT / 'low_06_multihyp_3d.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Multi-Hypothesis XY Grid (top 4 samples, 2×2)
# ═══════════════════════════════════════════════════════════════════════
def chart_multihyp_xy(samples):
    top4 = samples[:4]
    K_alive = top4[0][2].shape[0]
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f'LOW Model — Multi-Hypothesis K={K_alive} (XY View, Top 4 Samples)\n'
                 'Dark=History  Green=GT  Colored=Hypotheses (thicker=higher confidence)',
                 fontsize=12, fontweight='bold')
    for idx in range(4):
        ax = axs[idx // 2, idx % 2]
        hist, target, all_preds, best_pred, conf, intent, min_fde = top4[idx]
        lp = hist[-1, :3]; hp = hist[:, :3]; ga = target[:, :3]; K = all_preds.shape[0]
        all_x = np.concatenate([hp[:, 0], ga[:, 0], (lp + best_pred[:, :3])[:, 0]])
        all_y = np.concatenate([hp[:, 1], ga[:, 1], (lp + best_pred[:, :3])[:, 1]])
        r = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.5)
        xm = (all_x.max() + all_x.min()) / 2; ym = (all_y.max() + all_y.min()) / 2
        h = r * 0.65; ax.set_xlim(xm - h, xm + h); ax.set_ylim(ym - h, ym + h)
        ax.plot(hp[:, 0], hp[:, 1], color=HC, lw=2, alpha=0.7, label='History')
        ax.plot(ga[:, 0], ga[:, 1], color=GC, lw=2.5, label='GT')
        cp_arr = torch.softmax(conf, dim=1)[0].cpu().numpy()
        for k in range(K):
            ha = lp + all_preds[k, :, :3]
            alpha = 0.3 + 0.7 * cp_arr[k]
            lw = 1.0 + 2.5 * cp_arr[k]
            ax.plot(ha[:, 0], ha[:, 1], color=MH_C[k], lw=lw,
                    alpha=max(0.3, alpha), label=f'H{k+1} ({cp_arr[k]:.0%})')
        ba = lp + best_pred[:, :3]
        for s, e, lbl in LSEG:
            ax.plot(ga[e-1, 0], ga[e-1, 1], 'o', color=GC, ms=8, alpha=0.9)
            ax.plot(ba[e-1, 0], ba[e-1, 1], 's', color=BC, ms=8, alpha=0.9)
        ax.set_title(f'#{idx+1} {INTENT_6[intent]}  minFDE={min_fde:.2f}m', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7, loc='best', ncol=2)
    p = OUT / 'low_04_multihyp_xy.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Per-Second Error Bars (24 samples)
# ═══════════════════════════════════════════════════════════════════════
def chart_persecond_error(samples):
    fig, ax = plt.subplots(figsize=(24, 8))
    fig.suptitle('LOW Model — Per-Second Independent Prediction Error (24 Best Samples)',
                 fontsize=12, fontweight='bold')
    sl = [s[2] for s in LSEG]; x = np.arange(len(sl))
    w = 0.7 / len(samples)
    for idx, (hist, target, all_preds, best_pred, conf, intent, min_fde) in enumerate(samples):
        lp = hist[-1, :3]; ba = lp + best_pred[:, :3]; ga = target[:, :3]
        se = seg_errs(ba, ga, LSEG); means = [np.mean(e) for _, e in se]
        off = (idx - len(samples) / 2 + 0.5) * w
        ax.bar(x + off, means, w, alpha=0.85, label=f'#{idx+1} {INTENT_6[intent]}')
        for xi, m in zip(x, means):
            ax.text(xi + off, m + 0.03, f'{m:.2f}', ha='center', fontsize=4, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(sl, fontsize=10)
    ax.set_ylabel('Mean L2 Error (m)', fontsize=10)
    ax.set_xlabel('Prediction Time Segment', fontsize=10)
    ax.legend(fontsize=5, ncol=6, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    p = OUT / 'low_24_per_second_error.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Error Data Table (24 samples, per-second errors in numbers)
# ═══════════════════════════════════════════════════════════════════════
def chart_error_table(samples):
    """Render a clean table of per-second errors for all 24 samples."""
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.axis('off')
    fig.suptitle('LOW Model — Per-Second Error Data (Best Confidence Trajectory vs Ground Truth)',
                 fontsize=13, fontweight='bold', y=0.98)

    # Build table data
    headers = ['#', 'Intent', 'Speed', 'minFDE',
               '0-1s L2', '0-1s Z', '1-2s L2', '1-2s Z',
               '2-3s L2', '2-3s Z', '3-4s L2', '3-4s Z', 'Confidence']
    rows = []
    for idx, (hist, target, all_preds, best_pred, conf, intent, min_fde) in enumerate(samples):
        lp = hist[-1, :3]; ba = lp + best_pred[:, :3]; ga = target[:, :3]
        se = seg_errs(ba, ga, LSEG)
        z_errs = [np.mean(np.abs(ba[s:e, 2] - ga[s:e, 2])) for s, e, _ in LSEG]
        spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
        cp_arr = torch.softmax(conf, dim=1)[0].cpu().numpy()
        conf_str = '/'.join([f'{v:.2f}' for v in cp_arr])
        row = [f'{idx+1}', INTENT_6[intent], f'{spd:.1f}', f'{min_fde:.2f}',
               f'{np.mean(se[0][1]):.2f}', f'{z_errs[0]:.2f}',
               f'{np.mean(se[1][1]):.2f}', f'{z_errs[1]:.2f}',
               f'{np.mean(se[2][1]):.2f}', f'{z_errs[2]:.2f}',
               f'{np.mean(se[3][1]):.2f}', f'{z_errs[3]:.2f}',
               conf_str]
        rows.append(row)

    # Color rows by intent
    intent_colors = ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#FCE4EC', '#F3E5F5', '#ECEFF1']
    cell_colors = []
    for row in rows:
        intent_name = row[1]
        ci = INTENT_6.index(intent_name) if intent_name in INTENT_6 else 5
        cell_colors.append([intent_colors[ci]] * len(headers))

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                     cellColours=cell_colors,
                     colColours=['#37474F'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=7.5)

    # Subtitle with summary stats
    all_fde = [s[6] for s in samples]
    ax.text(0.5, 0.01,
            f'Summary: minFDE range [{min(all_fde):.2f}, {max(all_fde):.2f}]m  '
            f'mean={np.mean(all_fde):.2f}m  median={np.median(all_fde):.2f}m  '
            f'Active heads: {samples[0][2].shape[0]}',
            ha='center', fontsize=9, fontstyle='italic', transform=fig.transFigure)

    p = OUT / 'low_24_error_table.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 60)
    print('  LOW Multi-Hypothesis — 24 Best Samples Visualization')
    print('=' * 60)

    samples = collect(n_tgt=200)
    K_alive = samples[0][2].shape[0]
    print(f'\nActive heads: {K_alive} (dead heads filtered)')
    print(f'Selected: {len(samples)} samples')

    print(f'\n--- Generating Best Trajectory 3D (3 pages) ---')
    chart_best_3d(samples)

    print(f'\n--- Generating Best Trajectory XY (3 pages) ---')
    chart_best_xy(samples)

    print(f'\n--- Generating Multi-Hypothesis 3D (top 6) ---')
    chart_multihyp_3d(samples)

    print(f'\n--- Generating Multi-Hypothesis XY Grid (top 4) ---')
    chart_multihyp_xy(samples)

    print(f'\n--- Generating Per-Second Error Bars ---')
    chart_persecond_error(samples)

    print(f'\n--- Generating Error Data Table ---')
    chart_error_table(samples)

    print(f'\n{"=" * 60}')
    print(f'  Done! {len(samples)} samples, K={K_alive}, 10 charts')
    print(f'{"=" * 60}')
