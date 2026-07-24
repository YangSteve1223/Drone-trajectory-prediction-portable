#!/usr/bin/env python3
"""
LOW 40-FRAME Multi-Hypothesis Trajectory Visualization on LONG trajectories.

Reuses the EXACT data pipeline + model loading from train_multihead_low.py:
  - HIST_LEN=40, PRED_LEN=20, SCALE_POS=100
  - TM.make_adaptive_windows / TM.collect_windows-style logic on raw ../UAV-Flow-trajs
  - TM.build_40frame_model(K=5) then loads trained low_multihead_K5_40frame.pth
  - TM.mh_forward for inference

Predictions from forward_multi_head are ALREADY in meters (displacement from last
history pos). Absolute = last_hist_pos + pred_displacement. Targets are displacement (m).

Outputs (pic-results/, low40_ prefix):
  low40_multihyp_3d.png, low40_multihyp_xy.png,
  low40_per_second_error.png, low40_error_table.png
"""
import torch, numpy as np, sys, argparse, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))  # reuse train pipeline
import train_multihead_low as TM

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path(__file__).resolve().parents[1] / 'pic-results'; OUT.mkdir(parents=True, exist_ok=True)
WEIGHT_DIR = Path(__file__).resolve().parents[1] / 'weights'
TRAJ_DIR = Path(__file__).resolve().parents[2] / 'UAV-Flow-trajs'

INTENT_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
MH_C = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
HC, GC, BC = '#37474F', '#00C853', '#00E676'
LSEG = [(0, 5, '0-1s'), (5, 10, '1-2s'), (10, 15, '2-3s'), (15, 20, '3-4s')]
CONF_THRESHOLD = 0.05  # dead-head filter (softmax over confidences)
LONG_MIN = 150         # min raw-trajectory length


# ═══════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════
def load_model(K=5):
    model = TM.build_40frame_model(K)          # loads base + replaces w/ multi-head
    mc = torch.load(WEIGHT_DIR / f'low_multihead_K{K}_40frame.pth',
                    map_location=DEVICE, weights_only=False)
    model.ua_pgd.neural_decoder.load_state_dict(mc['multi_decoder_state'])
    model.ua_pgd.neural_decoder = model.ua_pgd.neural_decoder.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, hb):
    """hb: (B,40,6) in meters. Returns per-sample lists after dead-head filtering.
    all_preds/best_pred are DISPLACEMENT in meters; intent from dtp intent_weights."""
    hb = hb.to(DEVICE)
    h_norm = model._normalize(hb)
    enc = model.emam_se(h_norm)
    dtp = model.ia_dtp(enc, historical_trajectory=h_norm)
    mh = model.ua_pgd.forward_multi_head(
        encoded_feat=enc, global_anchor=dtp['global_anchor'],
        historical_trajectory=h_norm, intent_weights=dtp['intent_weights'])
    all_preds = mh['all_predictions']            # (K,B,P,3) meters
    conf = mh['confidences']                     # (B,K)
    intent = dtp['intent_weights'].argmax(dim=1) # (B,)
    return all_preds.cpu(), conf.cpu(), intent.cpu()


def filter_dead_heads(all_preds_i, conf_i):
    """all_preds_i: (K,P,3)  conf_i: (K,) -> alive preds (Ka,P,3), conf (Ka,), softmax (Ka,)."""
    cp = torch.softmax(conf_i, dim=0)
    alive = cp > CONF_THRESHOLD
    if alive.sum() < 1:
        alive[cp.argmax()] = True
    ap = all_preds_i[alive]
    cf = conf_i[alive]
    cp_out = torch.softmax(cf, dim=0)
    best_k = int(cf.argmax())
    return ap.numpy(), best_k, cf, cp_out.numpy()


# ═══════════════════════════════════════════════════════════════════════
#  Scoring helpers
# ═══════════════════════════════════════════════════════════════════════
def seg_errs(pa, ga, segs):
    return [(lbl, np.linalg.norm(pa[s:e] - ga[s:e], axis=1)) for s, e, lbl in segs]


def diversity_score(hist, target, all_preds, best_pred, min_fde, intent):
    """Score for picking diverse/representative windows: reward hypothesis spread,
    trajectory extent, and low-ish minFDE."""
    lp = hist[-1, :3]; ga = target[:, :3]
    K = all_preds.shape[0]
    ext = float(np.linalg.norm(ga[-1] - ga[0]))
    # spread of final hypothesis endpoints (multi-modality)
    ends = np.array([lp + all_preds[k, -1] for k in range(K)])
    spread = float(np.mean(np.linalg.norm(ends - ends.mean(0), axis=1))) if K > 1 else 0.0
    spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
    s = 0.0
    s += max(0, 5.0 - min_fde) * 4.0
    s += min(ext, 6.0) * 2.0
    s += min(spread, 3.0) * 4.0
    s += min(spd, 2.5) * 2.0
    if intent in (3, 4): s += 4.0   # up/down are visually interesting
    return s


# ═══════════════════════════════════════════════════════════════════════
#  Collect: long trajectories -> 40-frame windows -> diverse selection
# ═══════════════════════════════════════════════════════════════════════
def collect(model, n_select=24, n_trajs=400, K=5, batch_size=64):
    print(f'Collecting long trajectories (>={LONG_MIN} frames) from {TRAJ_DIR}...')
    files = sorted(TRAJ_DIR.glob('*.npz'))
    np.random.seed(42); np.random.shuffle(files)
    long_windows = []   # (hist(np), fut(np))
    scanned = 0
    for f in files:
        if len(long_windows) >= n_trajs:
            break
        d = np.load(f); traj = d['traj']
        if traj.shape[0] < LONG_MIN:
            continue
        scanned += 1
        hs, fs = TM.make_adaptive_windows(traj, TM.HIST_LEN)
        # one representative window per trajectory (the middle one) to keep diversity
        if not hs:
            continue
        mid = len(hs) // 2
        long_windows.append((hs[mid].astype(np.float32), fs[mid].astype(np.float32)))
    print(f'  Long trajs used: {len(long_windows)} (scanned {scanned})')

    # Batched inference
    scored = []
    single_fdes = []; min_fdes = []
    for b in range(0, len(long_windows), batch_size):
        chunk = long_windows[b:b + batch_size]
        hb = torch.from_numpy(np.stack([c[0] for c in chunk]))
        tb = np.stack([c[1] for c in chunk])            # (B,P,3) displacement m
        all_preds, conf, intent = predict_batch(model, hb)
        for i in range(len(chunk)):
            hist = chunk[i][0]; tgt = chunk[i][1]
            ap, best_k, cf, cp = filter_dead_heads(all_preds[:, i], conf[i])
            Ka = ap.shape[0]
            ga = tgt[:, :3]
            # minFDE over alive heads (displacement space == absolute-diff space)
            fde_k = [float(np.linalg.norm(ap[k, -1] - ga[-1])) for k in range(Ka)]
            mf = min(fde_k)
            best_pred = ap[best_k]
            single_fde = float(np.linalg.norm(best_pred[-1] - ga[-1]))
            it = int(intent[i])
            single_fdes.append(single_fde); min_fdes.append(mf)
            sc = diversity_score(hist, tgt, ap, best_pred, mf, it)
            scored.append((sc, mf, single_fde, hist, tgt, ap, best_pred, cf, cp, it))

    single_fdes = np.array(single_fdes); min_fdes = np.array(min_fdes)
    print(f'  Windows evaluated: {len(scored)}')
    print(f'  >>> mean single-FDE = {single_fdes.mean():.4f} m   '
          f'mean minFDE_{K} = {min_fdes.mean():.4f} m  (sanity: ~0.87 / ~0.60)')

    # Diverse selection: cap per-intent, sort by diversity score
    scored.sort(key=lambda x: x[0], reverse=True)
    sel = []; ic = defaultdict(int)
    cap = max(1, n_select // 3)
    for e in scored:
        it = e[9]
        if ic[it] < cap:
            sel.append(e); ic[it] += 1
        if len(sel) >= n_select:
            break
    for e in scored:                      # fill remainder if under-filled
        if len(sel) >= n_select: break
        if e not in sel: sel.append(e)
    print(f'  Selected {len(sel)} diverse windows')
    return sel, single_fdes, min_fdes


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Multi-Hypothesis 3D grid
# ═══════════════════════════════════════════════════════════════════════
def chart_multihyp_3d(samples, n_show=12):
    show = samples[:n_show]
    cols = 4; rows = (len(show) + cols - 1) // cols
    fig = plt.figure(figsize=(7 * cols, 5.5 * rows))
    fig.suptitle('LOW 40-FRAME Model (UAV-Flow long trajs) — Multi-Hypothesis 3D\n'
                 'Dark=History(40f/8s)  Green=GT  Colored=Hypotheses(thicker=higher conf)  '
                 'o=GT marker  s=Best marker',
                 fontsize=13, fontweight='bold', y=0.995)
    for idx, (sc, mf, sf, hist, tgt, ap, bp, cf, cp, it) in enumerate(show):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        lp = hist[-1, :3]; hp = hist[:, :3]; ga = tgt[:, :3]; ba = lp + bp
        K = ap.shape[0]
        ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color=HC, lw=1.5, alpha=0.5, label='History')
        ax.plot(ga[:, 0], ga[:, 1], ga[:, 2], color=GC, lw=2.5, label='GT')
        for k in range(K):
            ha = lp + ap[k]
            ax.plot(ha[:, 0], ha[:, 1], ha[:, 2], color=MH_C[k % len(MH_C)],
                    lw=1.0 + 2.5 * cp[k], alpha=max(0.3, 0.3 + 0.7 * cp[k]),
                    label=f'H{k+1} ({cp[k]:.0%})')
        for s, e, lbl in LSEG:
            ax.scatter(*ga[e-1], c=GC, s=22, marker='o', zorder=5)
            ax.scatter(*ba[e-1], c=BC, s=22, marker='s', zorder=5)
        se = seg_errs(ba, ga, LSEG)
        es = ' | '.join([f'{l}:{np.mean(e):.2f}' for l, e in se])
        spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
        ax.set_title(f'#{idx+1} {INTENT_6[it]} spd={spd:.1f} minFDE={mf:.2f}m\n{es}',
                     fontsize=6.5, family='monospace')
        z_all = np.concatenate([hp[:, 2], ga[:, 2], ba[:, 2]])
        xy_range = max(ga[:, 0].max() - ga[:, 0].min(), ga[:, 1].max() - ga[:, 1].min(), 0.5)
        z_mid = (z_all.max() + z_all.min()) / 2
        ax.set_zlim(z_mid - xy_range * 0.5, z_mid + xy_range * 0.5)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False; pane.set_edgecolor('w')
        ax.tick_params(labelsize=5)
        if idx == 0: ax.legend(fontsize=5.5, loc='upper left', ncol=2)
    p = OUT / 'low40_multihyp_3d.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Multi-Hypothesis XY grid
# ═══════════════════════════════════════════════════════════════════════
def chart_multihyp_xy(samples, n_show=12):
    show = samples[:n_show]
    cols = 4; rows = (len(show) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows), squeeze=False)
    fig.suptitle('LOW 40-FRAME Model — Multi-Hypothesis XY (top-down)\n'
                 'Dark=History  Green=GT  Colored=Hypotheses(thicker=higher conf)  '
                 'o=GT  s=Best',
                 fontsize=13, fontweight='bold')
    for idx in range(rows * cols):
        ax = axs[idx // cols, idx % cols]
        if idx >= len(show):
            ax.axis('off'); continue
        sc, mf, sf, hist, tgt, ap, bp, cf, cp, it = show[idx]
        lp = hist[-1, :3]; hp = hist[:, :3]; ga = tgt[:, :3]; ba = lp + bp; K = ap.shape[0]
        all_x = np.concatenate([hp[:, 0], ga[:, 0], ba[:, 0]])
        all_y = np.concatenate([hp[:, 1], ga[:, 1], ba[:, 1]])
        r = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.5)
        xm = (all_x.max() + all_x.min()) / 2; ym = (all_y.max() + all_y.min()) / 2
        hh = r * 0.65; ax.set_xlim(xm - hh, xm + hh); ax.set_ylim(ym - hh, ym + hh)
        ax.plot(hp[:, 0], hp[:, 1], color=HC, lw=2, alpha=0.7, label='History')
        ax.plot(ga[:, 0], ga[:, 1], color=GC, lw=2.5, label='GT')
        for k in range(K):
            ha = lp + ap[k]
            ax.plot(ha[:, 0], ha[:, 1], color=MH_C[k % len(MH_C)],
                    lw=1.0 + 2.5 * cp[k], alpha=max(0.3, 0.3 + 0.7 * cp[k]),
                    label=f'H{k+1} ({cp[k]:.0%})')
        for s, e, lbl in LSEG:
            ax.plot(ga[e-1, 0], ga[e-1, 1], 'o', color=GC, ms=7)
            ax.plot(ba[e-1, 0], ba[e-1, 1], 's', color=BC, ms=7)
        ax.set_title(f'#{idx+1} {INTENT_6[it]} minFDE={mf:.2f}m', fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=6, loc='best', ncol=2)
    p = OUT / 'low40_multihyp_xy.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Per-Second Error curve (single-best vs minK), mean over samples
# ═══════════════════════════════════════════════════════════════════════
def chart_persecond_error(samples, K=5):
    steps = 20
    single_err = np.zeros(steps); mink_err = np.zeros(steps)
    for sc, mf, sf, hist, tgt, ap, bp, cf, cp, it in samples:
        ga = tgt[:, :3]
        single_err += np.linalg.norm(bp - ga, axis=1)
        # per-step min over heads (oracle)
        per_step = np.stack([np.linalg.norm(ap[k] - ga, axis=1) for k in range(ap.shape[0])])
        mink_err += per_step.min(axis=0)
    single_err /= len(samples); mink_err /= len(samples)
    t = (np.arange(steps) + 1) * TM.DT   # seconds

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('LOW 40-FRAME — Per-Step Prediction Error (mean over selected long-traj samples)',
                 fontsize=12, fontweight='bold')
    ax.plot(t, single_err, '-o', color=BC, lw=2, ms=4, label=f'Single-best (mean FDE={single_err[-1]:.2f}m)')
    ax.plot(t, mink_err, '-s', color=MH_C[0], lw=2, ms=4, label=f'minK oracle K={K} (mean FDE={mink_err[-1]:.2f}m)')
    for xb in (1, 2, 3, 4):
        ax.axvline(xb, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('Prediction horizon (s)'); ax.set_ylabel('Mean L2 error (m)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    p = OUT / 'low40_per_second_error.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  CHART: Error table (per-sample minADE / minFDE / single)
# ═══════════════════════════════════════════════════════════════════════
def chart_error_table(samples, K=5):
    fig, ax = plt.subplots(figsize=(20, max(6, 0.5 * len(samples) + 3)))
    ax.axis('off')
    fig.suptitle('LOW 40-FRAME — Per-Sample Error Table (long trajectories)',
                 fontsize=13, fontweight='bold', y=0.98)
    headers = ['#', 'Intent', 'Speed', 'single ADE', 'single FDE',
               f'minADE_{K}', f'minFDE_{K}',
               '0-1s', '1-2s', '2-3s', '3-4s', 'Conf(alive)']
    rows = []; single_ades = []; min_ades = []; min_fdes = []; single_fdes = []
    for idx, (sc, mf, sf, hist, tgt, ap, bp, cf, cp, it) in enumerate(samples):
        ga = tgt[:, :3]
        s_ade = float(np.linalg.norm(bp - ga, axis=1).mean())
        per_step = np.stack([np.linalg.norm(ap[k] - ga, axis=1) for k in range(ap.shape[0])])
        m_ade = float(per_step.mean(axis=1).min())
        se = seg_errs(bp, ga, LSEG)
        spd = float(np.linalg.norm(hist[-5:, 3:6], axis=1).mean())
        conf_str = '/'.join(f'{v:.2f}' for v in cp)
        rows.append([f'{idx+1}', INTENT_6[it], f'{spd:.1f}', f'{s_ade:.2f}', f'{sf:.2f}',
                     f'{m_ade:.2f}', f'{mf:.2f}',
                     f'{np.mean(se[0][1]):.2f}', f'{np.mean(se[1][1]):.2f}',
                     f'{np.mean(se[2][1]):.2f}', f'{np.mean(se[3][1]):.2f}', conf_str])
        single_ades.append(s_ade); min_ades.append(m_ade); min_fdes.append(mf); single_fdes.append(sf)

    intent_colors = ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#FCE4EC', '#F3E5F5', '#ECEFF1']
    cell_colors = [[intent_colors[INTENT_6.index(r[1])]] * len(headers) for r in rows]
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                     cellColours=cell_colors, colColours=['#37474F'] * len(headers))
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
    for j in range(len(headers)):
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=8)
    ax.text(0.5, 0.02,
            f'Summary: single FDE mean={np.mean(single_fdes):.2f}m  minFDE_{K} mean={np.mean(min_fdes):.2f}m  '
            f'single ADE mean={np.mean(single_ades):.2f}m  minADE_{K} mean={np.mean(min_ades):.2f}m  '
            f'(n={len(samples)}, alive heads={samples[0][5].shape[0]})',
            ha='center', fontsize=10, fontstyle='italic', transform=fig.transFigure)
    p = OUT / 'low40_error_table.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--n_select', type=int, default=18)
    ap.add_argument('--n_trajs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    print('=' * 70)
    print('  LOW 40-FRAME Multi-Hypothesis Visualization (long trajectories)')
    print('=' * 70)
    model = load_model(args.K)
    samples, single_fdes, min_fdes = collect(
        model, n_select=args.n_select, n_trajs=args.n_trajs,
        K=args.K, batch_size=min(64, args.batch_size))
    Ka = samples[0][5].shape[0]
    print(f'\nActive heads after filtering: {Ka} (of K={args.K})')

    print('\n--- Multi-Hypothesis 3D ---'); chart_multihyp_3d(samples, n_show=min(12, len(samples)))
    print('--- Multi-Hypothesis XY ---'); chart_multihyp_xy(samples, n_show=min(12, len(samples)))
    print('--- Per-Second Error ---'); chart_persecond_error(samples, K=args.K)
    print('--- Error Table ---'); chart_error_table(samples, K=args.K)

    print('\n' + '=' * 70)
    print(f'  SANITY: mean single-FDE = {single_fdes.mean():.4f} m  '
          f'mean minFDE_{args.K} = {min_fdes.mean():.4f} m')
    print(f'  (expected ~0.87 single / ~0.60 minFDE_5 on long trajs)')
    print('=' * 70)




