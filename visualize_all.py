#!/usr/bin/env python3
"""
Full visualization suite for group meeting:
01: Low-speed 6 samples (base model, real data)
02: High-speed 6 samples (base model, real data)
03: Context Adapter Before/After on real low-speed data
"""

import torch, numpy as np, sys
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from utils.fast_data_loader import FastWindowDataset
from context_adapter import ContextAdapterV2, gen_traj, extract_windows

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 8})
MK = {'1s(25%)':5, '2s(50%)':10, '3s(75%)':15, '4s(100%)':19}
NPZ_MK = {'5s(25%)':5, '10s(50%)':10, '15s(75%)':15, '20s(100%)':19}
MC = ['#FF9800','#FF5722','#E91E63','#9C27B0']


def draw_3d(ax, hist, pred, target, markers, title):
    last = hist[-1,:3].cpu().numpy()
    hp = hist[:,:3].cpu().numpy()
    pa = pred.cpu().numpy() + last
    ta = target.cpu().numpy() + last
    ax.plot(hp[:,0],hp[:,1],hp[:,2], 'b-', lw=2)
    ax.plot(pa[:,0],pa[:,1],pa[:,2], 'r--', lw=2)
    ax.plot(ta[:,0],ta[:,1],ta[:,2], 'g--', lw=2)
    ax.scatter(hp[-1,0],hp[-1,1],hp[-1,2], c='b', s=100, marker='s', zorder=5)
    for j,(lbl,fi) in enumerate(zip(markers.keys(),markers.values())):
        ax.scatter(pa[fi,0],pa[fi,1],pa[fi,2], c=MC[j], s=70, marker='D', zorder=10, ec='k',lw=0.5)
        ax.scatter(ta[fi,0],ta[fi,1],ta[fi,2], c=MC[j], s=55, marker='o', zorder=10, ec='k',lw=0.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,0.4])


def draw_xy(ax, hist, pred, target, markers, title):
    last = hist[-1,:2].cpu().numpy()
    hp = hist[:,:2].cpu().numpy()
    pa = pred.cpu().numpy()[:,:2] + last
    ta = target.cpu().numpy()[:,:2] + last
    ax.plot(hp[:,0],hp[:,1], 'b-', lw=1.5)
    ax.plot(pa[:,0],pa[:,1], 'r--', lw=1.5)
    ax.plot(ta[:,0],ta[:,1], 'g--', lw=1.5)
    ax.scatter(hp[-1,0],hp[-1,1], c='b', s=60, marker='s', zorder=5)
    for j,(lbl,fi) in enumerate(zip(markers.keys(),markers.values())):
        ax.scatter(pa[fi,0],pa[fi,1], c=MC[j], s=50, marker='D', zorder=10, ec='k',lw=0.5)
        ax.scatter(ta[fi,0],ta[fi,1], c=MC[j], s=40, marker='o', zorder=10, ec='k',lw=0.5)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_box_aspect(1); ax.grid(True, alpha=0.3)


def draw_table(ax, r):
    ax.axis('off')
    errs = [r['step_err'][fi] for fi in r['markers'].values()]
    z_pr = np.ptp(r['pred'].cpu().numpy()[:,2])
    z_tr = np.ptp(r['target'].cpu().numpy()[:,2])
    cols = list(r['markers'].keys()) + ['Z pred', 'Z true']
    vals = [f'{e:.4f}' for e in errs] + [f'{z_pr:.4f}', f'{z_tr:.4f}']
    tbl = ax.table(cellText=[vals], colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.8)
    max_e = max(errs) if max(errs)>0 else 1.0
    for j,e in enumerate(errs):
        tbl[(1,j)].set_facecolor((1.0, 1.0-e/max_e*0.4, 1.0-e/max_e*0.4))
    for j in range(len(cols)):
        tbl[(0,j)].set_facecolor('#EEEEEE'); tbl[(0,j)].get_text().set_fontweight('bold')


def process_real_data(p, ds_low, ds_high):
    """Collect 6 low + 6 high real samples."""
    np.random.seed(42)
    low_idx = np.random.choice(len(ds_low), 6, replace=False)
    high_idx = np.random.choice(len(ds_high), 20, replace=False)
    intent_names = ['STRAIGHT','TURN_L','TURN_R','ASCEND','DESC','HOVER']
    results = []

    for lbl, ds, indices in [('UAV-Flow', ds_low, low_idx), ('SimCruise', ds_high, high_idx)]:
        cnt = 0
        for idx in indices:
            if cnt >= 6: break
            hist, pred_data, intent = ds[idx]
            iv = intent.item() if isinstance(intent, torch.Tensor) else int(intent)
            target = pred_data[:,:3] - hist[-1:,:3]
            with torch.no_grad():
                out = p.predict(hist.unsqueeze(0).to(p.device))
            pred = out['predictions'][0].cpu()
            err = torch.norm(pred-target, dim=-1).cpu().numpy()
            mk = MK if 'UAV' in lbl else NPZ_MK
            dt_lbl = '5Hz' if 'UAV' in lbl else '1Hz'
            results.append({
                'label': f'{lbl}#{idx}', 'intent': intent_names[iv] if iv<6 else '?',
                'speed': out['speed'].item(), 'route': str(out['route'][0]), 'dt': dt_lbl,
                'hist': hist, 'pred': pred, 'target': target,
                'step_err': err, 'markers': mk,
            })
            cnt += 1
    return results


def make_base_figure(results, title, out_path):
    """One sample per row: [3D | XY | Table]"""
    n = len(results)
    fig = plt.figure(figsize=(22, 5.2*n))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    for i, r in enumerate(results):
        ax3 = fig.add_subplot(n, 3, 3*i+1, projection='3d')
        draw_3d(ax3, r['hist'], r['pred'], r['target'], r['markers'],
                '[%s] %s [%s] %.1fm/s' % (r['label'], r['intent'], r['route'], r['speed']))
        ax2 = fig.add_subplot(n, 3, 3*i+2)
        draw_xy(ax2, r['hist'], r['pred'], r['target'], r['markers'],
                'XY: %s %s %.1fm/s' % (r['label'], r['intent'], r['speed']))
        ax_t = fig.add_subplot(n, 3, 3*i+3)
        draw_table(ax_t, r)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print('Saved:', out_path)
    plt.close()


def make_adapter_comparison(low_results, out_path):
    """Train context adapter on synthetic data matching each sample's pattern,
    then compare Before vs After on the real samples."""
    p = DronePredictor()
    device = p.device

    # Train one adapter per sample (matching speed/turn to the sample)
    adapted = []
    for s in low_results:
        speed = max(0.5, s['speed'])  # estimate speed
        # Infer turn rate from trajectory curvature
        hist_pos = s['hist'][:,:3].cpu().numpy()
        headings = []
        for i in range(1, len(hist_pos)):
            d = hist_pos[i,:2] - hist_pos[i-1,:2]
            if np.linalg.norm(d) > 0.01:
                headings.append(np.arctan2(d[1], d[0]))
        if len(headings) > 1:
            turn_rate = abs(np.diff(np.unwrap(headings)).mean()) / 0.2  # rad/s -> deg/s
            turn_rate = np.clip(np.degrees(turn_rate), 3, 40)
        else:
            turn_rate = 10

        print('  Adapter for %s: speed=%.1f turn=%.0f' % (s['label'], speed, turn_rate))
        pos, vel = gen_traj(speed, turn_rate, n=800)

        train_data = [extract_windows(pos, vel, st, ctx_len=60)
                      for st in range(0, 550, 2)]

        adapter = ContextAdapterV2(d_model=p.low.d_model, hidden=128).to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-5)

        ua = p.low.ua_pgd
        _orig_nd = ua.neural_decoder.forward

        def make_hook(ua_obj, _orig):
            def hooked(encoded, step_encoding):
                if hasattr(ua_obj, '_ctx') and ua_obj._ctx is not None:
                    encoded = encoded + 0.15 * ua_obj._ctx
                return _orig(encoded, step_encoding)
            return hooked

        ua.neural_decoder.forward = make_hook(ua, _orig_nd)

        bs = 64
        for ep in range(10):
            perm = np.random.permutation(len(train_data))
            for b in range(0, len(train_data), bs):
                idx = perm[b:b+bs]
                hb = torch.stack([train_data[i][0] for i in idx]).to(device)
                cb = torch.stack([train_data[i][1] for i in idx]).to(device)
                tb = torch.stack([train_data[i][2] for i in idx]).to(device)
                opt.zero_grad()
                ua._ctx = adapter(cb)
                out = p.low(hb, force_predict=True)
                loss = F.mse_loss(out['predictions'], tb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
                opt.step()

        # Test: Before vs After on the real sample
        h = s['hist'].unsqueeze(0).to(device)
        ua._ctx = None
        with torch.no_grad():
            out_b = p.low(h, force_predict=True)
        pred_b = out_b['predictions'][0].cpu()

        # Generate synthetic 60-frame context from the sample's hist
        hist_np = s['hist'].cpu().numpy()
        # Repeat/pad to 60 frames
        ctx = np.tile(hist_np, (3, 1))[:60]  # naive repeat
        ctx_t = torch.from_numpy(ctx).float().unsqueeze(0).to(device)
        with torch.no_grad():
            ua._ctx = adapter(ctx_t)
        with torch.no_grad():
            out_a = p.low(h, force_predict=True)
        pred_a = out_a['predictions'][0].cpu()

        ua.neural_decoder.forward = _orig_nd

        err_b = torch.norm(pred_b - s['target'], dim=-1).cpu().numpy()
        err_a = torch.norm(pred_a - s['target'], dim=-1).cpu().numpy()

        adapted.append({
            **s, 'pred_before': pred_b, 'pred_after': pred_a,
            'err_before': err_b, 'err_after': err_a,
        })

    # Figure: 6 rows, [3D Before | 3D After | XY Before | XY After | Table]
    fig = plt.figure(figsize=(28, 26))
    fig.suptitle('Context Adapter on Real Low-Speed Samples\n'
                 'Before (base model) vs After (with 60-frame context adapter)',
                 fontsize=14, fontweight='bold', y=0.998)

    for i, r in enumerate(adapted):
        be = [r['err_before'][fi] for fi in MK.values()]
        ae = [r['err_after'][fi] for fi in MK.values()]

        ax1 = fig.add_subplot(6, 4, 4*i+1, projection='3d')
        draw_3d(ax1, r['hist'], r['pred_before'], r['target'], MK,
                'BEFORE %s %s\n1s=%.3f 2s=%.3f 3s=%.3f 4s=%.3f' % (
                    r['label'], r['intent'], be[0],be[1],be[2],be[3]))

        ax2 = fig.add_subplot(6, 4, 4*i+2, projection='3d')
        draw_3d(ax2, r['hist'], r['pred_after'], r['target'], MK,
                'AFTER %s %s\n1s=%.3f 2s=%.3f 3s=%.3f 4s=%.3f' % (
                    r['label'], r['intent'], ae[0],ae[1],ae[2],ae[3]))

        ax3 = fig.add_subplot(6, 4, 4*i+3)
        draw_xy(ax3, r['hist'], r['pred_before'], r['target'], MK,
                'XY Before | 1s=%.3f 2s=%.3f 3s=%.3f 4s=%.3f' % (be[0],be[1],be[2],be[3]))

        ax4 = fig.add_subplot(6, 4, 4*i+4)
        draw_xy(ax4, r['hist'], r['pred_after'], r['target'], MK,
                'XY After | 1s=%.3f 2s=%.3f 3s=%.3f 4s=%.3f' % (ae[0],ae[1],ae[2],ae[3]))

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print('Saved:', out_path)
    plt.close()

    # Summary
    print()
    for r in adapted:
        be4 = r['err_before'][19]; ae4 = r['err_after'][19]
        print('%s: 4s %.4f -> %.4f (%+.4f)' % (r['label'], be4, ae4, be4-ae4))
    avg_b = np.mean([r['err_before'][19] for r in adapted])
    avg_a = np.mean([r['err_after'][19] for r in adapted])
    print('AVG: %.4f -> %.4f (%+.4f, %.1f%%)' % (avg_b, avg_a, avg_b-avg_a, (avg_b-avg_a)/avg_b*100))


def main():
    print('Loading...')
    p = DronePredictor()
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    ds_high = FastWindowDataset('../SimCruise', split='test', label_remap={4:3})

    all_r = process_real_data(p, ds_low, ds_high)
    low_r = [r for r in all_r if 'UAV' in r['label']]
    high_r = [r for r in all_r if 'NPZ' in r['label']]

    # Fig 1: Low-speed base model
    make_base_figure(low_r,
                     'Low-Speed Model (UAV-Flow, 5Hz, DJI) — 6 Real Samples\n'
                     'Blue=History | Red=Predicted | Green=Ground Truth',
                     out_dir / '01_low_speed.png')

    # Fig 2: High-speed base model
    make_base_figure(high_r,
                     'High-Speed Model (SimCruise, 1Hz) — 6 Real Samples\n'
                     'Blue=History | Red=Predicted | Green=Ground Truth',
                     out_dir / '02_high_speed.png')

    # Fig 3: Context Adapter Before/After on low-speed
    make_adapter_comparison(low_r, out_dir / '03_context_adapter_real.png')

    print('\nDone! All charts saved.')


if __name__ == '__main__':
    main()
