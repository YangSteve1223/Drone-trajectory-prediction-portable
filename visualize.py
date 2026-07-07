#!/usr/bin/env python3
"""
3D轨迹预测可视化 — 简洁排版
每个样本一行: 3D视图 | XY俯视图(正方形) | 误差表
低/高速各一张图, 真实数据。
"""

import torch, numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from utils.fast_data_loader import FastWindowDataset

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 9})

UAV_MARKERS = {'1s(25%)':5, '2s(50%)':10, '3s(75%)':15, '4s(100%)':19}
NPZ_MARKERS = {'5s(25%)':5, '10s(50%)':10, '15s(75%)':15, '20s(100%)':19}
MCOLORS = ['#FF9800','#FF5722','#E91E63','#9C27B0']


def draw_3d(ax, r, markers):
    """3D轨迹 + 时间节点标记"""
    last = r['hist'][-1,:3].cpu().numpy()
    hp = r['hist'][:,:3].cpu().numpy()
    pa = r['pred'].cpu().numpy() + last
    ta = r['target'].cpu().numpy() + last

    ax.plot(hp[:,0],hp[:,1],hp[:,2], 'b-', lw=2)
    ax.plot(pa[:,0],pa[:,1],pa[:,2], 'r--', lw=2)
    ax.plot(ta[:,0],ta[:,1],ta[:,2], 'g--', lw=2)
    ax.scatter(hp[-1,0],hp[-1,1],hp[-1,2], c='b', s=100, marker='s', zorder=5)

    for j,(lbl,fi) in enumerate(zip(markers.keys(),markers.values())):
        c = MCOLORS[j]
        ax.scatter(pa[fi,0],pa[fi,1],pa[fi,2], c=c, s=70, marker='D', zorder=10, ec='k', lw=0.5)
        ax.scatter(ta[fi,0],ta[fi,1],ta[fi,2], c=c, s=55, marker='o', zorder=10, ec='k', lw=0.5)
        mid = (pa[fi]+ta[fi])/2
        ax.text(mid[0],mid[1],mid[2], lbl.replace('(','\n('), fontsize=6, ha='center', color=c,
                fontweight='bold', bbox=dict(fc='white',alpha=0.7,pad=1))

    title = f'[{r["label"]}] {r["intent"]}  [{r["route"]}]  {r["speed"]:.1f} m/s'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,0.4])


def draw_xy(ax, r, markers):
    """XY俯视图 — 正方形, 两轴独立scale"""
    last = r['hist'][-1,:2].cpu().numpy()
    hp = r['hist'][:,:2].cpu().numpy()
    pa = r['pred'].cpu().numpy()[:,:2] + last
    ta = r['target'].cpu().numpy()[:,:2] + last

    ax.plot(hp[:,0],hp[:,1], 'b-', lw=2)
    ax.plot(pa[:,0],pa[:,1], 'r--', lw=2)
    ax.plot(ta[:,0],ta[:,1], 'g--', lw=2)
    ax.scatter(hp[-1,0],hp[-1,1], c='b', s=80, marker='s', zorder=5)

    for j,(lbl,fi) in enumerate(zip(markers.keys(),markers.values())):
        c = MCOLORS[j]
        ax.scatter(pa[fi,0],pa[fi,1], c=c, s=60, marker='D', zorder=10, ec='k', lw=0.5)
        ax.scatter(ta[fi,0],ta[fi,1], c=c, s=45, marker='o', zorder=10, ec='k', lw=0.5)
        ax.plot([pa[fi,0],ta[fi,0]],[pa[fi,1],ta[fi,1]], color=c, ls=':', lw=0.8, alpha=0.6)
        ax.annotate(lbl, (pa[fi,0],pa[fi,1]), fontsize=6, ha='left', color=c, fontweight='bold',
                    xytext=(3,3), textcoords='offset points')

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('XY Top-down View', fontsize=10, fontweight='bold')
    # 正方形: 强制axes box为正方形
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)


def draw_error_table(ax_table, r):
    """误差表"""
    ax_table.axis('off')
    markers = r['markers']
    errs = [r['step_err'][fi] for fi in markers.values()]
    z_pr = np.ptp(r['pred'].cpu().numpy()[:,2])
    z_tr = np.ptp(r['target'].cpu().numpy()[:,2])

    col_labels = list(markers.keys()) + ['Z pred', 'Z true']
    vals = [f'{e:.4f}' for e in errs] + [f'{z_pr:.4f}', f'{z_tr:.4f}']
    cell = [vals]

    tbl = ax_table.table(cellText=cell, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.05, 1.8)

    max_e = max(errs) if max(errs)>0 else 1.0
    for j,e in enumerate(errs):
        tbl[(1,j)].set_facecolor((1.0, 1.0-e/max_e*0.4, 1.0-e/max_e*0.4))
    for j in range(len(col_labels)):
        tbl[(0,j)].set_facecolor('#EEEEEE')
        tbl[(0,j)].get_text().set_fontweight('bold')


def build_results(p, ds_low, ds_high):
    """收集12个样本"""
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
            diff = pred - target
            step_err = torch.norm(diff, dim=-1).cpu().numpy()
            markers = UAV_MARKERS if 'UAV' in lbl else NPZ_MARKERS
            dt_lbl = '5Hz' if 'UAV' in lbl else '1Hz'
            intent_str = intent_names[iv] if iv<6 else f'C{iv}'

            results.append({
                'label': f'{lbl}#{idx}', 'intent': intent_str, 'dt': dt_lbl,
                'speed': out['speed'].item(), 'route': str(out['route'][0]),
                'hist': hist, 'pred': pred, 'target': target,
                'step_err': step_err, 'markers': markers,
            })
            cnt += 1
    return results


def make_figure(results, title_prefix, out_path):
    """生成一张图: 每个样本一行 [3D | XY | 误差表]"""
    n = len(results)
    fig = plt.figure(figsize=(22, 5.5 * n))
    fig.suptitle(title_prefix, fontsize=15, fontweight='bold', y=0.995)

    for i, r in enumerate(results):
        # 3D
        ax3d = fig.add_subplot(n, 3, 3*i+1, projection='3d')
        draw_3d(ax3d, r, r['markers'])

        # XY
        ax_xy = fig.add_subplot(n, 3, 3*i+2)
        draw_xy(ax_xy, r, r['markers'])

        # 误差表
        ax_tbl = fig.add_subplot(n, 3, 3*i+3)
        draw_error_table(ax_tbl, r)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def make_error_curve(low_res, high_res, out_path):
    """逐进度误差曲线"""
    fig, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('3D Position Error by Prediction Progress', fontsize=14, fontweight='bold')
    x_t = [25,50,75,100]
    cl = plt.cm.Blues(np.linspace(0.35,0.9,len(low_res)))
    ch = plt.cm.Reds(np.linspace(0.35,0.9,len(high_res)))

    for i,r in enumerate(low_res):
        errs = [r['step_err'][fi] for fi in r['markers'].values()]
        lbl = f'{r["label"].split("#")[1]} {r["intent"]} {r["speed"]:.1f}m/s'
        ax_l.plot(x_t, errs, 'o-', color=cl[i], lw=2.5, ms=10, label=lbl)
        for x,e in zip(x_t,errs):
            ax_l.annotate(f'{e:.3f}',(x,e), textcoords='offset points', xytext=(0,10), fontsize=8, ha='center', color=cl[i])
    ax_l.set_xlabel('Progress (%)'); ax_l.set_ylabel('Error (m)')
    ax_l.set_title(f'Low-speed  avg speed={np.mean([r["speed"] for r in low_res]):.1f}m/s', fontsize=12, fontweight='bold')
    ax_l.legend(fontsize=7, loc='upper left', ncol=2); ax_l.grid(True,alpha=0.3)
    ax_l.set_xticks(x_t); ax_l.set_ylim(bottom=0)

    for i,r in enumerate(high_res):
        errs = [r['step_err'][fi] for fi in r['markers'].values()]
        lbl = f'{r["label"].split("#")[1]} {r["intent"]} {r["speed"]:.0f}m/s'
        ax_h.plot(x_t, errs, 'o-', color=ch[i], lw=2.5, ms=10, label=lbl)
        for x,e in zip(x_t,errs):
            ax_h.annotate(f'{e:.3f}',(x,e), textcoords='offset points', xytext=(0,10), fontsize=8, ha='center', color=ch[i])
    ax_h.set_xlabel('Progress (%)'); ax_h.set_ylabel('Error (m)')
    ax_h.set_title(f'High-speed  avg speed={np.mean([r["speed"] for r in high_res]):.0f}m/s', fontsize=12, fontweight='bold')
    ax_h.legend(fontsize=7, loc='upper left', ncol=2); ax_h.grid(True,alpha=0.3)
    ax_h.set_xticks(x_t); ax_h.set_ylim(bottom=0)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    print('Loading...')
    p = DronePredictor()
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    ds_high = FastWindowDataset('../SimCruise', split='test', label_remap={4:3})

    all_r = build_results(p, ds_low, ds_high)
    low_r = [r for r in all_r if 'UAV' in r['label']]
    high_r = [r for r in all_r if 'NPZ' in r['label']]

    make_figure(low_r,
                'Low-Speed Model (UAV-Flow, 5Hz, DJI drones) — Per-Progress Error\n'
                'Left=3D | Middle=XY Top-down | Right=Error Table (m)',
                out_dir / '01_low_speed.png')

    make_figure(high_r,
                'High-Speed Model (SimCruise, 1Hz) — Per-Progress Error\n'
                'Left=3D | Middle=XY Top-down | Right=Error Table (m)',
                out_dir / '02_high_speed.png')

    make_error_curve(low_r, high_r, out_dir / '03_error_curves.png')

    # 数据表
    print(f'\n{"="*110}')
    for name, res in [('LOW-SPEED', low_r), ('HIGH-SPEED', high_r)]:
        print(f'\n--- {name} ---')
        for r in res:
            errs = [r['step_err'][fi] for fi in r['markers'].values()]
            z_pr = np.ptp(r['pred'].cpu().numpy()[:,2])
            z_tr = np.ptp(r['target'].cpu().numpy()[:,2])
            labels = list(r['markers'].keys())
            print(f'{r["label"]:<20} {r["intent"]:<8} {r["speed"]:5.1f}m/s [{r["route"]}]')
            print(f'  Errors: {labels[0]}={errs[0]:.4f}  {labels[1]}={errs[1]:.4f}  {labels[2]}={errs[2]:.4f}  {labels[3]}={errs[3]:.4f}  |  Z pred={z_pr:.4f}  Z true={z_tr:.4f}')

        avg_errs = [np.mean([r['step_err'][fi] for r in res]) for fi in res[0]['markers'].values()]
        print(f'  AVG:    {labels[0]}={avg_errs[0]:.4f}  {labels[1]}={avg_errs[1]:.4f}  {labels[2]}={avg_errs[2]:.4f}  {labels[3]}={avg_errs[3]:.4f}')

    print(f'\nDone! 3 images saved to {out_dir}/')


if __name__ == '__main__':
    main()
