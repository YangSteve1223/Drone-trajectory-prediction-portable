#!/usr/bin/env python3
"""
Context Adapter on REAL UAV-Flow raw trajectories.
Uses long trajectories (>100 frames) from UAV-Flow-trajs/.
60-frame context -> adapter -> better prediction.
"""

import torch, numpy as np, sys, glob, os
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from context_adapter import ContextAdapterV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 9})
MK = {'1s':5, '2s':10, '3s':15, '4s':19}
MC = ['#FF9800','#FF5722','#E91E63','#9C27B0']


def load_traj(filepath):
    """Load raw trajectory, return (positions, velocities) as numpy arrays."""
    d = np.load(filepath)
    traj = d['traj']  # (T, 6): [pos_x, pos_y, pos_z, vx?, vy?, vz?] or [x,y,z,roll,heading,pitch]
    d.close()
    # Determine columns: if columns 3-5 look like velocities (small, zero-mean), use directly
    # If they look like angles, compute velocities from positions
    pos = traj[:, :3].astype(np.float32)

    # Check if columns 3-5 are velocities or angles
    col3_range = np.ptp(traj[:, 3])
    col4_range = np.ptp(traj[:, 4])
    # Velocities for DJI drones should be < 5 m/s, angles would be radians
    if col3_range < 5 and col4_range < 5:
        vel = traj[:, 3:6].astype(np.float32)  # already velocity
    else:
        # Compute velocity from position differences
        vel = np.zeros_like(pos)
        vel[1:] = (pos[1:] - pos[:-1]) / 0.2  # 5Hz
        vel[0] = vel[1]
    return pos, vel


def extract_windows_from_traj(pos, vel, start, hist_len=20, ctx_len=60, pred_len=20):
    """Extract windows from a long trajectory."""
    # Short window: last 20 frames of context
    hist_start = start + ctx_len - hist_len
    origin = pos[hist_start]
    hp = pos[hist_start:hist_start+hist_len] - origin
    hist = np.concatenate([hp, vel[hist_start:hist_start+hist_len]], axis=1)

    # Long context: ctx_len frames
    ctx_origin = pos[start]
    ctx_pos = pos[start:start+ctx_len] - ctx_origin
    ctx = np.concatenate([ctx_pos, vel[start:start+ctx_len]], axis=1)

    # Target
    target_start = hist_start + hist_len
    target = pos[target_start:target_start+pred_len] - pos[target_start-1]

    return (torch.from_numpy(hist).float(),
            torch.from_numpy(ctx).float(),
            torch.from_numpy(target).float())


def draw_xy(ax, hist, pred, target, title):
    last = hist[-1,:2].cpu().numpy()
    hp = hist[:,:2].cpu().numpy()
    pa = pred.cpu().numpy()[:,:2] + last
    ta = target.cpu().numpy()[:,:2] + last
    ax.plot(ta[:,0],ta[:,1], 'g-', lw=2.5, alpha=0.6, label='Truth')
    ax.plot(hp[:,0],hp[:,1], 'b-', lw=2, label='Hist')
    ax.plot(pa[:,0],pa[:,1], 'r--', lw=2, label='Pred')
    ax.scatter(pa[:,0],pa[:,1], c='red', s=8, zorder=10, alpha=0.5)
    for j,(lbl,fi) in enumerate(zip(MK.keys(),MK.values())):
        ax.scatter(pa[fi,0],pa[fi,1], c=MC[j], s=60, marker='D', zorder=15, ec='k',lw=1)
        ax.scatter(ta[fi,0],ta[fi,1], c=MC[j], s=50, marker='o', zorder=15, ec='k',lw=1)
    ax.scatter(hp[-1,0],hp[-1,1], c='b', s=70, marker='s', zorder=5)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_box_aspect(1); ax.grid(True,alpha=0.3); ax.legend(fontsize=6)


def main():
    print('Loading predictor...')
    p = DronePredictor()
    device = p.device
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find long trajectories (>150 frames for enough context windows)
    traj_dir = Path('../UAV-Flow-trajs')
    all_files = sorted(glob.glob(str(traj_dir / '*.npz')))

    long_trajs = []
    for f in all_files:
        d = np.load(f)
        T = d['traj'].shape[0]
        d.close()
        if T >= 150:
            long_trajs.append(f)

    np.random.seed(42)
    selected = list(np.random.choice(long_trajs, min(6, len(long_trajs)), replace=False))
    print('Selected %d long trajectories (from %d total >=150 frames)' % (len(selected), len(long_trajs)))

    all_test_results = []

    for traj_idx, tfile in enumerate(selected):
        name = os.path.basename(tfile).replace('.npz', '')
        print('\nTraining on: %s' % name)
        pos, vel = load_traj(tfile)
        T = pos.shape[0]
        print('  Frames: %d (%.1f seconds at 5Hz)' % (T, T*0.2))

        # Max usable start: T - ctx_len - pred_len = T - 80
        max_start = T - 80
        train_end = int(max_start * 0.65)  # First 65% for training
        train_data = [extract_windows_from_traj(pos, vel, s, ctx_len=60)
                      for s in range(0, train_end, 2)]
        print('  Training windows: %d (max_start=%d)' % (len(train_data), max_start))

        # Adapter
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
        for ep in range(15):
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

        # Test: 4 held-out windows from last 35% of usable range
        test_starts = np.linspace(train_end + 5, max_start - 1, 4, dtype=int)
        test_res = []

        for ts in test_starts:
            h, ctx, tgt = extract_windows_from_traj(pos, vel, ts, ctx_len=60)

            ua._ctx = None
            with torch.no_grad():
                out_b = p.low(h.unsqueeze(0).to(device), force_predict=True)
            pb = out_b['predictions'][0].cpu()

            with torch.no_grad():
                ua._ctx = adapter(ctx.unsqueeze(0).to(device))
            with torch.no_grad():
                out_a = p.low(h.unsqueeze(0).to(device), force_predict=True)
            pa = out_a['predictions'][0].cpu()

            def jerk(pred):
                v=torch.diff(pred,dim=0); a=torch.diff(v,dim=0)
                j=torch.diff(a,dim=0); return j.abs().mean().item()

            test_res.append({
                'start': ts, 'hist': h, 'target': tgt,
                'pred_b': pb, 'pred_a': pa,
                'jerk_b': jerk(pb), 'jerk_a': jerk(pa),
                'err_b4': torch.norm(pb[-1]-tgt[-1]).item(),
                'err_a4': torch.norm(pa[-1]-tgt[-1]).item(),
            })

        ua.neural_decoder.forward = _orig_nd
        all_test_results.append({'name': name, 'tests': test_res})

    # ---- Figure: XY comparison ----
    fig, axes = plt.subplots(len(selected), 4, figsize=(22, 5*len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Context Adapter on REAL UAV-Flow Trajectories\n'
                 '60-frame context from same flight -> adapter trained per-drone\n'
                 'Blue=History  Red=Predicted  Green=Ground Truth',
                 fontsize=14, fontweight='bold', y=1.005)

    for t_idx, tr in enumerate(all_test_results):
        for i in range(min(4, len(tr['tests']))):
            r = tr['tests'][i]
            ax = axes[t_idx, i]
            draw_xy(ax, r['hist'], r['pred_a'], r['target'],
                    '%s t=%d\n4s: %.3f (was %.3f) jerk: %.4f (%.4f)' % (
                        tr['name'][:20], r['start'],
                        r['err_a4'], r['err_b4'],
                        r['jerk_a'], r['jerk_b']))

    p1 = out_dir / '04_context_real_long.png'
    fig.savefig(p1, dpi=120, bbox_inches='tight')
    print('\nSaved:', p1)
    plt.close()

    # ---- Summary ----
    print()
    print('=' * 90)
    print('Context Adapter on REAL long UAV-Flow trajectories')
    print('%-25s %-12s %-12s %-12s' % ('Trajectory', 'B 4s err', 'A 4s err', 'Delta'))
    print('-' * 65)
    all_b4 = []; all_a4 = []
    for tr in all_test_results:
        b4 = np.mean([t['err_b4'] for t in tr['tests']])
        a4 = np.mean([t['err_a4'] for t in tr['tests']])
        all_b4.append(b4); all_a4.append(a4)
        print('%-25s %-12.4f %-12.4f %+-12.4f' % (tr['name'][:25], b4, a4, b4-a4))
    print('-' * 65)
    avg_b = np.mean(all_b4); avg_a = np.mean(all_a4)
    print('%-25s %-12.4f %-12.4f %+-12.4f (%.1f%%)' % ('AVERAGE', avg_b, avg_a, avg_b-avg_a, (avg_b-avg_a)/avg_b*100))


if __name__ == '__main__':
    main()
