#!/usr/bin/env python3
"""
LoRA with massively expanded training data:
1000-frame trajectories -> ~480 windows each, batch=64, rank=32
Tests whether MORE DATA fixes the trajectory quality issue.
"""

import torch, numpy as np, sys
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from lora import LoRAAdapter

plt.rcParams.update({'font.size': 9})
MK = {'1s':5, '2s':10, '3s':15, '4s':19}
MC = ['#FF9800','#FF5722','#E91E63','#9C27B0']
HEAD = ['ua_pgd.neural_decoder.delta_head', 'ua_pgd.anchor_to_pos.2']


def gen_traj(speed, turn_rate, n=1000, dt=0.2):
    pos=np.zeros((n,3)); vel=np.zeros((n,3)); h=0.0
    for i in range(1,n):
        h+=np.deg2rad(turn_rate)*dt
        vx=speed*np.cos(h); vy=speed*np.sin(h)
        pos[i,0]=pos[i-1,0]+vx*dt; pos[i,1]=pos[i-1,1]+vy*dt
        vel[i,0]=vx; vel[i,1]=vy
    return pos, vel

def win(pos, vel, s, hl=20, pl=20):
    o=pos[s]; hp=pos[s:s+hl]-o
    h=np.concatenate([hp,vel[s:s+hl]], axis=1)
    t=pos[s+hl:s+hl+pl]-pos[s+hl-1]
    return torch.from_numpy(h).float(), torch.from_numpy(t).float()

def draw_xy_detail(ax, hist, pred, target, title):
    last = hist[-1,:2].cpu().numpy()
    hp = hist[:,:2].cpu().numpy()
    pa = pred.cpu().numpy()[:,:2] + last
    ta = target.cpu().numpy()[:,:2] + last
    ax.plot(ta[:,0],ta[:,1], 'g-', lw=2.5, alpha=0.6, label='Truth')
    ax.plot(hp[:,0],hp[:,1], 'b-', lw=2, label='History')
    ax.plot(pa[:,0],pa[:,1], 'r--', lw=2, label='Pred')
    ax.scatter(pa[:,0],pa[:,1], c='red', s=10, zorder=10, alpha=0.5)
    for j,(lbl,fi) in enumerate(zip(MK.keys(),MK.values())):
        ax.scatter(pa[fi,0],pa[fi,1], c=MC[j], s=70, marker='D', zorder=15, ec='k',lw=1)
        ax.scatter(ta[fi,0],ta[fi,1], c=MC[j], s=55, marker='o', zorder=15, ec='k',lw=1)
    ax.scatter(hp[-1,0],hp[-1,1], c='b', s=80, marker='s', zorder=5)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_box_aspect(1); ax.grid(True,alpha=0.3); ax.legend(fontsize=7)


def main():
    print('Loading...')
    p = DronePredictor()
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = p.device

    # 3 long trajectories (1000 frames = 200 seconds each)
    configs = [
        ('SlowTurn', 1.0, 15),
        ('FastStr', 3.0, 3),
        ('SharpTurn', 1.5, 30),
    ]

    all_results = {}
    for tname, speed, turn in configs:
        print('Generating %s (1000 frames)...' % tname)
        pos, vel = gen_traj(speed, turn, n=1000)

        # Training: windows [0:800] stride=1 -> ~780 windows
        train_wins = []
        for s in range(0, 780, 1):
            h, t = win(pos, vel, s)
            train_wins.append((h, t))

        # Test: windows from [820:960] stride=20 (unseen region)
        test_starts = list(range(820, 960, 20))

        for rank in [8, 32]:
            key = '%s_r%d' % (tname, rank)
            print('  Training r=%d on %d windows...' % (rank, len(train_wins)))

            adapter = LoRAAdapter(p.low, r=rank, alpha=rank*2.0,
                                  lora_targets=[], head_targets=HEAD)
            adapter.activate()
            opt = torch.optim.AdamW(adapter.get_trainable_params(), lr=5e-4, weight_decay=1e-5)
            bs = 64

            for ep in range(5):
                perm = np.random.permutation(len(train_wins))
                for b in range(0, len(train_wins), bs):
                    idx = perm[b:b+bs]
                    hb = torch.stack([train_wins[i][0] for i in idx]).to(device)
                    tb = torch.stack([train_wins[i][1] for i in idx]).to(device)
                    opt.zero_grad()
                    out = p.low(hb, force_predict=True)
                    l = F.mse_loss(out['predictions'], tb)
                    # Add mild jerk penalty
                    v = torch.diff(out['predictions'], dim=1)
                    a = torch.diff(v, dim=1)
                    j = torch.diff(a, dim=1)
                    l = l + 0.01 * (j**2).mean()
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.get_trainable_params(), 1.0)
                    opt.step()

            lora_s = adapter.get_lora_state()
            head_s = adapter.get_head_state()

            # Evaluate on test windows
            test_res = []
            for ts in test_starts:
                h, tgt = win(pos, vel, ts)
                adapter.deactivate()
                with torch.no_grad():
                    out_b = p.low(h.unsqueeze(0).to(device), force_predict=True)
                pb = out_b['predictions'][0].cpu()

                adapter.activate()
                adapter.load_lora_state(lora_s)
                adapter.load_head_state(head_s)
                with torch.no_grad():
                    out_a = p.low(h.unsqueeze(0).to(device), force_predict=True)
                pa = out_a['predictions'][0].cpu()

                def jerk(p):
                    v=torch.diff(p,dim=0); a=torch.diff(v,dim=0)
                    j=torch.diff(a,dim=0); return j.abs().mean().item()

                test_res.append({
                    'start': ts, 'hist': h, 'target': tgt,
                    'pred_b': pb, 'pred_a': pa,
                    'jerk_b': jerk(pb), 'jerk_a': jerk(pa),
                    'err_b4': torch.norm(pb[19]-tgt[19]).item(),
                    'err_a4': torch.norm(pa[19]-tgt[19]).item(),
                })

            adapter.deactivate()
            all_results[key] = test_res

    # ---- Figure: Honest XY, best 4 test windows per trajectory ----
    fig, axes = plt.subplots(6, 4, figsize=(24, 30))
    fig.suptitle('LoRA with 780 Training Windows (1000-frame trajectory) — Honest XY Assessment\n'
                 'Top 3 rows: r=8  |  Bottom 3 rows: r=32  |  Blue=History Red=Pred Green=Truth\n'
                 'Red dots = per-frame prediction (clustered=smooth, scattered=jagged)',
                 fontsize=14, fontweight='bold', y=1.005)

    for t_idx, (tname, _, _) in enumerate(configs):
        for rank_idx, rank in enumerate([8, 32]):
            key = '%s_r%d' % (tname, rank)
            res = all_results[key]
            row_base = t_idx + rank_idx * 3
            # Pick 4 representative test windows
            picks = [0, 2, 4, 6]  # starts: 820, 860, 900, 940
            for col, pi in enumerate(picks):
                r = res[pi]
                ax = axes[row_base, col]

                # Show both Before and After on same plot?
                # Better: show After in detail since user cares about trajectory quality
                draw_xy_detail(ax, r['hist'], r['pred_a'], r['target'],
                               '%s r=%d start=%d AFTER\n4s err=%.3f jerk=%.4f\nBefore 4s=%.3f jerk=%.4f' % (
                                   tname, rank, r['start'],
                                   r['err_a4'], r['jerk_a'],
                                   r['err_b4'], r['jerk_b']))

    plt.tight_layout(pad=1.5)
    p1 = out_dir / '04_lora_large_data.png'
    fig.savefig(p1, dpi=120, bbox_inches='tight')
    print('Saved:', p1)
    plt.close()

    # ---- Summary ----
    print()
    print('=' * 120)
    print('LoRA with 780 training windows (vs 90 before) — 4s Endpoint Error + Jerk')
    print('%-15s %-6s %-12s %-12s %-12s %-12s %-12s %-12s' % (
        'Trajectory','Rank','B4 err','A4 err','Delta','B4 jerk','A4 jerk','Jerk change'))
    print('-' * 100)
    for tname, _, _ in configs:
        for rank in [8, 32]:
            key = '%s_r%d'%(tname,rank)
            res = all_results[key]
            b4 = np.mean([r['err_b4'] for r in res])
            a4 = np.mean([r['err_a4'] for r in res])
            bj = np.mean([r['jerk_b'] for r in res])
            aj = np.mean([r['jerk_a'] for r in res])
            print('%-15s %-6s %-12.4f %-12.4f %+-12.4f %-12.6f %-12.6f %+-12.6f' % (
                tname, 'r=%d'%rank, b4, a4, b4-a4, bj, aj, aj-bj))
    print()
    print('Jerk increase = trajectory became LESS smooth (bad)')
    print('Jerk decrease = trajectory became MORE smooth (good)')
    print('If jerk increases despite more data: LoRA fundamentally unsuitable')
    print('If jerk decreases: more data + mild regularization works')


if __name__ == '__main__':
    main()
