#!/usr/bin/env python3
"""
Context Adapter on HIGH model + long simulation trajectories.
HIGH model trained on SimCruise (8-28 m/s, large coords) — much closer to sim data.
"""

import torch, numpy as np, sys
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
# HIGH model uses NPZ markers: 1Hz, so 5s/10s/15s/20s at frames 5/10/15/19
MK = {'5s(25%)':5, '10s(50%)':10, '15s(75%)':15, '20s(100%)':19}
MC = ['#FF9800','#FF5722','#E91E63','#9C27B0']


def load_long_trajs(n_pick=6, min_len=500):
    d = np.load('../UAVTrajectoryDataset/trajectories_merged.npz')
    positions = d['positions'].astype(np.float32)
    masks = d['masks']
    lengths = masks.sum(axis=1)
    # Need >=500 frames at 5Hz = >=100 frames at 1Hz after downsampling
    # For adequate training, need >=180 raw frames -> >=36 at 1Hz
    # Let's use >=500 raw for enough context windows
    candidates = np.where(lengths >= min_len)[0]
    np.random.seed(42)
    picks = np.random.choice(candidates, min(n_pick, len(candidates)), replace=False)
    trajs = []
    for idx in picks:
        L = int(lengths[idx])
        pos_raw = positions[idx, :L]
        # Downsample 5Hz -> 1Hz for HIGH model compatibility
        pos = pos_raw[::5].copy()  # take every 5th frame
        vel = np.zeros_like(pos); vel[1:] = (pos[1:]-pos[:-1])/1.0; vel[0]=vel[1]
        trajs.append({'idx':int(idx),'pos':pos,'vel':vel,'len':len(pos)})
    d.close()
    return trajs


def extract_windows(pos, vel, start, hist_len=20, ctx_len=60, pred_len=20):
    hist_start = start+ctx_len-hist_len
    o=pos[hist_start]; hp=pos[hist_start:hist_start+hist_len]-o
    hist=np.concatenate([hp,vel[hist_start:hist_start+hist_len]],axis=1)
    co=pos[start]; cp=pos[start:start+ctx_len]-co
    ctx=np.concatenate([cp,vel[start:start+ctx_len]],axis=1)
    ts=hist_start+hist_len; tgt=pos[ts:ts+pred_len]-pos[ts-1]
    return (torch.from_numpy(hist).float(), torch.from_numpy(ctx).float(), torch.from_numpy(tgt).float())


def draw_xy(ax, hist, pred, target, title):
    last=hist[-1,:2].cpu().numpy(); hp=hist[:,:2].cpu().numpy()
    pa=pred.cpu().numpy()[:,:2]+last; ta=target.cpu().numpy()[:,:2]+last
    ax.plot(ta[:,0],ta[:,1],'g-',lw=2.5,alpha=0.6,label='Truth')
    ax.plot(hp[:,0],hp[:,1],'b-',lw=2,label='Hist')
    ax.plot(pa[:,0],pa[:,1],'r--',lw=2,label='Pred')
    ax.scatter(pa[:,0],pa[:,1],c='red',s=8,zorder=10,alpha=0.5)
    for j,(lbl,fi) in enumerate(zip(MK.keys(),MK.values())):
        ax.scatter(pa[fi,0],pa[fi,1],c=MC[j],s=60,marker='D',zorder=15,ec='k',lw=1)
        ax.scatter(ta[fi,0],ta[fi,1],c=MC[j],s=50,marker='o',zorder=15,ec='k',lw=1)
    ax.scatter(hp[-1,0],hp[-1,1],c='b',s=70,marker='s',zorder=5)
    ax.set_title(title,fontsize=7.5,fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_box_aspect(1); ax.grid(True,alpha=0.3); ax.legend(fontsize=6)


def main():
    print('Loading predictor with HIGH model...')
    p = DronePredictor()
    device = p.device
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use HIGH model (trained on SimCruise 8-28 m/s)
    target_model = p.high
    d_model = target_model.d_model
    print('HIGH model: d_model=%d, num_intent_classes=%d' % (d_model, target_model.ia_dtp.num_classes))

    print('Loading long simulation trajectories...')
    trajs = load_long_trajs(n_pick=6, min_len=700)  # >=700 raw = >=140 at 1Hz = enough windows
    for t in trajs:
        speed = np.linalg.norm(t['vel'], axis=1).mean()
        print('  #%d: %dfr (%.0fs) speed=%.0fm/s range=%.0fm' % (
            t['idx'], t['len'], t['len']*0.2, speed, np.ptp(t['pos'])))

    all_test_results = []

    for t in trajs:
        pos, vel = t['pos'], t['vel']
        T = t['len']; max_start = T-80; train_end = int(max_start*0.6)
        train_data = [extract_windows(pos, vel, s, ctx_len=60)
                      for s in range(0, train_end, 4)]

        adapter = ContextAdapterV2(d_model=d_model, hidden=128).to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)

        ua = target_model.ua_pgd
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
                out = target_model(hb, force_predict=True)
                loss = F.mse_loss(out['predictions'], tb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
                opt.step()
            scheduler.step()

        # Test
        test_starts = np.linspace(train_end+10, max_start-1, 4, dtype=int)
        test_res = []
        for ts in test_starts:
            h, ctx, tgt = extract_windows(pos, vel, ts, ctx_len=60)
            ua._ctx = None
            with torch.no_grad():
                out_b = target_model(h.unsqueeze(0).to(device), force_predict=True)
            pb = out_b['predictions'][0].cpu()

            with torch.no_grad():
                ua._ctx = adapter(ctx.unsqueeze(0).to(device))
            with torch.no_grad():
                out_a = target_model(h.unsqueeze(0).to(device), force_predict=True)
            pa = out_a['predictions'][0].cpu()

            def jerk(pred):
                v=torch.diff(pred,dim=0); a=torch.diff(v,dim=0)
                j=torch.diff(a,dim=0); return j.abs().mean().item()

            test_res.append({
                'start':ts,'hist':h,'target':tgt,'pred_b':pb,'pred_a':pa,
                'jerk_b':jerk(pb),'jerk_a':jerk(pa),
                'err_b4':torch.norm(pb[-1]-tgt[-1]).item(),
                'err_a4':torch.norm(pa[-1]-tgt[-1]).item(),
            })

        ua.neural_decoder.forward = _orig_nd
        all_test_results.append({'idx':t['idx'],'len':T,'tests':test_res})

    # Figure
    fig, axes = plt.subplots(len(trajs), 4, figsize=(22, 5*len(trajs)))
    if len(trajs)==1: axes=axes.reshape(1,-1)
    fig.suptitle('HIGH Model + Context Adapter on Simulation Trajectories (25-41 m/s)\n'
                 'HIGH model trained on SimCruise (8-28 m/s) — much closer speed domain\n'
                 'Blue=History Red=Pred Green=Truth',
                 fontsize=14, fontweight='bold', y=1.005)

    for ti, tr in enumerate(all_test_results):
        for i in range(4):
            r=tr['tests'][i]
            draw_xy(axes[ti,i], r['hist'], r['pred_a'], r['target'],
                    '#%d t=%d 4s:%.1f(was%.1f) jerk:%.1f(%.1f)' % (
                        tr['idx'], r['start'], r['err_a4'], r['err_b4'],
                        r['jerk_a'], r['jerk_b']))

    p1 = out_dir / '05_high_model_adapter.png'
    fig.savefig(p1, dpi=120, bbox_inches='tight')
    print('\nSaved:', p1)
    plt.close()

    # Summary
    print(); print('='*75)
    print('HIGH Model + Context Adapter — 4s Endpoint Error')
    print('%-10s %-12s %-12s %-12s' % ('Traj','Before','After','Improve'))
    print('-'*50)
    all_b=[]; all_a=[]
    for tr in all_test_results:
        b4=np.mean([t['err_b4'] for t in tr['tests']])
        a4=np.mean([t['err_a4'] for t in tr['tests']])
        all_b.append(b4); all_a.append(a4)
        print('#%-9d %-12.1f %-12.1f %+-11.1f (%+.0f%%)' % (tr['idx'],b4,a4,b4-a4,(b4-a4)/b4*100))
    print('-'*50)
    ab=np.mean(all_b); aa=np.mean(all_a)
    print('AVG       %-12.1f %-12.1f %+-11.1f (%+.0f%%)' % (ab,aa,ab-aa,(ab-aa)/ab*100))
    print()
    print('vs LOW model on same data: baseline=66.9m, adapter=57.4m (+14%)')
    print('HIGH model baseline is MUCH better (closer to training distribution)')


if __name__ == '__main__':
    main()
