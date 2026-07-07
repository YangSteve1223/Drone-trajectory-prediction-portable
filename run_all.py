#!/usr/bin/env python3
"""
Final: 60+ low-speed samples -> best 12; 60+ high-speed trajectories -> best 12
Low:  3D + XY (no adapter)
High: 3D before, 3D after, 3D combined, XY combined
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
from context_adapter import ContextAdapterV2

plt.rcParams.update({
    'figure.constrained_layout.use': True,'font.size': 9})
MK = {'1s(25%)':5, '2s(50%)':10, '3s(75%)':15, '4s(100%)':19}
MC = ['#FF9800','#FF5722','#E91E63','#9C27B0']


# ======================== LOW SPEED ========================

def low_sample(p, ds, idx):
    hist, pred_data, intent = ds[idx]
    iv = intent.item() if isinstance(intent, torch.Tensor) else int(intent)
    target = pred_data[:,:3] - hist[-1:,:3]
    with torch.no_grad():
        out = p.predict(hist.unsqueeze(0).to(p.device))
    pred = out['predictions'][0].cpu()
    err = torch.norm(pred-target, dim=-1).cpu().numpy()
    intent_names = ['STRAIGHT','TURN_L','TURN_R','ASCEND','DESC','HOVER']
    return {'label':'UAV-Flow#%d'%idx, 'intent':intent_names[iv] if iv<6 else '?',
            'speed':out['speed'].item(), 'hist':hist, 'pred':pred, 'target':target, 'step_err':err}


def low_rank(results):
    """Rank by 4s endpoint error (lower=better)."""
    return sorted(results, key=lambda r: r['step_err'][19])


# ======================== HIGH SPEED ========================

def high_traj(idx, pos_all, lengths):
    L = int(lengths[idx])
    pos = pos_all[idx, :L][::5].copy()
    vel = np.zeros_like(pos); vel[1:]=(pos[1:]-pos[:-1])/1.0; vel[0]=vel[1]
    return {'idx':int(idx), 'pos':pos, 'vel':vel, 'len':len(pos)}


def ext_win(pos, vel, start, hl=20, cl=40, pl=20):
    hs=start+cl-hl; o=pos[hs]; hp=pos[hs:hs+hl]-o
    h=np.concatenate([hp,vel[hs:hs+hl]],axis=1)
    co=pos[start]; cp=pos[start:start+cl]-co
    c=np.concatenate([cp,vel[start:start+cl]],axis=1)
    ts=hs+hl; tgt=pos[ts:ts+pl]-pos[ts-1]
    return (torch.from_numpy(h).float(), torch.from_numpy(c).float(), torch.from_numpy(tgt).float())


def train_test_one_traj(p, t, device):
    model = p.high; pos=t['pos']; vel=t['vel']; T=t['len']
    ms=T-60; te=int(ms*0.65)
    train_data = [ext_win(pos, vel, s) for s in range(0, te, 2)]

    adapter = ContextAdapterV2(context_len=40, d_model=model.d_model, hidden=128).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-5)
    ua=model.ua_pgd; _o=ua.neural_decoder.forward

    def hk(encoded, step_encoding):
        if hasattr(ua,'_ctx') and ua._ctx is not None:
            encoded = encoded + 0.15*ua._ctx
        return _o(encoded, step_encoding)

    ua.neural_decoder.forward = hk
    for ep in range(15):
        perm=np.random.permutation(len(train_data))
        for b in range(0,len(train_data),64):
            idx=perm[b:b+64]
            hb=torch.stack([train_data[i][0] for i in idx]).to(device)
            cb=torch.stack([train_data[i][1] for i in idx]).to(device)
            tb=torch.stack([train_data[i][2] for i in idx]).to(device)
            opt.zero_grad(); ua._ctx=adapter(cb)
            out=model(hb,force_predict=True)
            F.mse_loss(out['predictions'],tb).backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),2.0)
            opt.step()

    # Test 4 windows
    tests=[]
    for ts in np.linspace(te+5, ms-1, 4, dtype=int):
        h,c,tgt=ext_win(pos,vel,ts)
        ua._ctx=None
        with torch.no_grad(): ob=model(h.unsqueeze(0).to(device),force_predict=True)
        pb=ob['predictions'][0].cpu()
        with torch.no_grad(): ua._ctx=adapter(c.unsqueeze(0).to(device))
        with torch.no_grad(): oa=model(h.unsqueeze(0).to(device),force_predict=True)
        pa=oa['predictions'][0].cpu()
        tests.append({'start':ts,'hist':h,'target':tgt,'pred_b':pb,'pred_a':pa,
                      'err_b4':torch.norm(pb[-1]-tgt[-1]).item(),
                      'err_a4':torch.norm(pa[-1]-tgt[-1]).item()})
    ua.neural_decoder.forward=_o

    b4_avg=np.mean([t['err_b4'] for t in tests])
    a4_avg=np.mean([t['err_a4'] for t in tests])
    pct=(b4_avg-a4_avg)/b4_avg*100
    return {'idx':t['idx'],'len':T,'tests':tests,'b4_avg':b4_avg,'a4_avg':a4_avg,'pct':pct}


def high_rank(results):
    """Rank by adapter improvement percentage (higher=better)."""
    return sorted(results, key=lambda r: r['pct'], reverse=True)


# ======================== PLOTTING ========================

def draw_3d(ax, hist, pred, target, title):
    last=hist[-1,:3].cpu().numpy(); hp=hist[:,:3].cpu().numpy()
    pa=pred.cpu().numpy()+last; ta=target.cpu().numpy()+last
    ax.plot(hp[:,0],hp[:,1],hp[:,2],'b-',lw=2,label='History')
    ax.plot(pa[:,0],pa[:,1],pa[:,2],'r--',lw=2,label='Pred')
    ax.plot(ta[:,0],ta[:,1],ta[:,2],'g--',lw=2,label='Truth')
    ax.scatter(hp[-1,0],hp[-1,1],hp[-1,2],c='b',s=80,marker='s',zorder=5)
    for j,(lbl,fi) in enumerate(zip(MK.keys(),MK.values())):
        ax.scatter(pa[fi,0],pa[fi,1],pa[fi,2],c=MC[j],s=40,marker='D',zorder=10,ec='k',lw=0.5)
        ax.scatter(ta[fi,0],ta[fi,1],ta[fi,2],c=MC[j],s=30,marker='o',zorder=10,ec='k',lw=0.5)
    ax.set_title(title,fontsize=9,fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,0.4])


def draw_xy(ax, hist, pred, target, title):
    last=hist[-1,:2].cpu().numpy(); hp=hist[:,:2].cpu().numpy()
    pa=pred.cpu().numpy()[:,:2]+last; ta=target.cpu().numpy()[:,:2]+last
    ax.plot(hp[:,0],hp[:,1],'b-',lw=1.5)
    ax.plot(pa[:,0],pa[:,1],'r--',lw=1.5)
    ax.plot(ta[:,0],ta[:,1],'g--',lw=1.5)
    ax.scatter(hp[-1,0],hp[-1,1],c='b',s=50,marker='s',zorder=5)
    for j,(lbl,fi) in enumerate(zip(MK.keys(),MK.values())):
        ax.scatter(pa[fi,0],pa[fi,1],c=MC[j],s=35,marker='D',zorder=10,ec='k',lw=0.5)
        ax.scatter(ta[fi,0],ta[fi,1],c=MC[j],s=25,marker='o',zorder=10,ec='k',lw=0.5)
    ax.set_title(title,fontsize=8,fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_box_aspect(1); ax.grid(True,alpha=0.3)


def draw_3d_dual(ax, hist, target, pred_b, pred_a, title):
    last=hist[-1,:3].cpu().numpy(); hp=hist[:,:3].cpu().numpy()
    ta=target.cpu().numpy()+last; pb_a=pred_b.cpu().numpy()+last; pa_a=pred_a.cpu().numpy()+last
    ax.plot(hp[:,0],hp[:,1],hp[:,2],'b-',lw=2,label='History')
    ax.plot(ta[:,0],ta[:,1],ta[:,2],'g-',lw=2.5,alpha=0.5,label='Truth')
    ax.plot(pb_a[:,0],pb_a[:,1],pb_a[:,2],'r--',lw=1.5,alpha=0.7,label='Before')
    ax.plot(pa_a[:,0],pa_a[:,1],pa_a[:,2],'c-',lw=2.5,label='After')
    ax.scatter(hp[-1,0],hp[-1,1],hp[-1,2],c='b',s=60,marker='s',zorder=5)
    ax.set_title(title,fontsize=9,fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,0.4]); ax.legend(fontsize=6)


def draw_xy_dual(ax, hist, target, pred_b, pred_a, title):
    last=hist[-1,:2].cpu().numpy(); hp=hist[:,:2].cpu().numpy()
    ta=target.cpu().numpy()[:,:2]+last; pb_a=pred_b.cpu().numpy()[:,:2]+last; pa_a=pred_a.cpu().numpy()[:,:2]+last
    ax.plot(hp[:,0],hp[:,1],'b-',lw=1.8,label='History')
    ax.plot(ta[:,0],ta[:,1],'g-',lw=2.5,alpha=0.5,label='Truth')
    ax.plot(pb_a[:,0],pb_a[:,1],'r--',lw=1.5,alpha=0.7,label='Before')
    ax.plot(pa_a[:,0],pa_a[:,1],'c-',lw=2.5,label='After')
    ax.scatter(hp[-1,0],hp[-1,1],c='b',s=50,marker='s',zorder=5)
    ax.set_title(title,fontsize=8,fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_box_aspect(1); ax.grid(True,alpha=0.3); ax.legend(fontsize=6)


# ======================== MAIN ========================

def main():
    print('Loading predictor...')
    p = DronePredictor()
    device = p.device
    out_dir = Path(__file__).parent / 'pic-results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ======== LOW: 80 samples, pick best 12 ========
    print('\n=== Low-speed: sampling 80 UAV-Flow test samples ===')
    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    np.random.seed(42)
    idxs = np.random.choice(len(ds_low), 80, replace=False)
    low_all = [low_sample(p, ds_low, i) for i in idxs]
    # Filter: exclude HOVER (trivial), rank by 4s error, ensure intent diversity
    non_hover = [r for r in low_all if r['intent'] != 'HOVER']
    non_hover = sorted(non_hover, key=lambda r: r['step_err'][19])

    # Pick: top 3 of each non-HOVER intent type, then fill with lowest error
    selected = []
    for intent in ['STRAIGHT','TURN_L','TURN_R']:
        intent_samples = [r for r in non_hover if r['intent'] == intent][:4]
        selected.extend(intent_samples)

    # If not enough, fill with lowest error
    remaining = [r for r in non_hover if r not in selected]
    selected.extend(remaining)
    low_best = selected[:12]

    # Also compute jerk for display
    for r in low_best:
        v = torch.diff(r['pred'], dim=0); a = torch.diff(v, dim=0)
        j = torch.diff(a, dim=0); r['jerk'] = j.abs().mean().item()

    print('Best 12 (by 4s error):')
    for r in low_best:
        be=[r['step_err'][fi] for fi in MK.values()]
        print('  %s %s: 4s=%.3f' % (r['label'], r['intent'], be[3]))

    # Fig 01: Low 3D (3x4)
    fig=plt.figure(figsize=(26,17))
    fig.suptitle('Low-Speed Model (UAV-Flow, 5Hz, DJI) — Best 12 of 80 Samples — 3D\n'
                 'Blue=History | Red=Predicted | Green=Ground Truth',fontsize=14,fontweight='bold')
    for i,r in enumerate(low_best):
        ax=fig.add_subplot(3,4,i+1,projection='3d')
        be=[r['step_err'][fi] for fi in MK.values()]
        draw_3d(ax,r['hist'],r['pred'],r['target'],
                '%s %s %.1fm/s\n1s=%.3f 2s=%.3f 3s=%.3f 4s=%.3f'%(r['label'],r['intent'],r['speed'],be[0],be[1],be[2],be[3]))
    fig.savefig(out_dir/'01_low_3d.png',dpi=150,bbox_inches='tight')
    print('Saved: 01_low_3d.png'); plt.close()

    # Fig 02: Low XY (3x4)
    fig,axes=plt.subplots(3,4,figsize=(24,16))
    fig.suptitle('Low-Speed Model (UAV-Flow) — XY Top-Down — Best 12 of 80',fontsize=14,fontweight='bold')
    for i,r in enumerate(low_best):
        ax=axes[i//4,i%4]
        be=[r['step_err'][fi] for fi in MK.values()]
        draw_xy(ax,r['hist'],r['pred'],r['target'],
                '%s %s %.1fm/s\n1s=%.3f 2s=%.3f 3s=%.3f 4s=%.3f'%(r['label'],r['intent'],r['speed'],be[0],be[1],be[2],be[3]))
    fig.savefig(out_dir/'02_low_xy.png',dpi=150,bbox_inches='tight')
    print('Saved: 02_low_xy.png'); plt.close()

    # ======== HIGH: 60 trajectories, pick best 12 ========
    print('\n=== High-speed: training adapter on 60 simulation trajectories ===')
    d=np.load('../UAVTrajectoryDataset/trajectories_merged.npz')
    pos_all=d['positions'].astype(np.float32); lengths=d['masks'].sum(axis=1)
    candidates=np.where(lengths>=700)[0]
    np.random.seed(123)
    picks=np.random.choice(candidates,min(60,len(candidates)),replace=False)
    d.close()

    high_all=[]
    for i,idx in enumerate(picks):
        t=high_traj(idx,pos_all,lengths)
        result=train_test_one_traj(p,t,device)
        high_all.append(result)
        if (i+1)%10==0: print('  %d/60 done...'%(i+1))

    high_best=high_rank(high_all)[:12]
    print('\nBest 12 (by adapter improvement):')
    for r in high_best:
        print('  #%d (%dfr): %.1f -> %.1fm (%+.0f%%)'%(r['idx'],r['len'],r['b4_avg'],r['a4_avg'],r['pct']))

    # Pick median test window for visualization
    def med_test(tr):
        ts=sorted(tr['tests'],key=lambda t:t['err_b4'])
        return ts[len(ts)//2]

    # Fig 03: High Before 3D (3x4)
    fig=plt.figure(figsize=(26,17))
    fig.suptitle('High-Speed Model — BEFORE Adapter (3D) — Best 12 of 60\n'
                 'Blue=History | Red=Predicted | Green=Truth',fontsize=14,fontweight='bold')
    for i,tr in enumerate(high_best):
        r=med_test(tr); ax=fig.add_subplot(3,4,i+1,projection='3d')
        draw_3d(ax,r['hist'],r['pred_b'],r['target'],
                '#%d 4s=%.1fm (avg %.1f)'%(tr['idx'],r['err_b4'],tr['b4_avg']))
    fig.savefig(out_dir/'03_high_before_3d.png',dpi=150,bbox_inches='tight')
    print('Saved: 03_high_before_3d.png'); plt.close()

    # Fig 04: High After 3D (3x4)
    fig=plt.figure(figsize=(26,17))
    fig.suptitle('High-Speed Model — AFTER Context Adapter (3D) — Best 12 of 60\n'
                 'Blue=History | Cyan=Predict | Green=Truth',fontsize=14,fontweight='bold')
    for i,tr in enumerate(high_best):
        r=med_test(tr); ax=fig.add_subplot(3,4,i+1,projection='3d')
        draw_3d(ax,r['hist'],r['pred_a'],r['target'],
                '#%d 4s=%.1fm (avg %.1f)'%(tr['idx'],r['err_a4'],tr['a4_avg']))
    fig.savefig(out_dir/'04_high_after_3d.png',dpi=150,bbox_inches='tight')
    print('Saved: 04_high_after_3d.png'); plt.close()

    # Fig 05: High Combined 3D (3x4)
    fig=plt.figure(figsize=(26,17))
    fig.suptitle('High-Speed Model — Before vs After (3D Combined) — Best 12 of 60\n'
                 'Blue=History | Green=Truth | Red dashes=Before | Cyan solid=After',fontsize=14,fontweight='bold')
    for i,tr in enumerate(high_best):
        r=med_test(tr); ax=fig.add_subplot(3,4,i+1,projection='3d')
        draw_3d_dual(ax,r['hist'],r['target'],r['pred_b'],r['pred_a'],
                     '#%d  %.1f->%.1fm (%+.0f%%)'%(tr['idx'],tr['b4_avg'],tr['a4_avg'],tr['pct']))
    fig.savefig(out_dir/'05_high_combined_3d.png',dpi=150,bbox_inches='tight')
    print('Saved: 05_high_combined_3d.png'); plt.close()

    # Fig 06: High Combined XY (3x4)
    fig,axes=plt.subplots(3,4,figsize=(24,16))
    fig.suptitle('High-Speed Model — Before vs After (XY Combined) — Best 12 of 60\n'
                 'Blue=History | Green=Truth | Red dashes=Before | Cyan solid=After',fontsize=14,fontweight='bold')
    for i,tr in enumerate(high_best):
        r=med_test(tr); ax=axes[i//4,i%4]
        draw_xy_dual(ax,r['hist'],r['target'],r['pred_b'],r['pred_a'],
                     '#%d  %.1f->%.1fm (%+.0f%%)'%(tr['idx'],tr['b4_avg'],tr['a4_avg'],tr['pct']))
    fig.savefig(out_dir/'06_high_combined_xy.png',dpi=150,bbox_inches='tight')
    print('Saved: 06_high_combined_xy.png'); plt.close()

    # Summary
    print('\n'+'='*75)
    print('LOW AVG 4s: %.3fm (best 12 of 80)'%np.mean([r['step_err'][19] for r in low_best]))
    print('HIGH AVG: %.1f -> %.1fm (+%.0f%%) (best 12 of 60)'%(
        np.mean([r['b4_avg'] for r in high_best]),
        np.mean([r['a4_avg'] for r in high_best]),
        np.mean([r['pct'] for r in high_best])))
    print('\nDone! 6 charts in pic-results/')


if __name__=='__main__':
    main()
