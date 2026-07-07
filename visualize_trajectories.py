#!/usr/bin/env python3
"""Comprehensive trajectory charts — LOW & HIGH, 12 best samples each, time-segmented errors."""
import torch, numpy as np, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder
from utils.fast_data_loader import FastWindowDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'figure.constrained_layout.use': True, 'font.size': 7,
                     'axes.titlesize': 8, 'axes.labelsize': 7})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path(__file__).parent / 'pic-results'; OUT.mkdir(parents=True, exist_ok=True)
INTENT_6 = ['STRAIGHT','TURN_L','TURN_R','ASCEND','DESC','HOVER']
INTENT_4 = ['STRAIGHT','TURN_L','TURN_R','DESCEND']
MH_C = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA']
HC, GC, SC = '#37474F', '#00C853', '#FF6D00'

# ── Models ─────────────────────────────────────────────────────────
def load_low():
    m = TrajectoryPredictor(input_dim=6,history_len=20,pred_len=20,d_model=128,d_state=16,d_conv=4,expand=2,emam_n_layers=2,num_intent_classes=6,use_trigger=True,trigger_mode='simple').to(DEVICE).eval()
    c = torch.load('weights/low_speed_6class.pth',map_location=DEVICE,weights_only=False)
    m.load_state_dict(c['model_state_dict']); return m

def load_high_single():
    m = TrajectoryPredictor(input_dim=6,history_len=20,pred_len=20,d_model=128,d_state=16,d_conv=4,expand=2,emam_n_layers=2,num_intent_classes=4,use_trigger=True,trigger_mode='simple').to(DEVICE).eval()
    c = torch.load('weights/high_speed_4class.pth',map_location=DEVICE,weights_only=False)
    m.load_state_dict(c['model_state_dict']); return m

def load_high_multi():
    m = TrajectoryPredictor(input_dim=6,history_len=20,pred_len=20,d_model=128,d_state=16,d_conv=4,expand=2,emam_n_layers=2,num_intent_classes=4,use_trigger=True,trigger_mode='simple').to(DEVICE).eval()
    c = torch.load('weights/high_speed_4class.pth',map_location=DEVICE,weights_only=False)
    m.load_state_dict(c['model_state_dict']); m._norm_input = False
    m._get_scale_pos = lambda: 100.0
    def _n(h):
        s = h.new_tensor([100.,100.,100.,10.,10.,10.])
        return h/s.unsqueeze(0).unsqueeze(0)
    m._normalize = _n
    d = m.ua_pgd.replace_with_multi_head(K=5,noise_std=0.0)
    mc = torch.load('weights/high_multihead_K5.pth',map_location=DEVICE,weights_only=False)
    d.load_state_dict(mc['multi_decoder_state']); d = d.to(DEVICE); return m

# ── Predict ────────────────────────────────────────────────────────
@torch.no_grad()
def pred_low(m, h): return m(h, force_predict=True)['predictions']

@torch.no_grad()
def pred_high_single(m, h):
    o = m(h, force_predict=True); p = o['predictions'].clone()
    ip = torch.softmax(o['intent_logits'],dim=-1); dp = ip[:,3]
    dm = torch.ones(h.shape[0],device=DEVICE)
    sm = dp<0.05; wm = (dp>=0.05)&(dp<0.20)
    if sm.any(): dm[sm]=0.05
    if wm.any():
        t_ = (dp[wm]-0.05)/0.15; dm[wm]=0.30+0.70*t_
    ap = dm<1.0
    if ap.any(): p[ap,:,2]*=dm[ap].view(-1,1)
    return p

@torch.no_grad()
def pred_high_multi(m, h):
    hn = m._normalize(h); enc = m.emam_se(hn)
    dtp = m.ia_dtp(enc, historical_trajectory=hn)
    mh = m.ua_pgd.forward_multi_head(encoded_feat=enc, global_anchor=dtp['global_anchor'],
                                      historical_trajectory=hn, intent_weights=dtp['intent_weights'])
    return mh['all_predictions'], mh['predictions'], mh['confidences']

# ── Scoring ────────────────────────────────────────────────────────
def smoothness(t):
    if t.shape[0]<3: return 100.0
    v = t[1:]-t[:-1]; n = np.linalg.norm(v,axis=1)+1e-8; vn = v/n[:,None]
    d = np.clip(np.sum(vn[1:]*vn[:-1],axis=1),-1,1)
    return float(np.degrees(np.mean(np.abs(np.arccos(d)))))

def score_low(hist, target, pred, intent):
    lp = hist[-1,:3]; ga = target[:,:3]; pa = lp+pred[:,:3]
    ade = float(np.mean(np.linalg.norm(pa-ga,axis=1)))
    fde = float(np.linalg.norm(pa[-1]-ga[-1]))
    gs = smoothness(ga); ps = smoothness(pa)
    ext = float(np.linalg.norm(ga[-1]-ga[0]))
    spd = float(np.linalg.norm(hist[-5:,3:6],axis=1).mean())
    s = 0.0
    s += max(0,5.0-ade)*3.0 + max(0,10.0-fde)*2.0
    s += max(0,20.0-gs)*0.5 - max(0,ps-30.0)*0.3
    s += min(ext,8.0)*1.0 - abs(ade-fde)*0.5 + min(spd,3.0)*1.5
    return s

def score_high(hist, target, all_preds, best_pred, conf, intent):
    lp = hist[-1,:3]; ga = target[:,:3]
    mf = float('inf')
    for k in range(all_preds.shape[0]):
        e = np.linalg.norm(lp+all_preds[k,:,:3][-1]-ga[-1]); mf = min(mf,e)
    ba = lp+best_pred[:,:3]; bade = float(np.mean(np.linalg.norm(ba-ga,axis=1)))
    gs = smoothness(ga); ext = float(np.linalg.norm(ga[-1]-ga[0]))
    cp = torch.softmax(conf,dim=1)[0].cpu().numpy(); cs = float(np.std(cp))
    s = max(0,5.0-mf)*5.0 + max(0,3.0-bade)*3.0 + max(0,15.0-gs)*0.5
    s += min(ext,50.0)*0.3 + cs*20.0
    if intent==3: s+=5.0
    return s, mf

# ── Time segments ──────────────────────────────────────────────────
LSEG = [(0,5,'0-1s'),(5,10,'1-2s'),(10,15,'2-3s'),(15,20,'3-4s')]
HSEG = [(0,5,'0-5s'),(5,10,'5-10s'),(10,15,'10-15s'),(15,20,'15-20s')]

def seg_errs(pa, ga, segs):
    return [(lbl, np.linalg.norm(pa[s:e]-ga[s:e],axis=1)) for s,e,lbl in segs]

# ── Collect ────────────────────────────────────────────────────────
def collect_low(n_tgt=80):
    print(f'\n{"="*60}\nCollecting LOW samples (target {n_tgt})...')
    m = load_low(); ds = FastWindowDataset('../UAV-Flow-pure',split='test')
    ld = torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True,num_workers=0)
    scored = []; seen = 0
    for hb, tb, ib in ld:
        hb = hb.to(DEVICE); p = pred_low(m,hb)
        for i in range(hb.shape[0]):
            hist = hb[i].cpu().numpy(); tgt = tb[i].cpu().numpy()
            pr = p[i].cpu().numpy(); it = ib[i].item()
            s = score_low(hist,tgt,pr,it); scored.append((s,hist,tgt,pr,it)); seen+=1
        if seen>=n_tgt*3: break

    scored.sort(key=lambda x:x[0],reverse=True)
    sel = []; ic = defaultdict(int)
    for s,hi,ta,pr,it in scored:
        if ic[it]<5: sel.append((hi,ta,pr,it,s)); ic[it]+=1
        if len(sel)>=12: break
    while len(sel)<12:
        s,hi,ta,pr,it = scored[len(sel)]
        if (hi,ta,pr,it,s) not in sel: sel.append((hi,ta,pr,it,s))
    print(f'  Evaluated {seen}, selected {len(sel)}')
    for idx,(hi,ta,pr,it,s) in enumerate(sel):
        print(f'  #{idx+1}: {INTENT_6[it]:12s} score={s:.1f}')
    return sel

def collect_high(n_tgt=80):
    print(f'\n{"="*60}\nCollecting HIGH samples (target {n_tgt})...')
    m = load_high_multi(); ds = FastWindowDataset('../SimCruise',split='test',label_remap={4:3})
    ld = torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,num_workers=0)
    scored = []; seen = 0
    for hb, tb, ib in ld:
        hb = hb.to(DEVICE); ap,bp,cf = pred_high_multi(m,hb)
        for i in range(hb.shape[0]):
            hist = hb[i].cpu().numpy(); tgt = tb[i].cpu().numpy()
            a = ap[:,i].cpu().numpy(); b = bp[i].cpu().numpy()
            c = cf[i:i+1].cpu(); it = ib[i].item()
            if it>=4: continue
            s,mf = score_high(hist,tgt,a,b,c,it); scored.append((s,mf,hist,tgt,a,b,c,it)); seen+=1
        if seen>=n_tgt*3: break

    scored.sort(key=lambda x:x[0],reverse=True)
    sel = []; ic = defaultdict(int)
    for s,mf,hi,ta,a,b,c,it in scored:
        if ic[it]<5: sel.append((hi,ta,a,b,c,it,mf)); ic[it]+=1
        if len(sel)>=12: break
    while len(sel)<12:
        s,mf,hi,ta,a,b,c,it = scored[len(sel)]
        e = (hi,ta,a,b,c,it,mf)
        if e not in sel: sel.append(e)
    print(f'  Evaluated {seen}, selected {len(sel)}')
    for idx,(hi,ta,a,b,c,it,mf) in enumerate(sel):
        cp = torch.softmax(c,dim=1)[0].numpy()
        print(f'  #{idx+1}: {INTENT_4[it]:12s} minFDE={mf:.2f}m conf={cp}')
    return sel

# ── LOW charts ─────────────────────────────────────────────────────
def chart_low_3d(samples):
    fig = plt.figure(figsize=(22,26))
    fig.suptitle('LOW Model (UAV-Flow) — Best 12 Trajectory Predictions (3D)\n'
                 'Dark=History  Green=Ground Truth  Orange--=Prediction  '
                 'o=GT marker  s=Pred marker',
                 fontsize=12,fontweight='bold',y=0.99)
    for idx,(hist,target,pred,intent,score) in enumerate(samples):
        ax = fig.add_subplot(4,3,idx+1,projection='3d')
        lp = hist[-1,:3]; hp = hist[:,:3]; ga = target[:,:3]; pa = lp+pred[:,:3]
        ax.plot(hp[:,0],hp[:,1],hp[:,2],color=HC,lw=1.5,alpha=0.7,label='History')
        ax.plot(ga[:,0],ga[:,1],ga[:,2],color=GC,lw=2,label='GT')
        ax.plot(pa[:,0],pa[:,1],pa[:,2],color=SC,lw=1.8,ls='--',label='Pred')
        for s,e,lbl in LSEG:
            ax.scatter(*ga[e-1],c=GC,s=18,marker='o',alpha=0.8,zorder=5)
            ax.scatter(*pa[e-1],c=SC,s=18,marker='s',alpha=0.8,zorder=5)
        # Per-segment errors including Z
        se = seg_errs(pa,ga,LSEG)
        z_errs = [np.mean(np.abs(pa[s:e,2]-ga[s:e,2])) for s,e,_ in LSEG]
        es = ' | '.join([f'{l}:{np.mean(e):.2f}m' for l,e in se])
        zs = 'Z err: ' + ' | '.join([f'{lbl}:{ze:.2f}m' for ze,(_,_,lbl) in zip(z_errs,LSEG)])
        spd = float(np.linalg.norm(hist[-5:,3:6],axis=1).mean())
        ax.set_title(f'#{idx+1} {INTENT_6[intent]} spd={spd:.1f}m/s\n{es}\n{zs}',
                     fontsize=6,family='monospace')
        ax.xaxis.pane.fill=False; ax.yaxis.pane.fill=False; ax.zaxis.pane.fill=False
        ax.xaxis.pane.set_edgecolor('w'); ax.yaxis.pane.set_edgecolor('w'); ax.zaxis.pane.set_edgecolor('w')
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5,loc='upper left',ncol=3)
    p = OUT/'low_12_trajectories_3d.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

def chart_low_xy(samples):
    fig,axs = plt.subplots(4,3,figsize=(22,24))
    fig.suptitle('LOW Model (UAV-Flow) — Best 12 Trajectory Predictions (XY View)\n'
                 'Dark=History  Green=GT  Orange--=Pred  o=GT time marker  s=Pred marker',
                 fontsize=12,fontweight='bold')
    for idx,(hist,target,pred,intent,score) in enumerate(samples):
        ax = axs[idx//3,idx%3]; lp = hist[-1,:3]
        hp = hist[:,:3]; ga = target[:,:3]; pa = lp+pred[:,:3]
        ax.plot(hp[:,0],hp[:,1],color=HC,lw=1.5,alpha=0.7,label='History')
        ax.plot(ga[:,0],ga[:,1],color=GC,lw=2,label='GT')
        ax.plot(pa[:,0],pa[:,1],color=SC,lw=1.8,ls='--',label='Pred')
        for s,e,lbl in LSEG:
            ax.plot(ga[e-1,0],ga[e-1,1],'o',color=GC,ms=6,alpha=0.8)
            ax.plot(pa[e-1,0],pa[e-1,1],'s',color=SC,ms=6,alpha=0.8)
        se = seg_errs(pa,ga,LSEG)
        z_errs = [np.mean(np.abs(pa[s:e,2]-ga[s:e,2])) for s,e,_ in LSEG]
        es = ' | '.join([f'{l}:{np.mean(e):.2f}m' for l,e in se])
        zs = 'Z err: ' + ' | '.join([f'{lbl}:{ze:.2f}m' for ze,(_,_,lbl) in zip(z_errs,LSEG)])
        ax.set_title(f'#{idx+1} {INTENT_6[intent]} | {es}\n{zs}',fontsize=6,family='monospace')
        ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
        if idx==0: ax.legend(fontsize=6,loc='best')
    p = OUT/'low_12_trajectories_xy.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

def chart_low_error(samples):
    fig,ax = plt.subplots(figsize=(18,6))
    fig.suptitle('LOW Model — Per-Second Prediction Error (12 Samples)',fontsize=12,fontweight='bold')
    sl = [s[2] for s in LSEG]; x = np.arange(len(sl)); w = 0.7/len(samples)
    for idx,(hist,target,pred,intent,score) in enumerate(samples):
        lp = hist[-1,:3]; pa = lp+pred[:,:3]; ga = target[:,:3]
        se = seg_errs(pa,ga,LSEG); means = [np.mean(e) for _,e in se]
        off = (idx-len(samples)/2+0.5)*w
        ax.bar(x+off,means,w,alpha=0.85,label=f'#{idx+1} {INTENT_6[intent]}')
        for xi,m in zip(x,means): ax.text(xi+off,m+0.03,f'{m:.2f}',ha='center',fontsize=4.5,rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(sl,fontsize=9)
    ax.set_ylabel('Mean L2 Error (m)'); ax.legend(fontsize=5,ncol=4,loc='upper left'); ax.grid(True,alpha=0.3,axis='y')
    p = OUT/'low_12_persecond_error.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

# ── HIGH charts ────────────────────────────────────────────────────
def chart_high_multihyp_3d(samples):
    fig = plt.figure(figsize=(22,24))
    fig.suptitle('HIGH Model (SimCruise) — Multi-Hypothesis K=5 (3D)',
                 fontsize=13,fontweight='bold',y=0.98)
    for idx,(hist,target,all_preds,best_pred,conf,intent,min_fde) in enumerate(samples):
        ax = fig.add_subplot(4,3,idx+1,projection='3d')
        lp = hist[-1,:3]; hp = hist[:,:3]; ga = target[:,:3]
        ax.plot(hp[:,0],hp[:,1],hp[:,2],color=HC,lw=1.2,alpha=0.6)
        ax.plot(ga[:,0],ga[:,1],ga[:,2],color=GC,lw=2.5)
        cp = torch.softmax(conf,dim=1)[0].cpu().numpy()
        for k in range(5):
            ha = lp+all_preds[k,:,:3]; alpha = 0.4+0.6*cp[k]
            ax.plot(ha[:,0],ha[:,1],ha[:,2],color=MH_C[k],lw=1.3,alpha=max(0.35,alpha),
                   label=f'H{k+1}({cp[k]:.2f})')
        ba = lp+best_pred[:,:3]
        for s,e,lbl in HSEG:
            ax.scatter(*ga[e-1],c=GC,s=15,marker='o',alpha=0.8)
            ax.scatter(*ba[e-1],c='#00E676',s=12,marker='s',alpha=0.8)
        se = seg_errs(ba,ga,HSEG); es = ' | '.join([f'{l}:{np.mean(e):.1f}m' for l,e in se])
        ax.set_title(f'#{idx+1} {INTENT_4[intent]} minFDE={min_fde:.1f}m\n{es}',fontsize=6.5,family='monospace')
        ax.xaxis.pane.fill=False; ax.yaxis.pane.fill=False; ax.zaxis.pane.fill=False
        ax.xaxis.pane.set_edgecolor('w'); ax.yaxis.pane.set_edgecolor('w'); ax.zaxis.pane.set_edgecolor('w')
        ax.tick_params(labelsize=5)
    le2 = [Line2D([0],[0],color=HC,lw=1.2,label='History(20s)'),
           Line2D([0],[0],color=GC,lw=2.5,label='Ground Truth')]
    for k in range(5): le2.append(Line2D([0],[0],color=MH_C[k],lw=1.3,label=f'H{k+1}'))
    fig.legend(handles=le2,loc='lower center',ncol=7,fontsize=6)
    p = OUT/'high_12_multihyp_3d.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

def chart_high_best_3d(samples):
    fig = plt.figure(figsize=(22,24))
    fig.suptitle('HIGH Model — Best Confidence Trajectory (3D)',fontsize=13,fontweight='bold',y=0.98)
    for idx,(hist,target,all_preds,best_pred,conf,intent,min_fde) in enumerate(samples):
        ax = fig.add_subplot(4,3,idx+1,projection='3d')
        lp = hist[-1,:3]; hp = hist[:,:3]; ga = target[:,:3]; ba = lp+best_pred[:,:3]
        ax.plot(hp[:,0],hp[:,1],hp[:,2],color=HC,lw=1.5,alpha=0.7)
        ax.plot(ga[:,0],ga[:,1],ga[:,2],color=GC,lw=2.5)
        ax.plot(ba[:,0],ba[:,1],ba[:,2],color='#00E676',lw=2,ls='--')
        for s,e,lbl in HSEG:
            ax.scatter(*ga[e-1],c=GC,s=18,marker='o',alpha=0.8)
            ax.scatter(*ba[e-1],c='#00E676',s=18,marker='s',alpha=0.8)
        se = seg_errs(ba,ga,HSEG); es = ' | '.join([f'{l}:{np.mean(e):.1f}m' for l,e in se])
        ax.set_title(f'#{idx+1} {INTENT_4[intent]} minFDE={min_fde:.1f}m\n{es}',fontsize=6.5,family='monospace')
        ax.xaxis.pane.fill=False; ax.yaxis.pane.fill=False; ax.zaxis.pane.fill=False
        ax.xaxis.pane.set_edgecolor('w'); ax.yaxis.pane.set_edgecolor('w'); ax.zaxis.pane.set_edgecolor('w')
        ax.tick_params(labelsize=5)
    le3 = [Line2D([0],[0],color=HC,lw=1.5,label='History(20s)'),
           Line2D([0],[0],color=GC,lw=2.5,label='Ground Truth'),
           Line2D([0],[0],color='#00E676',lw=2,ls='--',label='Multi K=5 Best')]
    fig.legend(handles=le3,loc='lower center',ncol=3,fontsize=8)
    p = OUT/'high_12_best_trajectory_3d.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

def chart_high_best_xy(samples):
    fig,axs = plt.subplots(4,3,figsize=(24,26))
    fig.suptitle('HIGH Model — Best Confidence Trajectory (XY View)\n'
                 'Dark=History  Green=GT  BrightGreen--=Pred  o=GT marker  s=Pred marker',
                 fontsize=12,fontweight='bold')
    for idx,(hist,target,all_preds,best_pred,conf,intent,min_fde) in enumerate(samples):
        ax = axs[idx//3,idx%3]; lp = hist[-1,:3]; hp = hist[:,:3]
        ga = target[:,:3]; ba = lp+best_pred[:,:3]

        # Compute auto-limits with 10% padding so trajectory fills the subplot
        all_x = np.concatenate([hp[:,0], ga[:,0], ba[:,0]])
        all_y = np.concatenate([hp[:,1], ga[:,1], ba[:,1]])
        x_range = all_x.max() - all_x.min(); y_range = all_y.max() - all_y.min()
        cx, cy = (all_x.min()+all_x.max())/2, (all_y.min()+all_y.max())/2
        # Make square-ish bounds using the larger range
        half = max(x_range, y_range)*0.55 + 5.0  # +5m minimum
        ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)

        ax.plot(hp[:,0],hp[:,1],color=HC,lw=2,alpha=0.7,label='History')
        ax.plot(ga[:,0],ga[:,1],color=GC,lw=3,label='GT')
        ax.plot(ba[:,0],ba[:,1],color='#00E676',lw=2.5,ls='--',label='Pred')
        for s,e,lbl in HSEG:
            ax.plot(ga[e-1,0],ga[e-1,1],'o',color=GC,ms=8,alpha=0.9,zorder=5)
            ax.plot(ba[e-1,0],ba[e-1,1],'s',color='#00E676',ms=8,alpha=0.9,zorder=5)
        se = seg_errs(ba,ga,HSEG); es = ' | '.join([f'{l}:{np.mean(e):.1f}m' for l,e in se])
        ax.set_title(f'#{idx+1} {INTENT_4[intent]} minFDE={min_fde:.1f}m\n{es}',
                     fontsize=7,family='monospace')
        ax.grid(True,alpha=0.3)
        if idx==0: ax.legend(fontsize=7,loc='best')
    p = OUT/'high_12_best_trajectory_xy.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

def chart_high_error(samples):
    fig,ax = plt.subplots(figsize=(18,6))
    fig.suptitle('HIGH Model — Per-5s Prediction Error (12 Samples)',fontsize=12,fontweight='bold')
    sl = [s[2] for s in HSEG]; x = np.arange(len(sl)); w = 0.7/len(samples)
    for idx,(hist,target,all_preds,best_pred,conf,intent,min_fde) in enumerate(samples):
        lp = hist[-1,:3]; ba = lp+best_pred[:,:3]; ga = target[:,:3]
        se = seg_errs(ba,ga,HSEG); means = [np.mean(e) for _,e in se]
        off = (idx-len(samples)/2+0.5)*w
        ax.bar(x+off,means,w,alpha=0.85,label=f'#{idx+1} {INTENT_4[intent]}')
        for xi,m in zip(x,means): ax.text(xi+off,m+0.05,f'{m:.1f}',ha='center',fontsize=4.5,rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(sl,fontsize=9)
    ax.set_ylabel('Mean L2 Error (m)'); ax.legend(fontsize=5,ncol=4,loc='upper left'); ax.grid(True,alpha=0.3,axis='y')
    p = OUT/'high_12_persecond_error.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

def chart_high_adapter(samples):
    """Single vs Multi-head comparison."""
    n = min(6,len(samples))
    fig,axs = plt.subplots(2,3,figsize=(22,14))
    fig.suptitle('HIGH Model — Single vs Multi-Hypothesis Comparison (XY View)',
                 fontsize=13,fontweight='bold')
    sm = load_high_single()
    for idx in range(n):
        ax = axs[idx//3,idx%3]
        hist,target,all_preds,best_pred,conf,intent,min_fde = samples[idx]
        lp = hist[-1,:3]; hp = hist[:,:3]; ga = target[:,:3]; ba = lp+best_pred[:,:3]
        h_t = torch.from_numpy(hist).unsqueeze(0).to(DEVICE)
        sp = pred_high_single(sm,h_t); sa = lp+sp[0].cpu().numpy()[:,:3]
        # Auto-zoom to fill subplot
        all_x = np.concatenate([hp[:,0], ga[:,0], sa[:,0], ba[:,0]])
        all_y = np.concatenate([hp[:,1], ga[:,1], sa[:,1], ba[:,1]])
        cx = (all_x.min()+all_x.max())/2; cy = (all_y.min()+all_y.max())/2
        half = max(all_x.max()-all_x.min(), all_y.max()-all_y.min())*0.55 + 5.0
        ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)

        ax.plot(hp[:,0],hp[:,1],color=HC,lw=2,alpha=0.7,label='History')
        ax.plot(ga[:,0],ga[:,1],color=GC,lw=3,label='Ground Truth')
        ax.plot(sa[:,0],sa[:,1],color='#FF6D00',lw=2,ls='--',label='Single')
        ax.plot(ba[:,0],ba[:,1],color='#2979FF',lw=2.5,label='Multi K=5')
        for s,e,lbl in HSEG:
            ax.plot(ga[e-1,0],ga[e-1,1],'o',color=GC,ms=7,alpha=0.9,zorder=5)
            ax.plot(sa[e-1,0],sa[e-1,1],'s',color='#FF6D00',ms=7,alpha=0.9,zorder=5)
            ax.plot(ba[e-1,0],ba[e-1,1],'D',color='#2979FF',ms=7,alpha=0.9,zorder=5)
        sf = np.linalg.norm(sa[-1]-ga[-1]); mf = np.linalg.norm(ba[-1]-ga[-1])
        ax.set_title(f'#{idx+1} {INTENT_4[intent]} | FDE: {sf:.1f}->{mf:.1f}m ({(sf-mf)/sf*100:+.0f}%)',fontsize=7)
        ax.grid(True,alpha=0.3); ax.legend(fontsize=7,loc='best')
    for idx in range(n,6): axs[idx//3,idx%3].axis('off')
    p = OUT/'high_6_adapter_comparison.png'; fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f'  Saved: {p.name}')

# ── Main ───────────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    np.random.seed(42); torch.manual_seed(42)
    ls = collect_low(80)
    if ls: chart_low_3d(ls); chart_low_xy(ls); chart_low_error(ls)
    hs = collect_high(80)
    if hs: chart_high_multihyp_3d(hs); chart_high_best_3d(hs); chart_high_best_xy(hs); chart_high_error(hs); chart_high_adapter(hs)
    print(f'\n{"="*60}\nAll trajectory charts generated!')
    for f in sorted(OUT.glob('low_12_*')): print(f'  {f.name}')
    for f in sorted(OUT.glob('high_12_*')): print(f'  {f.name}')
    for f in sorted(OUT.glob('high_6_*')): print(f'  {f.name}')
    print(f'{"="*60}')

if __name__=='__main__': main()
