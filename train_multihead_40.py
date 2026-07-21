#!/usr/bin/env python3
"""
Multi-hypothesis (K heads, WTA) decoder integrated with the 40-FRAME LOW model.

Loads low_speed_6class_40frame.pth, replaces NeuralDecoder with MultiHeadNeuralDecoder,
trains the K heads with Winner-Takes-All loss on 40-frame adaptive windows from raw
UAV-Flow trajectories, and compares single-hypothesis vs minADE_K/minFDE_K.

Output: weights/low_multihead_K{K}_40frame.pth
"""

import torch, numpy as np, sys, warnings, json, argparse
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
WEIGHT_DIR = Path(__file__).parent / 'weights'
OUT_DIR = Path(__file__).parent / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 40, 20
DT = 0.2
TOTAL_WINDOWS = 30000
TRAIN_SPLIT = 0.85
SHORT_MAX, MEDIUM_MAX = 80, 150
SCALE_POS = 100.0


def make_adaptive_windows(traj, hist_len=40):
    n = traj.shape[0]
    if n < 60: stride = 1
    elif n < 120: stride = 2
    else: stride = max(2, (n - PRED_LEN) // hist_len)
    ml = hist_len * stride + PRED_LEN
    if n < ml: return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, max(1, stride // 2)):
        indices = np.arange(i, i + hist_len * stride, stride)[:hist_len]
        hists.append(traj[indices].copy())
        fut_start = i + hist_len * stride
        fut_abs = traj[fut_start:fut_start + PRED_LEN, :3]
        futs.append(fut_abs - traj[fut_start - 1, :3])
    return hists, futs


def collect_windows(total):
    all_trajs = []
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f); n = d['traj'].shape[0]
        all_trajs.append((d['traj'], n))
    short = [t for t, l in all_trajs if l < SHORT_MAX]
    medium = [t for t, l in all_trajs if SHORT_MAX <= l < MEDIUM_MAX]
    long_t = [t for t, l in all_trajs if l >= MEDIUM_MAX]
    np.random.seed(42)
    windows = []
    per = total // 3
    for group in [short, medium, long_t]:
        nc = 0; sh = list(group); np.random.shuffle(sh)
        for traj in sh:
            if nc >= per: break
            hs, fs = make_adaptive_windows(traj, HIST_LEN)
            for h, f in zip(hs, fs):
                windows.append((torch.from_numpy(h).float(), torch.from_numpy(f).float()))
                nc += 1
                if nc >= per: break
    np.random.shuffle(windows)
    return windows[:total]


def build_40frame_model(K):
    model = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE)
    ckpt = torch.load(WEIGHT_DIR / 'low_speed_6class_40frame.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    # Multi-head needs the manual-normalize path (forward_multi_head denormalizes by 100)
    model._norm_input = False
    def _normalize(hist):
        scale = hist.new_tensor([SCALE_POS, SCALE_POS, SCALE_POS, 10.0, 10.0, 10.0])
        return hist / scale.unsqueeze(0).unsqueeze(0)
    model._normalize = _normalize
    model.ua_pgd.replace_with_multi_head(K=K, noise_std=0.02)
    model.ua_pgd.neural_decoder.to(DEVICE)
    for name, param in model.named_parameters():
        param.requires_grad = 'neural_decoder' in name
    return model


def mh_forward(model, hb):
    """Run the multi-head decoder path. Returns all_predictions (K,B,P,3) in meters."""
    h_norm = model._normalize(hb)
    enc = model.emam_se(h_norm)
    dtp = model.ia_dtp(enc, historical_trajectory=h_norm)
    return model.ua_pgd.forward_multi_head(
        encoded_feat=enc, global_anchor=dtp['global_anchor'],
        historical_trajectory=h_norm, intent_weights=dtp['intent_weights'])

@torch.no_grad()
def evaluate(model, te_h, te_t, K, bs=128):
    model.eval()
    min_fde, min_ade, single_fde, single_ade = [], [], [], []
    for b in range(0, len(te_h), bs):
        hb = te_h[b:b + bs].to(DEVICE)
        tb = te_t[b:b + bs].to(DEVICE)               # (B,P,3) displacement targets (meters)
        out = mh_forward(model, hb)
        allp = out['all_predictions']                # (K,B,P,3) meters
        best = out['predictions']                    # (B,P,3) meters (highest-confidence)
        m = MultiHeadNeuralDecoder.compute_minade_fde(allp, tb)
        min_ade.extend(m['min_ade'].cpu().tolist())
        min_fde.extend(m['min_fde'].cpu().tolist())
        se = torch.norm(best - tb, dim=-1)
        single_ade.extend(se.mean(dim=1).cpu().tolist())
        single_fde.extend(se[:, -1].cpu().tolist())
    return (np.array(min_ade), np.array(min_fde),
            np.array(single_ade), np.array(single_fde))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--total_windows', type=int, default=TOTAL_WINDOWS)
    args = ap.parse_args()

    print('=' * 80)
    print(f'Multi-Hypothesis (K={args.K}, WTA) on 40-FRAME LOW model')
    print(f'  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}')
    print('=' * 80)

    print('\n[1/4] Collecting 40-frame adaptive windows...')
    windows = collect_windows(args.total_windows)
    n_tr = int(len(windows) * TRAIN_SPLIT)
    tr, te = windows[:n_tr], windows[n_tr:]
    tr_h = torch.stack([w[0] for w in tr]); tr_t = torch.stack([w[1] for w in tr])
    te_h = torch.stack([w[0] for w in te]); te_t = torch.stack([w[1] for w in te])
    print(f'  Train: {len(tr)}  Test: {len(te)}')

    print('\n[2/4] Building 40-frame model + K-head decoder...')
    model = build_40frame_model(args.K)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable (decoder only): {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

    # Baseline (before WTA training) — the freshly-perturbed K heads
    ba, bf, bsa, bsf = evaluate(model, te_h, te_t, args.K)
    print(f'  Baseline: single FDE={bsf.mean():.4f}m  minFDE_{args.K}={bf.mean():.4f}m')

    print(f'\n[3/4] Training K heads with WTA loss...')
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    idx = np.arange(len(tr)); best_fde = float('inf'); best_state = None
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(idx); losses = []
        for b in range(0, len(idx), args.batch_size):
            bi = idx[b:b + args.batch_size]
            hb = tr_h[bi].to(DEVICE); tb = tr_t[bi].to(DEVICE)
            opt.zero_grad()
            out = mh_forward(model, hb)
            preds_norm = out['all_predictions'] / SCALE_POS   # (K,B,P,3)
            target_norm = tb / SCALE_POS
            ld = MultiHeadNeuralDecoder.compute_wta_loss(
                preds_norm, out['all_logvars'], out['confidences'], target_norm)
            loss = ld['total_wta_loss']
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.ua_pgd.neural_decoder.parameters(), 5.0)
                opt.step(); losses.append(loss.item())
        sched.step()
        ma, mf, sa, sf = evaluate(model, te_h, te_t, args.K)
        if mf.mean() < best_fde:
            best_fde = mf.mean()
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.ua_pgd.neural_decoder.state_dict().items()}
        if ep % 2 == 0 or ep == args.epochs - 1:
            print(f'  Epoch {ep:2d}: loss={np.mean(losses):.4f}  '
                  f'single FDE={sf.mean():.4f}m  minFDE_{args.K}={mf.mean():.4f}m '
                  f'{"*" if mf.mean() == best_fde else ""}')

    if best_state is not None:
        model.ua_pgd.neural_decoder.load_state_dict(best_state)

    print(f'\n[4/4] Final evaluation...')
    ma, mf, sa, sf = evaluate(model, te_h, te_t, args.K)
    print(f'\n{"=" * 80}')
    print(f'MULTI-HYPOTHESIS on 40-FRAME LOW — RESULTS ({len(te)} test windows)')
    print(f'{"=" * 80}')
    print(f'  {"Metric":<20} {"Single-best":<16} {"minK (oracle)":<16} {"Change":<10}')
    print(f'  {"-" * 62}')
    print(f'  {"ADE":<20} {sa.mean():<16.4f} {ma.mean():<16.4f} {(sa.mean()-ma.mean())/max(sa.mean(),1e-6)*100:>+7.1f}%')
    print(f'  {"FDE":<20} {sf.mean():<16.4f} {mf.mean():<16.4f} {(sf.mean()-mf.mean())/max(sf.mean(),1e-6)*100:>+7.1f}%')
    print(f'  {"FDE P95":<20} {np.percentile(sf,95):<16.4f} {np.percentile(mf,95):<16.4f}')

    save_path = WEIGHT_DIR / f'low_multihead_K{args.K}_40frame.pth'
    torch.save({'multi_decoder_state': model.ua_pgd.neural_decoder.state_dict(),
                'K': args.K, 'base': 'low_speed_6class_40frame.pth',
                'min_fde': float(mf.mean()), 'single_fde': float(sf.mean())}, save_path)
    print(f'\n  Saved: {save_path}')

    summary = {'K': args.K, 'base': 'low_speed_6class_40frame.pth',
               'test_windows': len(te),
               'single_ade': float(sa.mean()), 'single_fde': float(sf.mean()),
               'min_ade': float(ma.mean()), 'min_fde': float(mf.mean()),
               'fde_gain_pct': float((sf.mean() - mf.mean()) / max(sf.mean(), 1e-6) * 100),
               'baseline_single_fde': float(bsf.mean()), 'baseline_min_fde': float(bf.mean())}
    json.dump(summary, open(OUT_DIR / f'multihead_40_K{args.K}.json', 'w'), indent=2)
    print(f'  Results saved: pic-results/multihead_40_K{args.K}.json')
    print('=' * 80)


if __name__ == '__main__':
    main()
