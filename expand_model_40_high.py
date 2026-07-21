#!/usr/bin/env python3
"""
HIGH model 20->40 frame expansion (SimCruise, 4-class, 1Hz).

Mirror of expand_model_40.py but for the HIGH-speed cruise model:
  - Base: high_speed_4class.pth (num_intent_classes=4)
  - Data: raw trajectories from SimCruise/.../trajectories_merged.npz
          (positions, velocities, masks; variable length up to 839 frames, 1Hz)
  - HIGH window convention: each history window is CENTERED on its first frame
    (first-frame position -> origin), matching the pre-windowed SimCruise data.
  - Fair comparison: paired 20 vs 40 windows share the same anchor + future.

Output: weights/high_speed_4class_40frame.pth
"""

import torch, numpy as np, sys, warnings, json
import torch.nn.functional as F
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from emam_model import TrajectoryPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIM_DIR = Path(__file__).parent.parent / 'SimCruise'
WEIGHT_DIR = Path(__file__).parent / 'weights'
OUT_DIR = Path(__file__).parent / 'pic-results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_HIST_LEN = 40
PRED_LEN = 20
DT = 1.0                       # HIGH is 1 Hz
N_CLASSES = 4

TOTAL_TRAJS = 6000             # raw trajectories to sample windows from
MAX_WIN_PER_TRAJ = 6
TRAIN_SPLIT = 0.85
BATCH_SIZE = 32
EPOCHS = 15
LR_MAX, LR_MIN = 1e-4, 1e-6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

BETA_HUBER = 0.20
W_DIR, W_SMOOTH, W_JERK = 0.10, 0.15, 0.10
W_BOUNDARY = 0.30


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def load_raw_trajs(max_trajs):
    """Load raw SimCruise trajectories as list of (L,6) arrays [x,y,z,vx,vy,vz]."""
    merged = sorted(SIM_DIR.rglob('trajectories_merged.npz'))
    if not merged:
        raise FileNotFoundError('trajectories_merged.npz not found under SimCruise/')
    d = np.load(merged[0])
    pos, vel, mask = d['positions'], d['velocities'], d['masks']
    lengths = mask.sum(axis=1)
    # keep trajectories long enough for a 40-frame (stride 2) window + pred
    min_len = 39 * 2 + 1 + PRED_LEN
    idx = np.where(lengths >= min_len)[0]
    rng = np.random.RandomState(42); rng.shuffle(idx)
    idx = idx[:max_trajs]
    trajs = []
    for i in idx:
        L = int(lengths[i])
        p = pos[i, :L].astype(np.float32); v = vel[i, :L].astype(np.float32)
        trajs.append(np.concatenate([p, v], axis=1))   # (L, 6)
    return trajs


def make_paired_windows(traj, stride40=2, max_win=6):
    """Paired 20/40-frame windows, HIGH-centered (history minus first frame pos)."""
    n = traj.shape[0]
    span40 = 39 * stride40
    j_start = max(19, span40)
    anchors = list(range(j_start, n - PRED_LEN))
    if not anchors: return [], [], []
    # subsample anchors for coverage without explosion
    if len(anchors) > max_win:
        sel = np.linspace(0, len(anchors) - 1, max_win).astype(int)
        anchors = [anchors[s] for s in sel]
    h20s, h40s, futs = [], [], []
    for j in anchors:
        h20 = traj[j - 19:j + 1].copy()
        idx40 = np.arange(j - span40, j + 1, stride40)[:40]
        if len(idx40) < 40: continue
        h40 = traj[idx40].copy()
        fut_abs = traj[j + 1:j + 1 + PRED_LEN, :3]
        if fut_abs.shape[0] < PRED_LEN: continue
        fut = fut_abs - traj[j, :3]                    # future relative to anchor
        # HIGH centering: subtract each window's first-frame position from its positions
        h20 = h20.copy(); h40 = h40.copy()
        h20[:, :3] -= h20[0, :3]
        h40[:, :3] -= h40[0, :3]
        h20s.append(h20); h40s.append(h40); futs.append(fut)
    return h20s, h40s, futs


def compute_loss(pred, target, history):
    loss_huber = F.smooth_l1_loss(pred, target, beta=BETA_HUBER)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]; true_vel = target[:, 1:, :] - target[:, :-1, :]
    loss_dir = (1.0 - F.cosine_similarity(pred_vel, true_vel, dim=-1)).mean()
    pred_acc = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
    loss_smooth = (pred_acc ** 2).mean()
    pred_jerk = pred[:, 3:, :] - 3 * pred[:, 2:-1, :] + 3 * pred[:, 1:-2, :] - pred[:, :-3, :]
    loss_jerk = (pred_jerk ** 2).mean()
    hist_last_vel = history[:, -1, 3:6]
    pc = pred[:, 0, :]; ec = hist_last_vel * DT
    loss_boundary = ((pc - ec) ** 2).mean()
    return (loss_huber + W_DIR * loss_dir + W_SMOOTH * loss_smooth
            + W_JERK * loss_jerk + W_BOUNDARY * loss_boundary)

def build_model(hist_len):
    return TrajectoryPredictor(
        input_dim=6, history_len=hist_len, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=N_CLASSES,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE)


def eval_model(model, hists, futs_t, futs_np, bs=BATCH_SIZE):
    preds = []
    for b in range(0, len(hists), bs):
        hb = hists[b:b + bs].to(DEVICE)
        with torch.no_grad():
            preds.append(model(hb, force_predict=True)['predictions'].cpu())
    preds = torch.cat(preds, dim=0)
    fde = torch.norm(preds[:, -1, :] - futs_t[:, -1, :], dim=-1).mean().item()
    ade = torch.norm(preds - futs_t, dim=-1).mean(dim=1).mean().item()
    de = np.array([dir_err(preds[i, -1, :2].numpy(), futs_np[i][-1, :2]) for i in range(len(hists))])
    dmean = float(np.mean(de)); cata = float(np.sum(de >= 90) / len(de) * 100)
    gap = float(np.linalg.norm(preds[:, 0, :].numpy(), axis=1).mean())
    return dict(ade=ade, fde=fde, dir=dmean, cata=cata, gap=gap)


def main():
    print('=' * 80)
    print('HIGH Model Expansion: 20->40 frames (SimCruise, 4-class, 1Hz)')
    print(f'  New history_len: {NEW_HIST_LEN} (was 20)  Pred: {PRED_LEN}')
    print(f'  Trajs: {TOTAL_TRAJS}  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}')
    print('=' * 80)

    # ── Load original 20-frame HIGH model ──
    print('\n[1/5] Loading original 20-frame HIGH model...')
    model_20 = build_model(20).eval()
    ckpt_20 = torch.load(WEIGHT_DIR / 'high_speed_4class.pth', map_location=DEVICE)
    model_20.load_state_dict(ckpt_20['model_state_dict'])
    print(f'  Loaded. Params: {sum(p.numel() for p in model_20.parameters()):,}')

    # ── Create 40-frame model (weight transfer) ──
    print('\n[2/5] Creating 40-frame HIGH model (weight transfer)...')
    model_40 = build_model(NEW_HIST_LEN)
    state_20 = {k: v for k, v in ckpt_20['model_state_dict'].items() if k != 'intent_history'}
    missing, unexpected = model_40.load_state_dict(state_20, strict=False)
    print(f'  Transferred. Missing: {len(missing)} (intent_history only)  Unexpected: {len(unexpected)}')

    # ── Collect paired windows (traj-level split) ──
    print(f'\n[3/5] Loading raw trajectories + building paired windows...')
    trajs = load_raw_trajs(TOTAL_TRAJS)
    print(f'  Loaded {len(trajs)} raw trajectories (>= {39*2+1+PRED_LEN} frames)')
    rng = np.random.RandomState(7); order = np.arange(len(trajs)); rng.shuffle(order)
    n_test_traj = max(50, len(trajs) // 6)
    test_ids = set(order[:n_test_traj].tolist())

    tr_h40, tr_fut = [], []
    te_h20, te_h40, te_fut = [], [], []
    for ti, traj in enumerate(trajs):
        h20s, h40s, futs = make_paired_windows(traj, stride40=2, max_win=MAX_WIN_PER_TRAJ)
        if ti in test_ids:
            te_h20.extend(h20s); te_h40.extend(h40s); te_fut.extend(futs)
        else:
            tr_h40.extend(h40s); tr_fut.extend(futs)
    print(f'  Train (40f): {len(tr_h40)} windows | Test paired: {len(te_h20)} windows')

    tr_h40 = torch.from_numpy(np.stack(tr_h40)).float()
    tr_fut_t = torch.from_numpy(np.stack(tr_fut)).float()
    te_h20_t = torch.from_numpy(np.stack(te_h20)).float()
    te_h40_t = torch.from_numpy(np.stack(te_h40)).float()
    te_fut_t = torch.from_numpy(np.stack(te_fut)).float()
    te_fut_np = te_fut_t.numpy()

    # ── Baseline: 20-frame HIGH on the paired test set ──
    print('\n[4/5] Baseline evaluation (20-frame HIGH)...')
    b20 = eval_model(model_20, te_h20_t, te_fut_t, te_fut_np)
    print(f'  20-frame: ADE={b20["ade"]:.3f}  FDE={b20["fde"]:.3f}  Dir={b20["dir"]:.1f}deg  '
          f'Cata={b20["cata"]:.1f}%  Gap={b20["gap"]:.3f}')

    # ── Fine-tune 40-frame HIGH (freeze encoder, train decoder+gate) ──
    print(f'\n[5/5] Fine-tuning 40-frame HIGH ({EPOCHS} epochs)...')
    for name, param in model_40.named_parameters():
        if 'emam_se' in name: param.requires_grad_(False)
    trainable = [p for p in model_40.parameters() if p.requires_grad]
    print(f'  Trainable: {sum(p.numel() for p in trainable):,} / '
          f'{sum(p.numel() for p in model_40.parameters()):,} (froze emam_se)')

    opt = torch.optim.AdamW(trainable, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    n = len(tr_h40); idx = np.arange(n)
    best_fde = float('inf'); best_state = None
    for ep in range(EPOCHS):
        model_40.train(); np.random.shuffle(idx); losses = []
        for b in range(0, n, BATCH_SIZE):
            bi = idx[b:b + BATCH_SIZE]
            hb = tr_h40[bi].to(DEVICE); tb = tr_fut_t[bi].to(DEVICE)
            opt.zero_grad()
            pred = model_40(hb, force_predict=True)['predictions']
            loss = compute_loss(pred, tb, hb)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP); opt.step()
                losses.append(loss.item())
        sched.step()
        model_40.eval()
        vfde = torch.norm(
            torch.cat([model_40(te_h40_t[b:b+BATCH_SIZE].to(DEVICE), force_predict=True)['predictions'].cpu()
                       for b in range(0, len(te_h40_t), BATCH_SIZE)], dim=0)[:, -1, :]
            - te_fut_t[:, -1, :], dim=-1).mean().item()
        if vfde < best_fde:
            best_fde = vfde
            best_state = {k: v.detach().cpu().clone() for k, v in model_40.state_dict().items()}
        if ep % 3 == 0 or ep == EPOCHS - 1:
            print(f'  Epoch {ep:2d}: loss={np.mean(losses):.4f}  val_FDE={vfde:.3f}  lr={sched.get_last_lr()[0]:.6f}')

    if best_state is not None:
        model_40.load_state_dict(best_state)
    model_40.eval()

    # ── Final 40-frame eval on paired test set ──
    f40 = eval_model(model_40, te_h40_t, te_fut_t, te_fut_np)

    print(f'\n{"=" * 80}')
    print(f'HIGH Expansion 20->40 RESULTS (paired test set, {len(te_h20)} windows)')
    print(f'{"=" * 80}')
    print(f'  {"Metric":<16} {"20-frame":<14} {"40-frame":<14} {"Change":<10}')
    print(f'  {"-" * 54}')
    def chg(a, b, denom=None):
        d = denom if denom else max(a, 1e-6); return (a - b) / d * 100
    print(f'  {"ADE":<16} {b20["ade"]:<14.3f} {f40["ade"]:<14.3f} {chg(b20["ade"], f40["ade"]):>+7.1f}%')
    print(f'  {"FDE":<16} {b20["fde"]:<14.3f} {f40["fde"]:<14.3f} {chg(b20["fde"], f40["fde"]):>+7.1f}%')
    print(f'  {"Direction(deg)":<16} {b20["dir"]:<14.1f} {f40["dir"]:<14.1f} {chg(b20["dir"], f40["dir"], max(b20["dir"],0.1)):>+7.1f}%')
    print(f'  {"Cata(>90)%":<16} {b20["cata"]:<14.1f} {f40["cata"]:<14.1f}')
    print(f'  {"Boundary Gap":<16} {b20["gap"]:<14.3f} {f40["gap"]:<14.3f}')

    save_path = WEIGHT_DIR / 'high_speed_4class_40frame.pth'
    torch.save({'model_state_dict': best_state, 'history_len': NEW_HIST_LEN,
                'base_fde': b20['fde'], 'new_fde': f40['fde']}, save_path)
    print(f'\n  Saved: {save_path}')

    summary = {'base_model': 'high_speed_4class.pth', 'expanded': str(save_path),
               'train_windows': len(tr_h40), 'test_windows': len(te_h20),
               'b20': b20, 'f40': f40,
               'fde_gain_pct': float(chg(b20['fde'], f40['fde']))}
    json.dump(summary, open(OUT_DIR / 'high_expand_40.json', 'w'), indent=2)
    print(f'  Results saved: pic-results/high_expand_40.json')
    print('=' * 80)


if __name__ == '__main__':
    main()
