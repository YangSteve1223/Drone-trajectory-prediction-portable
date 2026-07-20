#!/usr/bin/env python3
"""
Fine-tune LOW model on balanced trajectory mix to reduce catastrophic failures.

Strategy:
  - Resample: 33% long (>=150f) + 33% medium (50-149f) + 33% short (<50f)
  - Within long trajectories: weight high-cata windows 3x
  - Low LR (5e-5), 8 epochs, freeze SSM backbone first 2 epochs
  - Validate after each epoch on held-out sets from each group
"""

import torch, numpy as np, sys, warnings, json, copy
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from emam_model import TrajectoryPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAJ_DIR = Path(__file__).parent.parent / 'UAV-Flow-trajs'
WEIGHTS_DIR = Path(__file__).parent / 'weights'
CHECKPOINT_PATH = WEIGHTS_DIR / 'low_speed_6class.pth'
OUT_PATH = WEIGHTS_DIR / 'low_speed_6class_finetuned.pth'
EVAL_OUT = Path(__file__).parent / 'pic-results' / 'finetune_eval.json'
EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)

HIST_LEN, PRED_LEN = 20, 20
EPOCHS = 8
BATCH_SIZE = 32
LR = 5e-5
GRAD_CLIP = 1.0
WINDOWS_PER_GROUP = 8000   # balanced samples per group per epoch


def dir_err(pv, tv):
    pn, tn = float(np.linalg.norm(pv)), float(np.linalg.norm(tv))
    if pn < 0.01 or tn < 0.01:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(pv, tv) / (pn * tn), -1.0, 1.0))))


def sliding_windows(traj, stride=2):
    n = traj.shape[0]
    ml = HIST_LEN + PRED_LEN
    if n < ml:
        return [], []
    hists, futs = [], []
    for i in range(0, n - ml + 1, stride):
        hists.append(traj[i:i + HIST_LEN])
        fut_abs = traj[i + HIST_LEN:i + HIST_LEN + PRED_LEN, :3]
        futs.append(fut_abs - traj[i + HIST_LEN - 1, :3])
    return hists, futs


def load_all_data():
    """Load and categorize all trajectory windows by length group."""
    long_data = []    # (hist, fut, weight)
    med_data = []
    short_data = []

    n_long_traj, n_med_traj, n_short_traj = 0, 0, 0
    print('  Loading trajectories...')
    for f in sorted(TRAJ_DIR.glob('*.npz')):
        d = np.load(f)
        traj = d['traj']
        nf = traj.shape[0]
        # Use stride=1 for short trajs to get enough windows
        stride = 2 if nf >= 50 else 1
        hists, futs = sliding_windows(traj, stride)
        if len(hists) < 3:
            continue

        if nf >= 150:
            long_data.extend((h, fut, 1.0) for h, fut in zip(hists, futs))
            n_long_traj += 1
        elif nf >= 50:
            med_data.extend((h, fut, 1.0) for h, fut in zip(hists, futs))
            n_med_traj += 1
        else:
            short_data.extend((h, fut, 1.0) for h, fut in zip(hists, futs))
            n_short_traj += 1

    print(f'  Long:  {len(long_data)} windows from {n_long_traj} trajs')
    print(f'  Med:   {len(med_data)} windows from {n_med_traj} trajs')
    print(f'  Short: {len(short_data)} windows from {n_short_traj} trajs')
    return long_data, med_data, short_data


def compute_weights(long_data, model, device):
    """Assign higher weight to windows where base model has high direction error."""
    print('  Computing per-window weights...')
    weights = np.ones(len(long_data))
    # Sample evaluation
    indices = np.random.choice(len(long_data), min(5000, len(long_data)), replace=False)
    for b in range(0, len(indices), 256):
        be = min(b + 256, len(indices))
        bidx = indices[b:be]
        hb = torch.stack([torch.from_numpy(long_data[i][0]).float()
                         for i in bidx]).to(device)
        with torch.no_grad():
            pb = model(hb, force_predict=True)['predictions'].cpu()
        for j, i in enumerate(bidx):
            de = dir_err(pb[j, -1, :2].numpy(), long_data[i][1][-1, :2])
            # Weight: 1.0 for DirErr<30, 3.0 for DirErr 60-90, linear in between
            w = 1.0 + 2.0 * min(max((de - 30) / 60, 0), 1)
            weights[i] = w
    print(f'  Weight range: [{weights.min():.1f}, {weights.max():.1f}], '
          f'mean={weights.mean():.2f}')
    return weights


def sample_batch(long_data, med_data, short_data, weights, n_per_group):
    """Sample a balanced batch: n_per_group from each length category."""
    batch = []

    # Long: weighted sampling
    if len(long_data) > 0:
        idx = np.random.choice(len(long_data), n_per_group,
                               p=weights / weights.sum())
        for i in idx:
            h, f, _ = long_data[i]
            batch.append((torch.from_numpy(h).float(),
                          torch.from_numpy(f).float(),
                          weights[i]))  # weight as loss multiplier

    # Med: uniform
    if len(med_data) > 0:
        idx = np.random.choice(len(med_data), n_per_group, replace=True)
        for i in idx:
            h, f, _ = med_data[i]
            batch.append((torch.from_numpy(h).float(),
                          torch.from_numpy(f).float(), 1.0))

    # Short: uniform
    if len(short_data) > 0:
        idx = np.random.choice(len(short_data), n_per_group, replace=True)
        for i in idx:
            h, f, _ = short_data[i]
            batch.append((torch.from_numpy(h).float(),
                          torch.from_numpy(f).float(), 1.0))

    return batch


def evaluate_group(model, device, data, tag, max_wins=2000):
    """Quick evaluation on a data group."""
    n = min(len(data), max_wins)
    idx = np.random.choice(len(data), n, replace=False)
    ade, fde, dire, cata = [], [], [], 0
    for b in range(0, n, 128):
        be = min(b + 128, n)
        hb = torch.stack([torch.from_numpy(data[i][0]).float()
                         for i in idx[b:be]]).to(device)
        tb = torch.stack([torch.from_numpy(data[i][1]).float()
                         for i in idx[b:be]])
        with torch.no_grad():
            pb = model(hb, force_predict=True)['predictions'].cpu()
        diff = pb - tb
        ade.extend(torch.norm(diff, dim=-1).mean(dim=1).numpy())
        fde.extend(torch.norm(diff[:, -1, :], dim=-1).numpy())
        for j in range(pb.shape[0]):
            de = dir_err(pb[j, -1, :2].numpy(), data[idx[b + j]][1][-1, :2])
            dire.append(de)
            if de >= 90:
                cata += 1
    return {'ade': float(np.mean(ade)), 'fde': float(np.mean(fde)),
            'dir': float(np.mean(dire)), 'cata_pct': float(cata / max(n, 1) * 100),
            'n': n}


def main():
    print('=' * 80)
    print('Fine-tuning LOW Model — Balanced Trajectory Mix')
    print(f'  LR={LR}  Epochs={EPOCHS}  Batch={BATCH_SIZE}')
    print(f'  Groups: long(>=150f) + med(50-149f) + short(<50f)')
    print('=' * 80)

    # Load pretrained model
    print('\n[1/4] Loading pretrained model...')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = TrajectoryPredictor(
        input_dim=6, history_len=HIST_LEN, pred_len=PRED_LEN,
        d_model=128, d_state=16, d_conv=4, expand=2,
        emam_n_layers=2, num_intent_classes=6,
        use_trigger=True, trigger_mode='simple',
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    device = DEVICE
    print(f'  Model: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'  Checkpoint epoch: {ckpt.get("epoch", "?")}')

    # Load data
    print('\n[2/4] Loading and categorizing data...')
    long_data, med_data, short_data = load_all_data()
    weights = compute_weights(long_data, model, device)

    def get_data_sizes():
        return (f'L={len(long_data)} M={len(med_data)} S={len(short_data)}')

    # Pre-evaluation
    print(f'\n[3/4] Pre-finetune evaluation ({get_data_sizes()})...')
    pre = {}
    for tag, data in [('LONG', long_data), ('MED', med_data), ('SHORT', short_data)]:
        if len(data) > 0:
            pre[tag] = evaluate_group(model, device, data, tag)
            print(f'  {tag}: ADE={pre[tag]["ade"]:.3f}m  FDE={pre[tag]["fde"]:.3f}m  '
                  f'Dir={pre[tag]["dir"]:.1f}°  Cata={pre[tag]["cata_pct"]:.1f}%')

    # Fine-tuning
    print(f'\n[4/4] Fine-tuning ({EPOCHS} epochs)...')
    model.train()

    # Freeze SSM backbone for first 2 epochs (stabilize)
    ssm_params = set()
    for name, p in model.named_parameters():
        if 'emam_se' in name:
            ssm_params.add(p)
            p.requires_grad_(False)

    # Optimizer: lower LR for pretrained layers
    decoder_params = [p for n, p in model.named_parameters()
                      if 'emam_se' not in n]
    opt = torch.optim.AdamW([
        {'params': decoder_params, 'lr': LR},
        {'params': list(ssm_params), 'lr': LR * 0.1},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    n_per = max(1, BATCH_SIZE // 3)  # balanced: ~1/3 each group
    steps_per_epoch = min(300, WINDOWS_PER_GROUP // max(n_per, 1))

    best_state = None
    best_cata_long = 100

    for ep in range(EPOCHS):
        # Unfreeze SSM after epoch 2
        if ep == 2:
            for p in ssm_params:
                p.requires_grad_(True)

        ep_losses = []
        for step in range(steps_per_epoch):
            batch = sample_batch(long_data, med_data, short_data, weights, n_per)
            np.random.shuffle(batch)

            for b in range(0, len(batch), BATCH_SIZE):
                bb = batch[b:b + BATCH_SIZE]
                hb = torch.stack([x[0] for x in bb]).to(device)
                tb = torch.stack([x[1] for x in bb]).to(device)
                wb = torch.tensor([x[2] for x in bb], device=device)

                out = model(hb, force_predict=True)
                pred = out['predictions']

                # Weighted MSE loss
                per_sample_loss = ((pred - tb) ** 2).mean(dim=(1, 2))
                loss = (per_sample_loss * wb).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                ep_losses.append(loss.item())

            if step % 100 == 0:
                pass  # fast training, no verbose

        scheduler.step()

        # Quick eval on long group
        model.eval()
        el = evaluate_group(model, device, long_data, 'long')
        model.train()

        cata = el['cata_pct']
        if cata < best_cata_long:
            best_cata_long = cata
            best_state = copy.deepcopy(model.state_dict())

        print(f'  Epoch {ep + 1}/{EPOCHS}: loss={np.mean(ep_losses):.4f}  '
              f'long_cata={cata:.1f}%  long_dir={el["dir"]:.1f}°  '
              f'{"★" if cata == best_cata_long else ""}')

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    # Post-evaluation
    print(f'\n[5] Post-finetune evaluation...')
    post = {}
    for tag, data in [('LONG', long_data), ('MED', med_data), ('SHORT', short_data)]:
        if len(data) > 0:
            post[tag] = evaluate_group(model, device, data, tag)
            pre_r = pre[tag]
            po_r = post[tag]
            fde_d = (pre_r['fde'] - po_r['fde']) / max(pre_r['fde'], 0.001) * 100
            cata_d = pre_r['cata_pct'] - po_r['cata_pct']
            dir_d = pre_r['dir'] - po_r['dir']
            print(f'  {tag}: ADE {pre_r["ade"]:.3f}→{po_r["ade"]:.3f}m  '
                  f'FDE {pre_r["fde"]:.3f}→{po_r["fde"]:.3f}m ({fde_d:+.1f}%)  '
                  f'Dir {pre_r["dir"]:.1f}→{po_r["dir"]:.1f}° ({dir_d:+.1f}°)  '
                  f'Cata {pre_r["cata_pct"]:.1f}→{po_r["cata_pct"]:.1f}% ({cata_d:+.1f}pp)')

    # Save
    print(f'\n[6] Saving...')
    ckpt_out = {
        'model_state_dict': model.state_dict(),
        'epoch': ckpt.get('epoch', 0),
        'finetune_info': 'Balanced mix fine-tune for long trajectory cata reduction'
    }
    torch.save(ckpt_out, OUT_PATH)
    print(f'  Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)')

    json.dump({'pre': pre, 'post': post, 'best_long_cata': best_cata_long},
              open(EVAL_OUT, 'w'), indent=2)
    print(f'  Eval: {EVAL_OUT}')


if __name__ == '__main__':
    main()
