#!/usr/bin/env python3
"""Compare single LOW vs multi-hypothesis LOW on full test set."""
import torch, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from emam_model import TrajectoryPredictor
from emam_model.ua_pgd import MultiHeadNeuralDecoder
from utils.fast_data_loader import FastWindowDataset
from tqdm import tqdm

device = torch.device('cuda')
data_path = Path('../UAV-Flow-pure')

# Load single LOW
model_single = TrajectoryPredictor(
    input_dim=6, history_len=20, pred_len=20,
    d_model=128, d_state=16, d_conv=4, expand=2,
    emam_n_layers=2, num_intent_classes=6,
    use_trigger=True, trigger_mode='simple',
).to(device).eval()
ckpt = torch.load('weights/low_speed_6class.pth', map_location=device, weights_only=False)
model_single.load_state_dict(ckpt['model_state_dict'])

# Load multi-head LOW
model_mh = TrajectoryPredictor(
    input_dim=6, history_len=20, pred_len=20,
    d_model=128, d_state=16, d_conv=4, expand=2,
    emam_n_layers=2, num_intent_classes=6,
    use_trigger=True, trigger_mode='simple',
).to(device).eval()
ckpt = torch.load('weights/low_speed_6class.pth', map_location=device, weights_only=False)
model_mh.load_state_dict(ckpt['model_state_dict'])
model_mh.ua_pgd.replace_with_multi_head(K=5)
model_mh.ua_pgd.neural_decoder = model_mh.ua_pgd.neural_decoder.to(device)
mh_ckpt = torch.load('weights/low_multihead_K5.pth', map_location=device, weights_only=False)
model_mh.ua_pgd.neural_decoder.load_state_dict(mh_ckpt['multi_decoder_state'])

model_mh._norm_input = False
model_mh._get_scale_pos = lambda: 100.0

def _normalize(hist):
    scale = hist.new_tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
    return hist / scale.unsqueeze(0).unsqueeze(0)

model_mh._normalize = _normalize

# Load data
dataset = FastWindowDataset(str(data_path), split='test')
print(f'Test samples: {len(dataset)}')

intent_names = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESCEND', 'HOVER']

results = {
    'single': {'ade': [], 'fde': [], 'dir_err': [], 'per_intent': defaultdict(list)},
    'multi': {'ade': [], 'fde': [], 'dir_err': [], 'per_intent': defaultdict(list)},
}

for i in tqdm(range(len(dataset)), desc='Comparing'):
    hist, target, intent = dataset[i]
    hist_batch = hist.unsqueeze(0).to(device)
    target_batch = target.unsqueeze(0).to(device)
    intent_idx = intent.item()

    # Single model
    with torch.no_grad():
        out_s = model_single(hist_batch, force_predict=True)
    pred_s = out_s['predictions']
    lp = hist_batch[0, -1, :3]
    pa_s = lp + pred_s[0]
    ga = target_batch[0, :, :3]

    diff_s = pa_s - ga
    l2_s = torch.norm(diff_s, dim=1)
    results['single']['ade'].append(l2_s.mean().item())
    results['single']['fde'].append(l2_s[-1].item())

    dir_s = pa_s[-1] - pa_s[0]
    dir_t = ga[-1] - ga[0]
    cos_s = torch.nn.functional.cosine_similarity(dir_s.unsqueeze(0), dir_t.unsqueeze(0))
    results['single']['dir_err'].append(torch.acos(torch.clamp(cos_s, -1, 1)).item() * 180 / np.pi)
    if intent_idx < 6:
        results['single']['per_intent'][intent_names[intent_idx]].append(l2_s[-1].item())

    # Multi-head model
    with torch.no_grad():
        h_norm = model_mh._normalize(hist_batch)
        enc = model_mh.emam_se(h_norm)
        dtp_out = model_mh.ia_dtp(enc, historical_trajectory=h_norm)
        mh_out = model_mh.ua_pgd.forward_multi_head(
            encoded_feat=enc,
            global_anchor=dtp_out['global_anchor'],
            historical_trajectory=h_norm,
            intent_weights=dtp_out['intent_weights'],
        )

    all_preds = mh_out['all_predictions']  # (K, B, P, 3) = (5, 1, 20, 3)
    K = all_preds.shape[0]  # K is dim 0!
    best_fde = float('inf')
    best_ade = float('inf')
    best_k = 0
    for k in range(K):
        pa_k = lp + all_preds[k, 0]  # (P, 3) for hypothesis k
        diff_k = pa_k - ga
        l2_k = torch.norm(diff_k, dim=1)
        fde_k = l2_k[-1].item()
        if fde_k < best_fde:
            best_fde = fde_k
            best_ade = l2_k.mean().item()
            best_k = k

    results['multi']['ade'].append(best_ade)
    results['multi']['fde'].append(best_fde)

    # Direction error for best-FDE hypothesis
    pa_best = lp + all_preds[best_k, 0]
    dir_best = pa_best[-1] - pa_best[0]
    cos_best = torch.nn.functional.cosine_similarity(dir_best.unsqueeze(0), dir_t.unsqueeze(0))
    results['multi']['dir_err'].append(torch.acos(torch.clamp(cos_best, -1, 1)).item() * 180 / np.pi)
    if intent_idx < 6:
        results['multi']['per_intent'][intent_names[intent_idx]].append(best_fde)

# Print comparison
print()
print('=' * 70)
print('  LOW Model: Before vs After (Multi-Hypothesis K=5 + Z Correction)')
print('=' * 70)
print(f'  {"Metric":30s}  {"Single (Before)":>15s}  {"Multi K=5 (After)":>15s}  {"Change":>10s}')
print(f'  {"-"*70}')

for label, key, unit, lower_better in [
    ('ADE mean', 'ade', 'm', True),
    ('ADE median', 'ade', 'm', True),
    ('FDE mean', 'fde', 'm', True),
    ('FDE median', 'fde', 'm', True),
    ('FDE P95', 'fde', 'm', True),
    ('Direction err mean', 'dir_err', 'deg', True),
    ('Direction err median', 'dir_err', 'deg', True),
]:
    s_arr = np.array(results['single'][key])
    m_arr = np.array(results['multi'][key])
    if 'mean' in label:
        s_val = s_arr.mean()
        m_val = m_arr.mean()
    elif 'median' in label:
        s_val = np.median(s_arr)
        m_val = np.median(m_arr)
    elif 'P95' in label:
        s_val = np.percentile(s_arr, 95)
        m_val = np.percentile(m_arr, 95)

    if lower_better:
        change = (s_val - m_val) / (s_val + 1e-8) * 100
        arrow = '+' if change > 0 else ''
    else:
        change = (m_val - s_val) / (s_val + 1e-8) * 100
        arrow = '+' if change > 0 else ''
    print(f'  {label:30s}  {s_val:15.3f}{unit}  {m_val:15.3f}{unit}  {arrow}{change:+.1f}%')

# Catastrophic
s_cat = (np.array(results['single']['dir_err']) > 90).mean() * 100
m_cat = (np.array(results['multi']['dir_err']) > 90).mean() * 100
change_cat = (s_cat - m_cat) / (s_cat + 1e-8) * 100
arrow_cat = '+' if change_cat > 0 else ''
print(f'  {"Catastrophic (>90deg)":30s}  {s_cat:14.2f}%  {m_cat:14.2f}%  {arrow_cat}{change_cat:+.1f}%')

print()
print('  Per-intent FDE comparison:')
for name in intent_names:
    s_arr = np.array(results['single']['per_intent'].get(name, [0]))
    m_arr = np.array(results['multi']['per_intent'].get(name, [0]))
    if len(s_arr) > 0 and s_arr.mean() > 0:
        change = (s_arr.mean() - m_arr.mean()) / s_arr.mean() * 100
        print(f'    {name:12s}: {s_arr.mean():.3f}m -> {m_arr.mean():.3f}m ({change:+.1f}%)  n={len(s_arr)}')

# Save
np.savez('_low_comparison.npz',
         single_ade=results['single']['ade'], single_fde=results['single']['fde'],
         single_dir=results['single']['dir_err'],
         multi_ade=results['multi']['ade'], multi_fde=results['multi']['fde'],
         multi_dir=results['multi']['dir_err'])
print()
print('Results saved to _low_comparison.npz')
