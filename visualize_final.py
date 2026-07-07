#!/usr/bin/env python3
"""
Final comprehensive visualization suite for group meeting.
Generates 8 high-quality scientific charts with clean layout:
  01_error_heatmap.png       — Per-step error heatmap (samples × timesteps)
  02_intent_confusion.png     — Intent confusion matrices (LOW 6×6 + HIGH 4×4)
  03_error_by_speed.png       — Error vs speed scatter + binned trend
  04_error_by_intent.png      — Per-intent error bar chart + error-over-time curves
  05_uncertainty_heatmap.png  — Model uncertainty (logvar) over prediction horizon
  06_adapter_comparison.png   — HIGH model: before vs after Context Adapter
  07_trajectory_grid.png      — 12-sample trajectory overlay grid
  08_summary_table.png        — Compact summary metrics table
"""

import torch, numpy as np, sys
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predictor import DronePredictor
from utils.fast_data_loader import FastWindowDataset
from context_adapter import ContextAdapterV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.dpi': 150,
    'figure.constrained_layout.use': True,  # prevent text overlap
})

INTENT_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
INTENT_4 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'DESCEND']
COLORS_6 = ['#2196F3','#FF9800','#F44336','#4CAF50','#9C27B0','#607D8B']
COLORS_4 = ['#2196F3','#FF9800','#F44336','#9C27B0']
DT_LOW, DT_HIGH = 0.2, 1.0
N_SAMPLES = 5000  # subset for heatmap visualization
BATCH_SIZE = 512
OUT_DIR = Path(__file__).parent / 'pic-results'


def collect_samples(ds, model, model_label, device, n_max=N_SAMPLES):
    """Collect predictions and targets for a subset of samples."""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_data = {'preds': [], 'targets': [], 'intents': [], 'speeds': [],
                'uncertainties': [], 'step_errs': [], 'intent_probs': [], 'hist': []}
    collected = 0

    for hist_batch, target_batch, intent_batch in loader:
        if collected >= n_max: break
        hist_batch = hist_batch.to(device)
        B = hist_batch.shape[0]
        take = min(B, n_max - collected)

        with torch.no_grad():
            out = model(hist_batch[:take], force_predict=True)

        preds = out['predictions'].cpu()
        last_pos = hist_batch[:take, -1:, :3].cpu()
        target_rel = target_batch[:take, :, :3] - last_pos

        # Z correction for HIGH (compute on CPU after preds are already on CPU)
        if model_label == 'HIGH':
            intent_prob = torch.softmax(out['intent_logits'][:take].cpu(), dim=-1)
            dp = intent_prob[:, 3]
            dampen = torch.ones(take)
            strong = dp < 0.05; weak = (dp >= 0.05) & (dp < 0.20)
            if strong.any(): dampen[strong] = 0.05
            if weak.any():
                t_val = (dp[weak] - 0.05) / 0.15
                dampen[weak] = 0.30 + 0.70 * t_val
            apply = dampen < 1.0
            if apply.any():
                preds[apply, :, 2] *= dampen[apply].view(-1, 1)

        step_err = torch.norm(preds - target_rel, dim=-1)

        all_data['preds'].append(preds)
        all_data['targets'].append(target_rel)
        all_data['intents'].append(intent_batch[:take])
        all_data['speeds'].append(DronePredictor.compute_speed(hist_batch[:take]).cpu())
        all_data['uncertainties'].append(out['uncertainty'][:take].cpu())
        all_data['step_errs'].append(step_err)
        all_data['intent_probs'].append(torch.softmax(out['intent_logits'][:take], dim=-1).cpu())
        all_data['hist'].append(hist_batch[:take, :, :3].cpu())

        collected += take
        if collected % 2000 == 0:
            print(f'    {collected}/{n_max}...')

    return {k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
            for k, v in all_data.items()}


# ================================================================
# Chart 1: Per-step error heatmap
# ================================================================
def chart_error_heatmap(data_low, data_high):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Per-Step Prediction Error Heatmap', fontsize=13, fontweight='bold', y=1.01)

    for ax, data, label, dt in [
        (axes[0], data_low, 'LOW (UAV-Flow, 5Hz)', DT_LOW),
        (axes[1], data_high, 'HIGH (SimCruise, 1Hz)', DT_HIGH),
    ]:
        errs = data['step_errs'].numpy()  # (N, 20)
        # Sort by endpoint error
        order = np.argsort(errs[:, -1])
        errs_sorted = errs[order]
        # Take subset for cleaner visualization
        n_show = min(200, len(errs_sorted))
        indices = np.linspace(0, len(errs_sorted)-1, n_show, dtype=int)
        errs_sub = errs_sorted[indices]

        im = ax.imshow(errs_sub, aspect='auto', cmap='YlOrRd',
                       extent=[dt, 20*dt, 0, n_show],
                       vmin=0, vmax=np.percentile(errs_sub, 95))
        ax.set_xlabel(f'Prediction Time (s)'); ax.set_ylabel('Sample (sorted by FDE)')
        ax.set_title(f'{label}\nColor = L2 error (m), rows sorted by final error')
        plt.colorbar(im, ax=ax, shrink=0.85, label='Error (m)')

        # Add step markers
        for s in [5, 10, 15]:
            ax.axvline(x=s*dt, color='white', ls=':', lw=0.5, alpha=0.5)

    path = OUT_DIR / '01_error_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 2: Intent confusion matrix
# ================================================================
def chart_intent_confusion(data_low, data_high):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Intent Classification: True vs Predicted', fontsize=13, fontweight='bold')

    for ax, data, names, colors, label in [
        (axes[0], data_low, INTENT_6, COLORS_6, 'LOW (UAV-Flow)'),
        (axes[1], data_high, INTENT_4, COLORS_4, 'HIGH (SimCruise)'),
    ]:
        intents_true = data['intents'].numpy().astype(int)
        intents_pred = data['intent_probs'].argmax(dim=-1).numpy()
        n_class = len(names)

        cm = np.zeros((n_class, n_class))
        for t, p in zip(intents_true, intents_pred):
            if t < n_class and p < n_class:
                cm[t, p] += 1
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(n_class)); ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_yticks(range(n_class)); ax.set_yticklabels(names)
        ax.set_xlabel('Predicted Intent'); ax.set_ylabel('True Intent')
        ax.set_title(f'{label}\n(accuracy={np.trace(cm)/cm.sum()*100:.1f}%)')

        for i in range(n_class):
            for j in range(n_class):
                if cm[i, j] > 0:
                    pct = cm_norm[i, j]
                    color = 'white' if pct > 0.5 else 'black'
                    ax.text(j, i, f'{int(cm[i,j])}\n({pct*100:.0f}%)',
                            ha='center', va='center', fontsize=6, color=color,
                            fontweight='bold' if i == j else 'normal')
        plt.colorbar(im, ax=ax, shrink=0.85, label='Fraction')

    path = OUT_DIR / '02_intent_confusion.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 3: Error vs Speed
# ================================================================
def chart_error_by_speed(data_low, data_high):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Prediction Error vs Speed', fontsize=13, fontweight='bold')

    for row, (data, label, names, colors, dt) in enumerate([
        (data_low, 'LOW (UAV-Flow)', INTENT_6, COLORS_6, DT_LOW),
        (data_high, 'HIGH (SimCruise)', INTENT_4, COLORS_4, DT_HIGH),
    ]):
        speeds = data['speeds'].numpy()
        fde = data['step_errs'][:, -1].numpy()
        intents = data['intents'].numpy().astype(int)
        intent_pred = data['intent_probs'].argmax(dim=-1).numpy()

        # Scatter by intent
        ax = axes[row, 0]
        for c in range(len(names)):
            mask = intents == c
            if mask.sum() < 5: continue
            ax.scatter(speeds[mask], fde[mask], c=colors[c], alpha=0.3, s=3, label=names[c])
        ax.set_xlabel('Speed (m/s)'); ax.set_ylabel('FDE (m)')
        ax.set_title(f'{label}: FDE vs Speed by True Intent')
        ax.legend(fontsize=6, markerscale=3, loc='upper right')
        ax.set_ylim(0, np.percentile(fde, 98))
        ax.grid(True, alpha=0.3)

        # Binned trend: mean FDE by speed bin
        ax = axes[row, 1]
        bins = np.linspace(speeds.min(), speeds.max(), 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []; bin_stds = []
        for i in range(len(bins)-1):
            mask = (speeds >= bins[i]) & (speeds < bins[i+1])
            if mask.sum() > 10:
                bin_means.append(fde[mask].mean())
                bin_stds.append(fde[mask].std())
            else:
                bin_means.append(np.nan); bin_stds.append(np.nan)
        bin_means = np.array(bin_means); bin_stds = np.array(bin_stds)
        valid = ~np.isnan(bin_means)
        ax.plot(bin_centers[valid], bin_means[valid], 'o-', color='#E91E63', lw=2, ms=4, label='Mean FDE')
        ax.fill_between(bin_centers[valid],
                        np.maximum(0, bin_means[valid] - bin_stds[valid]),
                        bin_means[valid] + bin_stds[valid],
                        alpha=0.2, color='#E91E63')
        ax.set_xlabel('Speed (m/s)'); ax.set_ylabel('FDE (m)')
        ax.set_title(f'{label}: Binned FDE vs Speed (±1 std)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    path = OUT_DIR / '03_error_by_speed.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 4: Per-intent error + error over time
# ================================================================
def chart_error_by_intent(data_low, data_high):
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle('Error Analysis by Intent', fontsize=13, fontweight='bold')

    for col, (data, names, colors, dt, label) in enumerate([
        (data_low, INTENT_6, COLORS_6, DT_LOW, 'LOW (UAV-Flow)'),
        (data_high, INTENT_4, COLORS_4, DT_HIGH, 'HIGH (SimCruise)'),
    ]):
        intents = data['intents'].numpy().astype(int)
        errs = data['step_errs'].numpy()  # (N, 20)
        time_axis = np.arange(1, 21) * dt

        # Bar chart: ADE/FDE by intent
        ax_bar = fig.add_subplot(2, 4, col*4 + 1)
        ade_means = []; fde_means = []; counts = []
        active_names = []
        for c, name in enumerate(names):
            mask = intents == c
            count = mask.sum()
            if count < 10:
                ade_means.append(0); fde_means.append(0); counts.append(0)
                active_names.append(name)
                continue
            ade_means.append(errs[mask].mean())
            fde_means.append(errs[mask, -1].mean())
            counts.append(count)
            active_names.append(name)

        x = np.arange(len(active_names)); w = 0.35
        bars1 = ax_bar.bar(x - w/2, ade_means, w, color=[plt.cm.Set2(i/len(active_names)) for i in range(len(active_names))],
                          alpha=0.8, label='ADE')
        bars2 = ax_bar.bar(x + w/2, fde_means, w, color=[plt.cm.Set2(i/len(active_names)) for i in range(len(active_names))],
                          alpha=0.4, label='FDE', hatch='//')
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(active_names, rotation=30, ha='right', fontsize=7)
        ax_bar.set_ylabel('Error (m)'); ax_bar.set_title(f'{label}: ADE/FDE by Intent')
        ax_bar.legend(fontsize=7); ax_bar.grid(True, alpha=0.3, axis='y')

        # Count labels on bars
        for i, (ade, c) in enumerate(zip(ade_means, counts)):
            if c > 0:
                ax_bar.text(i - w/2, ade + max(ade_means)*0.02, f'n={c}', ha='center', fontsize=5, rotation=90)

        # Error over time by intent
        ax_line = fig.add_subplot(2, 4, col*4 + 2)
        for c, (name, color) in enumerate(zip(names, colors)):
            mask = intents == c
            if mask.sum() < 10: continue
            mean_err = errs[mask].mean(axis=0)
            std_err = errs[mask].std(axis=0)
            ax_line.plot(time_axis, mean_err, '-', color=color, lw=2, label=f'{name} (n={mask.sum()})')
            ax_line.fill_between(time_axis, np.maximum(0, mean_err - std_err),
                                mean_err + std_err, alpha=0.15, color=color)
        ax_line.set_xlabel('Time (s)'); ax_line.set_ylabel('L2 Error (m)')
        ax_line.set_title(f'{label}: Error Growth Over Time')
        ax_line.legend(fontsize=6); ax_line.grid(True, alpha=0.3)

        # Per-dimension error share pie
        ax_pie = fig.add_subplot(2, 4, col*4 + 3)
        preds = data['preds'].numpy(); targets = data['targets'].numpy()
        err_x = np.abs(preds[:, :, 0] - targets[:, :, 0]).mean()
        err_y = np.abs(preds[:, :, 1] - targets[:, :, 1]).mean()
        err_z = np.abs(preds[:, :, 2] - targets[:, :, 2]).mean()
        sizes = [err_x, err_y, err_z]
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=['X', 'Y', 'Z'],
                                              autopct='%1.1f%%', colors=['#E91E63','#2196F3','#4CAF50'],
                                              startangle=90, explode=(0, 0, 0.05))
        for t in autotexts: t.set_fontsize(9); t.set_fontweight('bold')
        ax_pie.set_title(f'{label}: Error Dimension Share')

        # Intent probability entropy histogram
        ax_ent = fig.add_subplot(2, 4, col*4 + 4)
        intent_probs = data['intent_probs'].numpy()
        entropy = -(intent_probs * np.log(intent_probs + 1e-8)).sum(axis=1)
        max_ent = np.log(len(names))
        ax_ent.hist(entropy, bins=50, color='#607D8B', alpha=0.7, edgecolor='white')
        ax_ent.axvline(x=entropy.mean(), color='red', ls='--', lw=1.5, label=f'Mean={entropy.mean():.3f}')
        ax_ent.set_xlabel('Intent Entropy (bits)'); ax_ent.set_ylabel('Count')
        ax_ent.set_title(f'{label}: Intent Certainty Distribution\n'
                         f'(0=very certain, {max_ent:.1f}=completely uncertain)')
        ax_ent.legend(fontsize=7); ax_ent.grid(True, alpha=0.3)

    path = OUT_DIR / '04_error_by_intent.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 5: Uncertainty heatmap
# ================================================================
def chart_uncertainty(data_low, data_high):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Model Uncertainty (Predicted Variance) Analysis', fontsize=13, fontweight='bold')

    for col, (data, label, dt) in enumerate([
        (data_low, 'LOW (UAV-Flow)', DT_LOW),
        (data_high, 'HIGH (SimCruise)', DT_HIGH),
    ]):
        unc = data['uncertainties'].numpy()  # (N, 20, 3)
        errs = data['step_errs'].numpy()      # (N, 20)
        time_axis = np.arange(1, 21) * dt

        # Uncertainty over time per dimension
        ax = axes[0, col]
        for d, dname, color in [(0, 'X', '#E91E63'), (1, 'Y', '#2196F3'), (2, 'Z', '#4CAF50')]:
            mean_unc = unc[:, :, d].mean(axis=0)
            ax.plot(time_axis, mean_unc, '-', color=color, lw=2, label=f'{dname}')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean Log-Variance')
        ax.set_title(f'{label}: Uncertainty Growth Over Time')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Calibration: predicted variance vs actual error
        ax = axes[1, col]
        pred_var = np.exp(unc.clip(-10, 10)).mean(axis=(1, 2))  # (N,) mean variance
        fde = errs[:, -1]  # (N,)
        # Bin by variance
        bins = np.percentile(pred_var, np.linspace(0, 100, 15))
        bin_centers = []; bin_errors = []; bin_counts = []
        for i in range(len(bins)-1):
            mask = (pred_var >= bins[i]) & (pred_var < bins[i+1])
            if mask.sum() > 20:
                bin_centers.append(pred_var[mask].mean())
                bin_errors.append(fde[mask].mean())
                bin_counts.append(mask.sum())
        ax.scatter(bin_centers, bin_errors, s=[max(20, min(200, c/5)) for c in bin_counts],
                  c='#FF5722', alpha=0.7, edgecolors='black', lw=0.5)
        # Perfect calibration line
        max_val = max(max(bin_centers), 0.1) if bin_centers else 1
        ax.plot([0, max_val], [0, np.sqrt(max_val)], 'k--', lw=1, alpha=0.5, label='Perfect (σ)')
        ax.set_xlabel('Mean Predicted Variance'); ax.set_ylabel('Mean FDE (m)')
        ax.set_title(f'{label}: Uncertainty Calibration\n'
                     f'(dots should follow dashed line if well-calibrated)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    path = OUT_DIR / '05_uncertainty.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 6: Context Adapter Before/After on HIGH model + Simulation Data
# ================================================================
def chart_adapter_comparison():
    """
    Train Context Adapter on long simulation trajectories (same approach as context_sim.py).
    Tests on held-out windows from same trajectory. Shows before/after comparison.
    """
    print('\nRunning Context Adapter comparison on HIGH model with simulation data...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = DronePredictor(device=device)

    # Load long simulation trajectories
    d = np.load('../UAVTrajectoryDataset/trajectories_merged.npz')
    positions = d['positions'].astype(np.float32)
    masks = d['masks']
    lengths = masks.sum(axis=1)
    candidates = np.where(lengths >= 700)[0]  # >=140 frames at 1Hz
    np.random.seed(42)
    picks = np.random.choice(candidates, min(6, len(candidates)), replace=False)

    trajs = []
    for idx in picks:
        L = int(lengths[idx])
        pos_raw = positions[idx, :L]
        pos = pos_raw[::5].copy()  # 5Hz → 1Hz
        vel = np.zeros_like(pos); vel[1:] = (pos[1:]-pos[:-1])/1.0; vel[0]=vel[1]
        trajs.append({'idx': int(idx), 'pos': pos, 'vel': vel, 'len': len(pos)})
    d.close()

    print(f'  Loaded {len(trajs)} long trajectories')

    d_model = p.high.d_model
    all_results = []

    for t in trajs:
        pos, vel = t['pos'], t['vel']
        T = t['len']
        max_start = T - 80
        train_end = int(max_start * 0.6)

        # Training windows
        ctx_len = 60
        train_data = []
        for s in range(0, train_end, 2):
            hist_start = s + ctx_len - 20
            o = pos[hist_start]
            hp = pos[hist_start:hist_start+20] - o
            h = np.concatenate([hp, vel[hist_start:hist_start+20]], axis=1)
            co = pos[s]; cp = pos[s:s+ctx_len] - co
            c = np.concatenate([cp, vel[s:s+ctx_len]], axis=1)
            tgt_start = hist_start + 20
            tgt = pos[tgt_start:tgt_start+20] - pos[tgt_start-1]
            train_data.append((
                torch.from_numpy(h).float(), torch.from_numpy(c).float(),
                torch.from_numpy(tgt).float(),
            ))

        if len(train_data) < 8:
            print(f'  #{t["idx"]}: too few windows ({len(train_data)}), skipping')
            continue

        # Train adapter
        adapter = ContextAdapterV2(d_model=d_model, hidden=128).to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-5)

        ua = p.high.ua_pgd
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
                idxs = perm[b:b+bs]
                hb = torch.stack([train_data[i][0] for i in idxs]).to(device)
                cb = torch.stack([train_data[i][1] for i in idxs]).to(device)
                tb = torch.stack([train_data[i][2] for i in idxs]).to(device)
                opt.zero_grad()
                ua._ctx = adapter(cb)
                out_t = p.high(hb, force_predict=True)
                loss = F.mse_loss(out_t['predictions'], tb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
                opt.step()

        # Test on 4 held-out windows
        test_starts = np.linspace(train_end + 10, max_start - 1, 4, dtype=int)
        traj_results = []
        for ts in test_starts:
            hist_start = ts + ctx_len - 20
            o = pos[hist_start]
            hp = pos[hist_start:hist_start+20] - o
            h = np.concatenate([hp, vel[hist_start:hist_start+20]], axis=1)
            co = pos[ts]; cp = pos[ts:ts+ctx_len] - co
            c = np.concatenate([cp, vel[ts:ts+ctx_len]], axis=1)
            tgt_start = hist_start + 20
            tgt = pos[tgt_start:tgt_start+20] - pos[tgt_start-1]

            h_t = torch.from_numpy(h).float().unsqueeze(0).to(device)
            c_t = torch.from_numpy(c).float().unsqueeze(0).to(device)
            tgt_t = torch.from_numpy(tgt).float()

            # Before
            ua._ctx = None
            with torch.no_grad():
                out_b = p.high(h_t, force_predict=True)
            pb = out_b['predictions'][0].cpu()

            # After
            with torch.no_grad():
                ua._ctx = adapter(c_t)
            with torch.no_grad():
                out_a = p.high(h_t, force_predict=True)
            pa = out_a['predictions'][0].cpu()

            err_b = torch.norm(pb[-1] - tgt_t[-1]).item()
            err_a = torch.norm(pa[-1] - tgt_t[-1]).item()

            traj_results.append({
                'start': ts, 'hist': h_t[0].cpu(), 'ctx': c_t[0].cpu(),
                'target': tgt_t, 'pred_b': pb, 'pred_a': pa,
                'err_b': err_b, 'err_a': err_a,
            })

        ua.neural_decoder.forward = _orig_nd
        all_results.append({'idx': t['idx'], 'len': T, 'tests': traj_results})
        speed = np.linalg.norm(vel, axis=1).mean()
        avg_improve = np.mean([(r['err_b']-r['err_a'])/r['err_b']*100 for r in traj_results])
        print(f'    #{t["idx"]}: {T}fr speed={speed:.0f}m/s '
              f'err {np.mean([r["err_b"] for r in traj_results]):.1f}→{np.mean([r["err_a"] for r in traj_results]):.1f}m '
              f'({avg_improve:+.0f}%)')

    # --- Plot: 2 rows per trajectory, 4 test windows each ---
    n_trajs = len(all_results)
    fig, axes = plt.subplots(n_trajs, 4, figsize=(22, 5 * n_trajs))
    if n_trajs == 1: axes = axes.reshape(1, -1)
    fig.suptitle('HIGH Model + Context Adapter on Simulation Trajectories\n'
                 'Blue=History  Red=Before(Base)  Orange=After(Adapter)  Green=Truth\n'
                 'Adapter trained on first 60% of each trajectory, tested on held-out windows',
                 fontsize=13, fontweight='bold', y=1.005)

    MK = {'5s': 5, '10s': 10, '15s': 15, '20s': 19}
    MC = ['#FF9800', '#FF5722', '#E91E63', '#9C27B0']

    for ti, tr in enumerate(all_results):
        for i in range(4):
            r = tr['tests'][i]
            ax = axes[ti, i]
            last = r['hist'][-1, :2].numpy()
            hp = r['hist'][:, :2].numpy()
            pb = r['pred_b'].numpy()[:, :2] + last
            pa = r['pred_a'].numpy()[:, :2] + last
            tt = r['target'].numpy()[:, :2] + last

            ax.plot(tt[:, 0], tt[:, 1], '-', color='#4CAF50', lw=2.5, alpha=0.6, label='Truth')
            ax.plot(hp[:, 0], hp[:, 1], '-', color='#2196F3', lw=2, label='History')
            ax.plot(pb[:, 0], pb[:, 1], '--', color='#F44336', lw=1.5, label='Before')
            ax.plot(pa[:, 0], pa[:, 1], '-', color='#FF9800', lw=2, label='After')

            for j, (lbl, fi) in enumerate(zip(MK.keys(), MK.values())):
                ax.scatter(pa[fi, 0], pa[fi, 1], c=MC[j], s=50, marker='D', zorder=15, ec='k', lw=0.5)
                ax.scatter(tt[fi, 0], tt[fi, 1], c=MC[j], s=40, marker='o', zorder=15, ec='k', lw=0.5)
            ax.scatter(hp[-1, 0], hp[-1, 1], c='#2196F3', s=60, marker='s', zorder=5)

            improve = (r['err_b'] - r['err_a']) / r['err_b'] * 100 if r['err_b'] > 0 else 0
            ax.set_title(f'#{tr["idx"]} t={r["start"]} | {r["err_b"]:.1f}→{r["err_a"]:.1f}m ({improve:+.0f}%)',
                         fontsize=7, fontweight='bold')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_box_aspect(1); ax.grid(True, alpha=0.3)
            if ti == 0 and i == 0:
                ax.legend(fontsize=5, loc='upper right')

    path = OUT_DIR / '06_adapter_comparison.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 7: Trajectory overlay grid
# ================================================================
def chart_trajectory_grid(data_low, data_high):
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('Trajectory Prediction Grid: 6 LOW + 6 HIGH Samples',
                 fontsize=13, fontweight='bold')

    # Pick representative samples from each intent
    np.random.seed(123)

    for dataset_idx, (data, label, names, colors, dt) in enumerate([
        (data_low, 'LOW', INTENT_6, COLORS_6, DT_LOW),
        (data_high, 'HIGH', INTENT_4, COLORS_4, DT_HIGH),
    ]):
        intents = data['intents'].numpy().astype(int)
        preds = data['preds'].numpy()
        targets = data['targets'].numpy()
        hists = data['hist'].numpy()
        speeds = data['speeds'].numpy()

        selected_indices = []
        for c in range(len(names)):
            mask = np.where(intents == c)[0]
            if len(mask) > 0:
                # Pick median-error sample
                fde_vals = data['step_errs'][mask, -1].numpy()
                median_idx = mask[np.argsort(fde_vals)[len(fde_vals)//2]]
                selected_indices.append(median_idx)

        # Fill to 6 samples
        while len(selected_indices) < 6:
            remaining = [i for i in range(len(intents)) if i not in selected_indices]
            if not remaining: break
            selected_indices.append(np.random.choice(remaining))

        for i, idx in enumerate(selected_indices[:6]):
            ax = fig.add_subplot(3, 4, dataset_idx * 6 + i + 1)
            last = hists[idx, -1, :2]
            hp = hists[idx, :, :2]
            pa = preds[idx, :, :2] + last
            ta = targets[idx, :, :2] + last

            ax.plot(hp[:, 0], hp[:, 1], '-', color='#2196F3', lw=2, label='Hist')
            ax.plot(pa[:, 0], pa[:, 1], '--', color='#F44336', lw=1.5, label='Pred')
            ax.plot(ta[:, 0], ta[:, 1], '-', color='#4CAF50', lw=2, alpha=0.6, label='Truth')
            ax.scatter(pa[-1, 0], pa[-1, 1], c='#F44336', s=40, marker='X', zorder=10)
            ax.scatter(ta[-1, 0], ta[-1, 1], c='#4CAF50', s=40, marker='o', zorder=10)

            intent_c = intents[idx]
            intent_name = names[intent_c] if intent_c < len(names) else '?'
            fde = data['step_errs'][idx, -1].item()
            ax.set_title(f'#{idx} {intent_name} {speeds[idx]:.1f}m/s FDE={fde:.2f}m',
                        fontsize=7, fontweight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_box_aspect(1); ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=5)

    path = OUT_DIR / '07_trajectory_grid.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Chart 8: Summary metrics table
# ================================================================
def chart_summary(data_low, data_high):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')
    fig.suptitle('Model Performance Summary', fontsize=14, fontweight='bold', y=1.01)

    rows = [
        'ADE (mean)', 'ADE (median)', 'ADE (P95)',
        'FDE (mean)', 'FDE (median)', 'FDE (P95)',
        'Direction Error (mean)', 'Direction Error (median)',
        'Speed RMSE (mean)',
        'Intent Accuracy',
        '---',
        'STRAIGHT FDE', 'TURN_L FDE', 'TURN_R FDE',
        'DESCEND FDE', 'HOVER FDE',
    ]

    def compute_row_vals(data, names):
        errs = data['step_errs'].numpy()
        ade = errs.mean(axis=1); fde = errs[:, -1]
        intents = data['intents'].numpy().astype(int)
        intent_pred = data['intent_probs'].argmax(dim=-1).numpy()
        acc = (intents == intent_pred).mean() * 100

        vals = [
            f'{ade.mean():.3f}m', f'{np.median(ade):.3f}m', f'{np.percentile(ade, 95):.3f}m',
            f'{fde.mean():.3f}m', f'{np.median(fde):.3f}m', f'{np.percentile(fde, 95):.3f}m',
        ]

        # Direction and speed errors
        preds_np = data['preds'].numpy(); targets_np = data['targets'].numpy()
        dir_errs = []
        for i in range(min(5000, len(preds_np))):
            pv = preds_np[i, -1, :2]; tv = targets_np[i, -1, :2]
            if np.linalg.norm(pv) > 0.01 and np.linalg.norm(tv) > 0.01:
                dot = np.dot(pv, tv) / (np.linalg.norm(pv) * np.linalg.norm(tv))
                dir_errs.append(np.degrees(np.arccos(np.clip(dot, -1, 1))))
        dir_errs = np.array(dir_errs) if dir_errs else np.array([0])
        vals += [f'{dir_errs.mean():.1f}deg', f'{np.median(dir_errs):.1f}deg']

        # Speed RMSE
        spd_rmse = np.sqrt(((preds_np - targets_np)**2).mean())
        vals.append(f'{spd_rmse:.3f}m/s')
        vals.append(f'{acc:.1f}%')
        vals.append('')

        # Per-intent FDE
        for c in range(len(names)):
            mask = intents == c
            if mask.sum() > 10:
                vals.append(f'{fde[mask].mean():.3f}m')
            else:
                vals.append('N/A')
        return vals

    low_vals = compute_row_vals(data_low, INTENT_6)
    high_vals = compute_row_vals(data_high, INTENT_4)

    # Pad to same length
    max_rows = max(len(low_vals), len(high_vals))
    while len(low_vals) < max_rows: low_vals.append('')
    while len(high_vals) < max_rows: high_vals.append('')
    while len(rows) < max_rows: rows.append('')

    tbl = ax.table(cellText=list(zip(low_vals, high_vals)),
                   rowLabels=rows[:max_rows],
                   colLabels=['LOW (UAV-Flow)', 'HIGH (SimCruise)'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.5, 1.8)

    # Style header
    for j in range(2):
        tbl[(0, j)].set_facecolor('#263238')
        tbl[(0, j)].get_text().set_color('white')
        tbl[(0, j)].get_text().set_fontweight('bold')

    # Style rows
    for i in range(1, max_rows):
        for j in range(2):
            if i % 2 == 0:
                tbl[(i, j)].set_facecolor('#F5F5F5')
        # Bold row labels
        tbl[(i, -1)].get_text().set_fontweight('bold')

    path = OUT_DIR / '08_summary_table.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ================================================================
# Main
# ================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    p = DronePredictor(device=device)

    # Collect data
    print('\nCollecting LOW samples...')
    ds_low = FastWindowDataset('../UAV-Flow-pure', split='test')
    data_low = collect_samples(ds_low, p.low, 'LOW', device, N_SAMPLES)

    print('\nCollecting HIGH samples...')
    ds_high = FastWindowDataset('../SimCruise', split='test', label_remap={4: 3})
    data_high = collect_samples(ds_high, p.high, 'HIGH', device, N_SAMPLES)

    # Generate charts
    print('\nGenerating charts...')
    chart_error_heatmap(data_low, data_high)
    chart_intent_confusion(data_low, data_high)
    chart_error_by_speed(data_low, data_high)
    chart_error_by_intent(data_low, data_high)
    chart_uncertainty(data_low, data_high)
    chart_adapter_comparison()
    chart_trajectory_grid(data_low, data_high)
    chart_summary(data_low, data_high)

    print('\nAll 8 charts saved to pic-results/')
    print('Done!')


if __name__ == '__main__':
    main()
