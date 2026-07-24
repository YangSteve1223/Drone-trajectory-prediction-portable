#!/usr/bin/env python3
"""
Summary figure for the 40-frame follow-up work (session 2):
  - LoRA progression on the 40-frame base (base / global / direction / stacked)
  - Multi-hypothesis on 40-frame (single vs minFDE_5)
  - Gate-LoRA on extreme turns (base vs +gate on the >60deg subset)

Reads the result JSONs in pic-results/ and renders one comparison panel.
Output: pic-results/session2_summary.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PR = Path(__file__).resolve().parents[1] / 'pic-results'
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'figure.dpi': 150})


def load(name):
    p = PR / name
    return json.load(open(p)) if p.exists() else None


def main():
    dir_lora = load('dir_lora_40.json')
    glob_lora = load('global_lora_40.json')
    stack = load('lora_stack_40.json')
    mh = load('multihead_40_K5.json')
    gate = load('gate_lora_40.json')

    fig = plt.figure(figsize=(15, 4.2))

    # Panel 1: Global LoRA variants FDE (mixed held-out, 1783 windows)
    ax1 = fig.add_subplot(1, 4, 1)
    labels = ['40f\nbase', '+global\nLoRA', '+dir\nLoRA']
    base_fde = dir_lora['base_fde']
    vals = [base_fde, glob_lora['lora_fde'], dir_lora['lora_fde']]
    colors = ['#90A4AE', '#42A5F5', '#FF6D00']
    bars = ax1.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    ax1.set_ylabel('FDE (m)'); ax1.set_title('Global LoRA on 40-frame\n(mixed held-out)', fontweight='bold')
    ax1.set_ylim(0, base_fde * 1.2)
    ax1.text(0.5, -0.30, f'dir-LoRA: +{dir_lora["fde_gain_pct"]:.1f}% FDE, '
             f'dir {dir_lora["base_dir"]:.1f}->{dir_lora["lora_dir"]:.1f}deg',
             transform=ax1.transAxes, ha='center', fontsize=7.5, color='#555')

    # Panel 2: Direction error improvement
    ax2 = fig.add_subplot(1, 4, 2)
    dlabels = ['40f\nbase', '+global\nLoRA', '+dir\nLoRA']
    dvals = [dir_lora['base_dir'], glob_lora['lora_dir'], dir_lora['lora_dir']]
    bars = ax2.bar(dlabels, dvals, color=colors, width=0.6)
    for b, v in zip(bars, dvals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f'{v:.1f}', ha='center', fontsize=8)
    ax2.set_ylabel('Direction error (deg)'); ax2.set_title('Direction error\n(lower is better)', fontweight='bold')
    ax2.set_ylim(0, dir_lora['base_dir'] * 1.25)

    # Panel 3: Multi-hypothesis + LoRA stacking
    ax3 = fig.add_subplot(1, 4, 3)
    m_single = mh['single_fde']; m_min = mh['min_fde']
    s_base = stack['base_mean_fde']; s_local = stack['local_mean_fde']; s_stack = stack['stacked_mean_fde']
    g = ['MH\nsingle', 'MH\nminK', 'stack\nbase', 'stack\n+local', 'stack\n+glob\n+local']
    gv = [m_single, m_min, s_base, s_local, s_stack]
    gc = ['#90A4AE', '#66BB6A', '#90A4AE', '#42A5F5', '#FF6D00']
    bars = ax3.bar(g, gv, color=gc, width=0.7)
    for b, v in zip(bars, gv):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.2f}', ha='center', fontsize=7.5)
    ax3.set_ylabel('FDE (m)'); ax3.set_title('Multi-hyp (K=5) & LoRA stacking\n(long-traj)', fontweight='bold')
    ax3.text(0.5, -0.32, f'minFDE_5: +{mh["fde_gain_pct"]:.0f}% | '
             f'stack vs local: +{stack["stacked_vs_local_pct"]:.0f}% '
             f'({stack["stacked_wins"]}/{stack["n_traj"]} wins)',
             transform=ax3.transAxes, ha='center', fontsize=7, color='#555')

    # Panel 4: Gate-LoRA on extreme turns
    ax4 = fig.add_subplot(1, 4, 4)
    be = gate['base_extreme']; ge = gate['gate_extreme']
    x = np.arange(2); w = 0.35
    ax4.bar(x - w / 2, [be['cata'], be['dir']], w, label='base', color='#90A4AE')
    ax4.bar(x + w / 2, [ge['cata'], ge['dir']], w, label='+gate-LoRA', color='#FF6D00')
    ax4.set_xticks(x); ax4.set_xticklabels(['Cata %', 'Dir (deg)'])
    ax4.set_title(f'Gate-LoRA on extreme turns\n(>{gate["turn_eval_deg"]:.0f}deg subset)', fontweight='bold')
    ax4.legend(fontsize=7)
    for i, (bv, gv_) in enumerate([(be['cata'], ge['cata']), (be['dir'], ge['dir'])]):
        ax4.text(i - w / 2, bv + 0.3, f'{bv:.1f}', ha='center', fontsize=7)
        ax4.text(i + w / 2, gv_ + 0.3, f'{gv_:.1f}', ha='center', fontsize=7)

    fig.suptitle('40-Frame Follow-ups: Global/Direction LoRA, Multi-Hypothesis, Stacking, Gate-LoRA',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = PR / 'session2_summary.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
