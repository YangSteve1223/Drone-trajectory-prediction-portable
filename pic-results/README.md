# Plot Reference

Dataset: UAV-Flow (real DJI, 5Hz) and SimCruise (sim cruise, 1Hz). K=5 multi-hypothesis (1 dead head filtered). Z-axis scaled to XY ratio in 3D plots.

## LOW (40-frame model)

| Plot | Content |
|:--|:--|
| `low40_multihyp_3d.png` | K=5 predictions (3D), adaptive-stride history |
| `low40_multihyp_xy.png` | XY top-down view |
| `low40_per_second_error.png` | Per-second error: single vs minFDE_5 |
| `low40_error_table.png` | Per-sample single/minADE/minFDE table |

## HIGH (20-frame model)

| Plot | Content |
|:--|:--|
| `high_12_multihyp_3d.png` | K=5 predictions (3D), 20s history@1Hz |
| `high_12_best_trajectory_3d.png` | Highest-confidence trajectory only |
| `high_12_best_trajectory_xy.png` | Top-down view |
| `high_12_persecond_error.png` | Per-5s error bars (12 samples) |
| `high_6_adapter_comparison.png` | Single vs multi-hypothesis (6 samples, XY) |

## Multi-Hypothesis Evaluation

| Plot | Content |
|:--|:--|
| `eval_multihypothesis.png` | 6-panel: FDE/ADE distribution, per-intent FDE, DESCEND FDE, scatter, metrics table |
| `eval_multihypothesis_error_growth.png` | Per-intent error growth over time |
| `multihyp_trajectory_grid.png` | 16-sample trajectory grid |
| `multihyp_descend_deep.png` | DESCEND sample deep-dive (all 5 heads) |
| `summary_table_updated.png` | Core metrics summary table |

## Session 2 (40-frame follow-ups)

| Plot | Content |
|:--|:--|
| `session2_summary.png` | 4-panel: global LoRA FDE, dir error, multi-hyp + LoRA stacking, gate-LoRA on extreme turns |
| `lora_40frame/` | Per-drone LoRA on 40-frame: overview + detail (eval_lora training output) |

## Key Numbers

| Experiment | Base | Result | Gain |
|:--|:--:|:--:|:--:|
| global_lora_40 (window-level) | FDE 0.826 | 0.703 | +14.9% |
| dir_lora_40 (window-level) | FDE 0.826 | 0.665 | +19.4% |
| dir_lora_40 cross-flight | FDE 0.758 | 0.702 | +7.3% |
| Multi-hyp K=5 (40f) | FDE 0.874 | 0.598 | +31.6% |
| LoRA stacking (global+local) | FDE 0.322 | 0.290 | +10.2% |
| Gate-LoRA extreme turns | cata 10.65% | 8.89% | -1.76pp |

> Negative: HIGH 20→40f hurts (1Hz cruise, no input bottleneck). Multi-frame physics seed rejected.
