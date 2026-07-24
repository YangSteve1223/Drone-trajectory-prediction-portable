# EMAM — Drone Trajectory Prediction System (Portable)

A drone trajectory prediction system built on the EMAM (Enhanced Mamba) architecture. Key features:

- **Speed-adaptive dual-model soft fusion** — LOW (real low-speed) + HIGH (simulated cruise)
- **40-frame long-history extension** — long-trajectory FDE cut from 2.32 m to 0.87 m, catastrophic failures reduced to near zero
- **LoRA personalization** — global LoRA (icing on the cake) + per-drone LoRA (online learning)
- **Multi-hypothesis prediction** — K=5 Winner-Takes-All, minFDE clearly outperforms the single-hypothesis model
- **Online continual learning** — per-drone streaming incremental adaptation (positive gains verified on both LOW and HIGH)

## Quick Start

```bash
pip install -r requirements.txt
```

The system has two inference entry points. Choose based on your use case:

### 1. `DronePredictor` — General Short-Range Inference (20-frame input)

With Z-axis correction + dual-model soft fusion. Suitable for general real-time prediction.

```python
import torch
from predictor import DronePredictor

predictor = DronePredictor()

# Input: (B, 20, 6) — 20 history frames, each [x, y, z, vx, vy, vz] (meters, m/s)
history = torch.randn(1, 20, 6)
result = predictor.predict(history)

result['predictions']    # (1, 20, 3) future displacement (meters)
result['intent_logits']  # intent classification logits
result['speed']          # current speed (m/s)
result['route']          # 'LOW' | 'HIGH' routing decision
```

### 2. `DeployedLowPredictor` — Long Trajectory + Per-Drone Online Learning (40-frame input)

For persistently-flying low-speed drones. Learns as it flies, building a personalized LoRA for each drone.

```python
import torch
from deploy import DeployedLowPredictor

deployer = DeployedLowPredictor(use_global=True)  # global LoRA enabled by default

hist = torch.randn(1, 40, 6)          # 40 history frames
result = deployer.predict(
    hist,
    drone_id='drone_007',             # set this to enable per-drone online LoRA
    ground_truth=future_disp,         # (1, 20, 3) previous-step ground truth, fed to online learning (optional)
    frames_seen=120,                  # cumulative frames streamed by this drone (for length gate)
)
result['predictions']  # (1, 20, 3) future displacement
```

Three gates (see `deploy.py` docstring for details):

1. **Length gate** (`online_min_frames=60`): short flights use the clean 40-frame base only, no LoRA.
2. **Global toggle** (`use_global`): master on/off switch for the shared low-speed global LoRA (`dir_lora_40`).
3. **Speed gate** (`global_max_speed=4.0`): disables global LoRA above 4 m/s (it was trained on 0–3 m/s real flights; out-of-domain use is unsafe).

## Architecture

```
Input (B,T,6) → EMam-SE (SSM encoder) → IA-DTP (intent classifier) → UA-PGD (decoder)
                                                             ├─ Physics Inertia Gate
                                                             ├─ Neural Decoder (1 or K heads)
                                                             └─ Kinematic Physics Model
```

### Dual-Model Soft Fusion

| Model | Dataset | Classes | Frequency | Speed Domain | History Length |
|:--|:--|:--|:--|:--|:--|
| LOW | UAV-Flow (real DJI) | 6-class | 5 Hz | 0–3 m/s | 40 frames (long) / 20 frames (short) |
| HIGH | SimCruise (simulated cruise) | 4-class | 1 Hz | 8–28 m/s | 20 frames |

- Soft fusion: `α = sigmoid((speed - 5.0) / 1.2)`, smooth blending in the [2, 8] m/s transition zone; hard assignment outside to avoid cross-domain contamination.
- **HIGH stays at 20 frames**: 40-frame expansion on 1 Hz cruise data is a net negative (20 frames = 20 s already); verified and rejected.

### 40-Frame Expansion (LOW)

- `low_speed_6class_40frame.pth`: migrated 147/148 weights from the 20-frame checkpoint, froze the encoder, fine-tuned the decoder + gate.
- Long-trajectory coverage: 7% → 84%, FDE: 0.87 m (vs. 2.32 m at 20 frames), catastrophic failures near zero.
- Adaptive-stride sampling: `stride = max(1, min(4, n // 60))`.

### LoRA Strategy

LoRA = **icing on the cake**, not the cake itself. The base model first handles all trajectory lengths; LoRA then provides incremental personalization.

- **Upstream targets (excluding delta_head)**: SSM in/out_proj + `ua_pgd.feat_compress` + `neural_decoder.proj.0`. Head-only target: `anchor_to_pos.2`. ~100K parameters.
- **Why delta_head is excluded**: it independently processes 20 prediction steps; applying LoRA on it amplifies step-to-step variance → zigzag trajectories.
- **Global LoRA**: `dir_lora_40.pth` (current best, trained with direction-weighted loss).
  - Window-level split: FDE +19.4%, direction error 13.8° → 12.1° (same-flight windows may appear in both train/test — optimistic).
  - **Cross-flight split (honest generalization, 20% flights held out): FDE +7.3%, direction +4.6%** (`eval/eval_global_lora_generalization.py`). Use this number for real deployment.
- **Stacking**: 40-frame base → merged global LoRA → per-drone online LoRA.

### Online Continual Learning

Per-drone streaming incremental adaptation: replay buffer + CUSUM drift detection + resident adapter, gentle config (lr 3e-5, accum 10, l2 0.05).

- **Causal verification finding**: under realistic deployment constraints (only causal warmup data available), streaming online (+2.4%) vastly outperforms offline batch fine-tuning (−202% catastrophic overfit).
- LOW: FDE +1.6%, 21/30 drones improve (5 Hz, limited data per flight segment → modest gain).
- HIGH: FDE +5.1%, 25/25 drones improve (cruise data has more systematic bias, no global layer → larger personalization headroom).

### Multi-Hypothesis Prediction

K=5 independent prediction heads + confidence scoring head, Winner-Takes-All training. At inference, pick the highest-confidence head, or evaluate with minFDE_K.

## Directory Structure

The root directory contains only **runtime libraries** (core modules imported everywhere) + the two inference entry points. Scripts are organized by purpose into `eval/` `train/` `viz/` `tests/`.

```
├── predictor.py              # DronePredictor — general inference entry (soft fusion + Z correction)
├── deploy.py                 # DeployedLowPredictor — long trajectory + online learning entry (3 gates)
├── online_config.py          # Authoritative online-learning config (LoRA targets / weight files / gate thresholds)
├── online_learner.py         # OnlineLearner — streaming incremental training
├── adapter_manager.py        # DroneAdapterManager — per-drone adapter persistence
├── streaming.py              # Streaming data buffer + drift detection
├── lora.py                   # LoRALinear / LoRAAdapter — low-rank adaptation core
├── dynamic_norm.py           # Dynamic normalization
├── fix_labels.py             # UAV-Flow label correction
├── emam_model/               # EMAM model architecture
│   ├── model.py              # TrajectoryPredictor
│   ├── emam_se.py            # Enhanced Mamba + SE (with optional chunked SSM scan)
│   ├── ia_dtp.py             # Intent-Aware DTP
│   ├── ua_pgd.py             # UA-PGD + multi-hypothesis decoder + kinematic physics model
│   ├── trigger.py            # Event trigger
│   └── bidirectional_mamba.py # Bidirectional enhancer (standalone experiment, not used in main pipeline)
├── utils/                    # Data loaders & evaluation metrics & logging
├── eval/                     # Evaluation scripts (evaluate / eval_lora / eval_online_* …)
├── train/                    # Training pipelines + LoRA/multi-hypothesis training + 40-frame expansion
├── viz/                      # Visualization + diagnostics + rollout
├── tests/                    # Regression tests (deploy gates / SSM scan) + latency benchmark
├── weights/                  # Pretrained weights (see table below)
├── pic-results/              # Evaluation plots + README.md
└── reviewtodelete/           # Deprecated file staging area (gitignored, pending manual deletion)
```

> Run from the project root, e.g. `python eval/evaluate.py`, `python tests/test_deploy_gates.py`.

**Docs:** `README.md` (this file — English), `README_CN.md` (Chinese), `INTERFACE.md` (system integration ICD), `ROADMAP.md` (multi-drone interaction & future directions).

## Weight Files (`weights/`)

| File | Description |
|:--|:--|
| `low_speed_6class.pth` | LOW 20-frame original model (short/mid-range) |
| `low_speed_6class_40frame.pth` | LOW 40-frame expansion (default base for long trajectories) |
| `high_speed_4class.pth` | HIGH 20-frame model |
| `dir_lora_40.pth` | **Current best global LoRA** (direction-weighted, cross-flight +7.3% / window-level +19.4% FDE) |
| `global_lora_40.pth` | Early global LoRA (+14.9%, superseded by dir_lora_40) |
| `gate_lora_40.pth` | Gate LoRA (reduces extreme-turn catastrophe rate −1.76 pp) |
| `low_multihead_K5_40frame.pth` | LOW K=5 multi-hypothesis heads |
| `high_multihead_K5.pth` | HIGH K=5 multi-hypothesis heads |

## Script Reference

| Script | Purpose |
|:--|:--|
| `predictor.py` | General inference entry (root) |
| `deploy.py` | Long-trajectory + online learning deployment entry (root, includes smoke self-test) |
| `eval/evaluate.py` | Comprehensive ADE/FDE/intent/uncertainty evaluation |
| `eval/eval_multihead.py` | Full test-set multi-hypothesis evaluation |
| `eval/eval_lora.py` / `eval/eval_lora_stack.py` | LoRA / global+local stacking evaluation |
| `eval/eval_online_learning.py` / `eval/eval_online_learning_high.py` | LOW / HIGH online learning validation |
| `eval/eval_online_vs_offline.py` | Online vs. offline batch comparison under causal constraints |
| `train/expand_model_low.py` / `train/expand_model_high.py` | 20→40 frame expansion experiments (LOW positive / HIGH negative) |
| `train/train_lora_direction.py` / `train/train_lora_gate.py` / `train/train_lora_global.py` | Global LoRA training variants |
| `train/train_multihead_low.py` / `train/train_multihead.py` | Multi-hypothesis decoder training (WTA) |
| `viz/visualize_low.py` / `viz/visualize_multihead.py` / `viz/visualize_trajectories.py` / `viz/visualize_summary.py` | Plot generation |
| `viz/diagnose_failures.py` | Deep-dive diagnostics on worst-case samples |
| `viz/rollout.py` | Autoregressive prediction extrapolation |
| `fix_labels.py` | UAV-Flow label correction (root) |
| `eval/eval_global_lora_generalization.py` | Global LoRA cross-flight generalization validation (flight-level disjoint split) |
| `tests/test_deploy_gates.py` | Regression tests for the three deployment gates + online learner resident adapter |
| `tests/test_ssm_scan.py` | Chunked SSM scan numerical equivalence test (standalone + pytest) |
| `tests/benchmark_latency.py` | Inference latency benchmark (both entry points + online update) |

> All scripts are run from the **project root** (e.g. `python eval/evaluate.py`). Scripts auto-locate the root directory to load `weights/` and datasets.

## Performance

### LOW — Full Improvement Chain (Long Trajectories)

| Stage | Long-Traj FDE | Direction Error | Catastrophe Rate | Coverage |
|:--|:--:|:--:|:--:|:--:|
| Original 20-frame | 2.32 m | 54° | 20.0% | 7% |
| 40-frame expansion | 0.87 m | ~15° | ~0% | 84% |
| 40-frame + LoRA | 0.23 m | ~5° | 0% | 84% |

### Multi-Hypothesis (K=5, minFDE_5 vs. single model)

| | LOW | HIGH |
|:--|:--:|:--:|
| minFDE_5 | 0.60 m (+31.6%) | 1.84 m (+66%) |
| minADE_5 | 0.41 m (+18.2%) | 0.93 m (+46%) |
| FDE P95 | 1.57 m | 5.22 m |

### Online Learning (Causal Verification)

| | Frozen | Streaming Online | Improvement |
|:--|:--:|:--:|:--:|
| LOW FDE | 0.648 m | 0.638 m | +1.6% (21/30 drones) |
| HIGH FDE | 4.28 m | 4.06 m | +5.1% (25/25 drones) |

### Global LoRA Generalization (Honest Evaluation)

| Split Method | FDE Gain | Direction Gain | Notes |
|:--|:--:|:--:|:--|
| Window-level (old) | +19.4% | +12.3% | Same-flight windows may appear in both train/test — optimistic |
| **Cross-flight (20% flights held out)** | **+7.3%** | **+4.6%** | Held-out flights never seen during training — **use this for deployment** |

### Inference Latency (RTX 3060 Laptop, batch=1, single-drone stream)

| Call Path | Mean | P50 | P95 | Throughput |
|:--|:--:|:--:|:--:|:--:|
| `DronePredictor.predict` (20f) | 35.3 ms | 34.2 ms | 44.2 ms | 28 fps |
| `DeployedLowPredictor.predict` (40f, inference only) | 20.1 ms | 18.9 ms | 27.3 ms | 50 fps |
| `DeployedLowPredictor.predict` (40f, with online update) | 30.3 ms | 19.9 ms | 117.9 ms | 33 fps |

> Real-time headroom: LOW at 5 Hz (200 ms/frame) has ~10× headroom; HIGH at 1 Hz (1000 ms/frame) has ~50× headroom. The online path's P95 spike comes from the LoRA update firing every 10 frames — still well under the frame budget.

## Model Specs

- Parameters: ~1.4M / model (d_model=128, d_state=16)
- Per-drone online LoRA: ~100K parameters
- Multi-hypothesis heads: +48K parameters (K=5)
- Input: 20 or 40 frames × 6 dimensions [pos, vel]; output: 20 frames × 3 dimensions displacement
- Hardware: RTX 3060 Laptop (6 GB), Windows 11, Python 3.10, CUDA 11.8

## Known Issues & Future Directions

- **LOW direction error** remains higher than HIGH; multi-hypothesis + dir_lora have improved it, but headroom remains.
- **Extreme turns (>150°)**: require architecture-level changes; gate_lora only mitigates.
- **Physics model 2-frame velocity seed**: a multi-frame least-squares seed was tested and verified negative (rejected; do not retry).
- **Global LoRA generalization**: honestly validated with a cross-flight split; held-out flight FDE +7.3% (the window-level +19.4% is optimistic). Architecture supports disabling global LoRA with a single flag for OOD scenarios.
- Test coverage: gates / online learning / SSM have regression tests + latency benchmark. The predictor soft-fusion and 40-frame expansion do not yet have dedicated regression tests.

## License

Research use only.
