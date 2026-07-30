# ROADMAP — EMAM Next-Step Research & Engineering Tracks

> Status: **planning only, not implemented.** This document is the master design brief
> for the next research phase. It is written so a future engineer (or agent) can build
> each track incrementally on top of the current single-agent EMAM codebase without a
> rewrite.
>
> Current system = **single-agent**: predict one drone's future from its own history.
> Five tracks are described below — each independent, each starting from the same
> verified baseline (§0). They compose cleanly (e.g. Track B long-horizon + Track C
> streaming→batch + Track D edge-deploy is a coherent chain).

### Track overview

| Track | Topic | Type | Effort | Payoff |
|:--|:--|:--|:--|:--|
| **A** | Multi-agent interaction (per-swarmer) | Research | 2–3 months | New capability |
| **B** | Long-horizon prediction (>4s) | Research + Engineering | 3–6 weeks | Broader applicability |
| **C** | Streaming vs batch — hardcore deepening | Research (paper core) | 4–8 weeks | Publication-grade contribution |
| **D** | Edge deployment (quantization / distillation) | Engineering | 2–4 weeks | Real-world deployability |
| **E** | Adversarial robustness | Research | 4–6 weeks | Safety / security contribution |

All tracks below share the same **§0 baseline** — this section is the single source of truth
for which interfaces exist and how to reuse them. A future agent reading any track can start
from §0 and build only what that track describes.

---

## 0. Where we start from (current code, verified)

The single-agent pipeline (see `README.md`) is the foundation. Key interfaces to reuse:

- `emam_model/model.py :: TrajectoryPredictor.forward(history, ...)` — the pipeline is:
  ```
  history (B,T,6)
    → normalize (÷[100,100,100,10,10,10])
    → emam_se(h)              # (B,T,d_model)  Mamba SSM encoder
    → ia_dtp(encoded, h)      # intent logits + enhanced_features (B,T,d_model) + global_anchor
    → trigger (optional)
    → ua_pgd(...)             # decoder → predictions (B,20,3)
  ```
- `emam_model/emam_se.py :: EMamSE.forward(x) -> (B,T,d_model)` — **per-agent temporal encoder.
  This stays untouched and is reused as-is (frozen at first).**
- `emam_model/ua_pgd.py :: UncertaintyAwarePGD.forward(...)` and `.forward_multi_head(...)` /
  `.replace_with_multi_head(K)` — the decoder + K=5 multi-hypothesis head. Reused, lightly extended.
- `lora.py`, `online_learner.py`, `adapter_manager.py`, `online_config.py` — the LoRA + online
  learning stack. Per-drone online LoRA only (global LoRA disabled by default — data leakage
  concern). Reused for **per-swarm** adaptation (see §4).

**The single most important design principle:** the interaction module is inserted
**between the encoder and the decoder**, operating on `enhanced_features (B,T,d_model)`.
Encoder and decoder change minimally. Everything new lives in one new module + one new data loader.

**Tensor-shape change that drives everything:**
```
single-agent:  (B, T, 6)            → (B, 20, 3)
multi-agent:   (B, N, T, 6) + mask (B, N)  → (B, N, 20, 3)
```
`N` = number of drones in a scene (variable, padded + masked). Everything downstream is about
carrying that extra `N` axis through the pipeline correctly.

---

# TRACK A — Interactive Multi-Agent / Multi-Drone Prediction

## A.1 Problem definition & scene assumptions (do this first, on paper)

Drones differ fundamentally from pedestrians/vehicles — get this right before coding:

- **Interaction is sparse.** In open 3D airspace most drones fly independently; real interaction
  only occurs during approach / formation / avoidance. → Use a **distance-gated sparse graph**,
  not dense all-pairs attention.
- **3D + high speed.** The interaction radius must account for the z-axis AND relative velocity
  (two drones closing at 20 m/s "interact" from much farther than two hovering). Radius should be
  a function of relative position and closing speed, not a fixed 2D constant.
- **Two regimes, handle separately.** LOW (real low-speed: formation / inspection → cooperative)
  vs HIGH (cruise → conflict / avoidance). **Pick one regime and get the pipeline working end-to-end
  before generalizing.** Recommend starting with the simulated HIGH/SimCruise regime (easiest to
  synthesize concurrent multi-drone scenes with ground-truth interactions).

**Deliverable:** an "interaction definition" note — what counts as an interaction, how the gating
radius is computed, and whether the metric of interest is cooperation quality or collision avoidance.

---

## A.2 Data & scene representation (THE gating bottleneck)

This is the biggest blocker. Current data is **independent single-agent trajectories**. Multi-agent
needs **concurrent trajectories of multiple drones in the same airspace at the same timestamps.**

- **Data source (priority order):**
  1. Can `SimCruise` be regenerated / configured to emit concurrent multi-drone scenes? Simulation
     is by far the easiest route to ground-truth interactions.
  2. Does real `UAV-Flow` have any synchronized multi-drone logs? If not, real-data multi-agent is
     out of scope short-term; stay on simulation.
- **Scene tensor:** extend input to `(B, N, T, 6)` with a `mask (B, N)` for variable N / agents
  entering-leaving the scene.
- **Coordinate frames — the silent killer:** each agent is ego-normalized for its own encoder
  (reuse `dynamic_norm.py`), BUT keep a **separate un-normalized copy in the shared global frame** —
  interactions (relative position/velocity) MUST be computed in the shared frame, never in per-agent
  local frames. Mixing these up is the most likely source of subtle bugs. Unit-test the transforms.

**Deliverable:** a multi-agent data loader (new file, e.g. `utils/multi_agent_loader.py`) yielding
`{history:(B,N,T,6), future:(B,N,20,3), mask:(B,N), global_pos:(B,N,T,3)}`, plus an **interaction-density
statistic** — what fraction of frames actually contain close-neighbor interactions. This number decides
whether heavy interaction modeling is even worth it (see §6 risks).

---

## A.3 Interaction module design (the core novelty)

Keep the Mamba encoder **per-agent and independent** — fold the agent axis into the batch, run the
encoder, reshape back:
```python
B, N, T, C = history.shape
h = history.reshape(B*N, T, C)
enc = emam_se(h).reshape(B, N, T, d_model)   # per-agent temporal features, encoder reused frozen
```
Then insert an interaction layer on `enc`. Three candidates, increasing complexity:

1. **Social pooling / distance-weighted aggregation** (fastest baseline). Each agent aggregates
   neighbor features weighted by inverse distance. Simple, interpretable, enough for a first baseline.
2. **Graph attention (GAT / Transformer over the agent axis)** — *recommended main line.* Build a
   **distance-gated sparse graph** (edges only within the interaction radius), attend over agents.
   Matches current SOTA practice, handles variable N, learns "who influences whom."
3. **Spatio-temporal factorized** — time axis via our Mamba, agent axis via attention, interleaved.
   Most expressive, heaviest. Only pursue if (2) plateaus.

Design constraints:
- New module, e.g. `emam_model/interaction.py :: InteractionModule(mode='pool'|'gat'|'st')`, placed
  **after `ia_dtp`, before `ua_pgd`**, consuming `enhanced_features` + `global_pos`, returning
  interaction-augmented features of the same shape. IA-DTP and UA-PGD stay essentially unchanged.
- **Permutation-equivariant** over agents (attention satisfies this natively — do NOT bake in agent order).
- **Distance gating** both matches the physical fact that drone interaction is sparse AND caps the
  O(N²) attention cost for real-time feasibility.

---

## A.4 Multi-agent decoding strategy

- **Marginal → Joint.** Start **marginal**: each agent decodes independently, but its features already
  carry neighbor context (simple to implement, reuses `ua_pgd` as-is per agent). Then advance to
  **joint**: predict scene-consistent futures so two predicted trajectories don't pass through each other.
- **Reuse the K=5 WTA head at scene level.** Extend `MultiHeadNeuralDecoder` so the K hypotheses are
  **joint** scene futures (K self-consistent scenes) rather than K independent per-agent guesses.
- **Collision-aware loss (new supervision unique to multi-agent):** add a min-inter-agent-distance
  penalty so predicted trajectories don't overlap. This is the signal single-agent training can't have.

---

## A.5 Training strategy (maximize reuse of existing assets)

- **Two-stage (mirror the 40-frame expansion):** load single-agent 40-frame weights, **freeze the
  encoder**, train only the interaction module + light decoder fine-tune → verify the interaction layer
  actually adds value → only then consider full fine-tune.
- **LoRA continuation:** the interaction module can itself carry a global LoRA / per-scene LoRA,
  continuing the "stable base + LoRA increment" philosophy.
- **Online learning, new axis:** single-agent online LoRA learns *this drone's personality*; multi-agent
  extends naturally to **per-swarm LoRA** — learn *this formation's / this airspace's interaction pattern*.
  Reuse `online_learner.py`; just point the adapter targets at the interaction module. This is a genuine
  differentiator and a natural continuation of our online-learning story.

---

## A.6 Evaluation & benchmarking (closes our current biggest gap vs SOTA)

This directly addresses the "we have no public benchmark" weakness noted in the SOTA review.

- **Multi-agent metrics:** joint ADE/FDE, **minJointFDE@K with K=20** (align with the field's convention —
  our current K=5 undersells us in cross-paper comparison), **collision rate**, scene-level consistency.
- **Public dataset:** run on at least one public multi-agent benchmark (pedestrian ETH/UCY, or a vehicle
  set) and compare against 1–2 baselines (social pooling, pure Transformer). This upgrades "engineering
  result" → "publishable comparison."
- **Ablations:** interaction on/off; sparse-gated graph vs fully-connected; Mamba vs Transformer temporal
  encoder. These justify the architectural choices (esp. the SSM-vs-Transformer question we currently
  can't answer).

---

## A.7 Risks & priority

- **#1 risk — data.** No concurrent multi-drone data ⇒ the whole line is blocked. **§2 is a go/no-go gate:**
  cheaply confirm we can obtain/simulate enough interaction-rich scenes before investing in §3+.
- **Interaction density caps the payoff.** If close interactions are rare in real scenes, the marginal
  benefit of heavy graph modeling is small — distance-gated pooling may suffice. Let the §2 statistic decide.
- **O(N²) attention cost.** Distance gating + sparse edges mitigate it; re-benchmark latency (reuse
  `tests/benchmark_latency.py`) against the 5 Hz (200 ms) / 1 Hz (1000 ms) frame budgets after each change.
- **Coordinate-frame mixing** (see §2) — keep ego-normalized features separate from global-frame geometry;
  unit-test the transforms.
- **Joint decoding is harder to train** than marginal — keep marginal as a fallback; only push to joint
  once the collision-loss version is stable.

---

## A.8 Minimum viable path (recommended first slice)

The smallest end-to-end chain that produces a first "interaction-aware" result without touching the
proven single-agent core:

1. `SimCruise` → synthesize multi-drone scenes; compute the interaction-density statistic (§2 gate).
2. Freeze existing 40-frame encoder; reshape `(B,N,T,6)` through it (§3).
3. Add a **distance-gated GAT** interaction layer on the encoded features (§3, option 2).
4. **Marginal** decoding via existing `ua_pgd` + a **collision-aware loss** (§4).
5. Evaluate joint ADE/FDE + collision rate vs a **social-pooling baseline** (§6).

If the interaction layer beats the no-interaction baseline on collision rate and joint FDE → escalate to
joint decoding (§4), per-swarm online LoRA (§5), and public-benchmark comparison (§6).

---

## A.9 New files / changes a future agent will create

| File | Purpose |
|:--|:--|
| `utils/multi_agent_loader.py` | Scene tensors `(B,N,T,6)` + `mask` + `global_pos`; interaction-density stat |
| `emam_model/interaction.py` | `InteractionModule(mode='pool'|'gat'|'st')`, inserted encoder→decoder |
| `emam_model/model.py` (edit) | Add multi-agent axis: reshape `B*N` into encoder, call interaction, reshape back |
| loss module (edit `ua_pgd`/new) | Collision / min-inter-agent-distance penalty; joint-WTA option |
| `train/train_multi_agent.py` | Two-stage: freeze encoder → train interaction (+ optional full FT) |
| `eval/eval_multi_agent.py` | Joint ADE/FDE, minJointFDE@K=20, collision rate, ablations, public-set compare |

---

## A.10 One-line summary

Reuse EMAM's per-agent Mamba encoder, insert a **distance-gated sparse graph-attention** interaction
layer on the encoded features, move decoding from marginal to joint with a collision-aware loss, keep the
"frozen base + LoRA increment + online personalization" training recipe (extending online learning from
per-drone to **per-swarm**), and finally validate on a public benchmark. **The make-or-break first step is
obtaining concurrent multi-drone data — treat §A.2 as the go/no-go gate.**

---

# TRACK B — Long-Horizon Prediction (>4 s)

> **Goal:** extend prediction from 20 frames (4 s LOW / 20 s HIGH) to 40+ frames
> without retraining the entire model from scratch. Trained once at the current
> horizon, but deployable for 8–10 s lookahead via autoregressive rollout, a
> hierarchical predictor, or a lightweight horizon-extension head.

## B.1 Why this matters

The current 40-frame expansion solved the *history* bottleneck (input side). But
4 s prediction is short for many applications: collision avoidance needs 8–10 s
lookahead; air-traffic separation standards assume the controller sees the next
10 s of intent. Extending the prediction horizon without rebuilding the model is
a high-impact, low-disruption addition.

The challenge is different from the LOW 40-frame expansion — that was about
exposing more input context. Long *prediction* horizon compounds errors: small
step-by-step drift → large end-point error. The physics model (`KinematicPhysicsModel`
in `ua_pgd.py`) provides a velocity-seeded baseline, but when the neural decoder
drifts, the kinematic correction alone can't pull it back at long horizons.

## B.2 Three approaches, pick by ceiling

**Approach 1: Autoregressive rollout (fastest, already prototyped)**
`viz/rollout.py` already implements a sliding-window autoregressive scheme:
feed the current prediction back as the next history. This works immediately — zero
retraining. Limitation: error accumulates with each rollout, and the 20-frame model
was never trained to consume self-predicted histories (distribution shift).

- **Deliverable:** a standalone `eval/eval_long_horizon.py` that runs rollout on held-out
  flights and reports FDE@4s, FDE@6s, FDE@8s, FDE@10s (LOW 5 Hz: 20/30/40/50 frames).
  Compare against the constant-velocity baseline (already in `model.py`) at each horizon
  — the baseline is literally competitive at longer prediction horizons because it doesn't
  drift. If rollout can't beat const-vel at 8s+, that's a finding in itself (keep for §C).
- **Code reuse:** `viz/rollout.py :: autoregressive_rollout(model, hist, n_rollouts, dt, device)`
  + `predictor.py :: DronePredictor.make_long_windows()` for data slicing.
- **Effort:** 1–2 days (script exists, just needs a proper evaluation harness).

**Approach 2: Hierarchical / coarse-fine predictor (recommended if rollout plateaus)**
Train a *lightweight horizon extender* on top of the frozen 40-frame encoder. The
extender predicts at a coarser temporal resolution (e.g. every 0.4 s vs 0.2 s) and
the existing fine decoder fills in between. All that changes: a new output head that
predicts `(B, 5, 3)` at stride 4 instead of `(B, 20, 3)` at stride 1 → covers 8 s.
Jointly train with the existing decoder via a shared feature extractor.

- **Code changes:** new subclass `HierarchicalDecoder` in `ua_pgd.py` (or a new file
  `emam_model/hierarchical_decoder.py`) that consumes `enhanced_features` + `global_anchor`,
  outputs `(B, T_coarse, 3)` with `T_coarse` configurable. Train end-to-end with the
  existing fine decoder frozen.
- **Loss:** standard Huber at coarse time-steps + a consistency penalty between coarse
  and fine at shared time points.
- **Effort:** 2–4 weeks (new module + train/eval scripts).

**Approach 3: Diffusion-based trajectory generation (ambitious)**
Frame long-horizon prediction as iterative denoising — the model starts from noise
and refines a full long trajectory conditioned on history. Expressive but heavy; only
pursue if (2) plateaus and there's GPU budget. Not detailed further in this document.

## B.3 Evaluation protocol

- Report **ADE / FDE at multiple horizons** (4s / 6s / 8s / 10s for LOW; 20s / 40s / 60s for HIGH).
- **Baseline:** constant-velocity extrapolation from the last 2-frame velocity (already
  computed in `KinematicPhysicsModel` and exposed via `force_predict`).
- **Upper-bound oracle:** feed the next 4s of real history as input — the gap between this
  and the long-horizon prediction measures how much of the error is information-limited
  vs model-limited. The current model closes ~90% of the gap at 4s; at 8s it might close 50%.
  That number itself is a publishable finding.
- **Uncertainty calibration:** extend the logvar head to the longer horizon and check
  whether uncertainty grows monotonically (it should — if it doesn't, the model is overconfident
  at long range).

## B.4 Risks

- Autoregressive compounding is the main limit. Approach 1 is a cheap gate: if rollout
  FDE@8s is within 2× the const-vel baseline, the model generalizes to longer horizons
  surprisingly well → publish that. If not → move to approach 2.
- HIGH model at 1 Hz is already predicting 20 s — the bottleneck is different (not horizon,
  but data sparsity). Skip long-horizon for HIGH (consistent with the 40-frame expansion
  result that was negative for HIGH).

## B.5 New files / changes

| File | Purpose |
|:--|:--|
| `eval/eval_long_horizon.py` | Rollout-based multi-horizon eval (FDE@4/6/8/10s, const-vel baseline, uncertainty growth) |
| `emam_model/hierarchical_decoder.py` | Coarse-fine hierarchical predictor (approach 2) |
| `train/train_hierarchical.py` | Two-stage: freeze encoder → train coarse head + fine consistency |
| `viz/visualize_long_horizon.py` | Long-horizon trajectory plots with uncertainty cones |

---

# TRACK C — Streaming vs Batch Hardcore Deepening

> **Goal:** turn the existing "streaming +2.4% / batch −202%" finding into a
> publication-grade contribution with mechanistic analysis, generalization
> across data regimes, and theoretical intuition.

## C.1 Why this matters — the scientific gap

The current finding (`eval_online_vs_offline.py`, 20 drones, single-flight warmup)
is the strongest *discovery* in this project: under causal deployment constraints,
offline batch fine-tuning catastrophically overfits while streaming incremental
adaptation gives a net gain. This is an **anti-intuitive result** — the naive
expectation is that "more data + full gradient descent = better." But it's
currently a single experiment. To become a hardcore contribution it needs:

1. **Generality** — does this hold across different data amounts, model sizes,
   optimizer settings?
2. **Mechanistic understanding** — *why* does batch fine-tuning collapse?
3. **Theoretical framing** — can we connect this to known phenomena (catastrophic
   forgetting, sharp minima, distribution shift under causal constraints)?

## C.2 Multi-regime sweep (breadth)

Design a grid of experiments, each run on `eval_online_vs_offline.py`-style causal splits:

| Dimension | Values | Reason |
|:--|:--|:--|
| Warmup fraction | 10%, 30%, 50%, 70% | Small warmup = batch has less data to overfit to |
| Drone count | 5, 10, 20, 30 | Sample-size dependency |
| LR sweep (batch) | 1e-5 to 1e-3 (5 values) | Is there a sweet spot where batch works? |
| LR sweep (streaming) | 1e-6 to 1e-4 (5 values) | How sensitive is streaming to LR? |
| Accumulation steps | 5, 10, 20, 50 | Update frequency vs stability trade-off |
| LoRA rank | r=4, 8, 16, 24, 48 | Model capacity vs overfit risk |

**Expected deliverable:** a phase diagram showing the region where streaming > batch.
The shape of this region (e.g. batch only wins at very large warmup + very small LR)
itself tells a story.

## C.3 Mechanistic analysis (depth)

Go beyond "batch overfits" to *why*:

1. **Weight-distance trajectory:** at each update, record the L2 distance of the
   batch/streaming LoRA weights from both (a) the zero-init and (b) one epoch ago.
   Hypothesis: batch takes larger steps early → overshoots into a sharp basin → validation
   collapses; streaming's small, frequent steps stay near the flat basin.

2. **Gradient coherence:** compute the average cosine similarity between gradients
   of consecutive mini-batches. Hypothesis: streaming sees temporally coherent gradients
   (adjacent windows from the same flight), which acts as implicit regularization; batch
   shuffles away the temporal structure and fights itself.

3. **Forgetting curve:** for the offline batch setting, measure FDE on the *training*
   warmup windows at each epoch. If batch training loss drops while validation rises,
   it's classic overfitting. If batch *never* converges on the training set (loss stays
   high), the problem is optimization failure, not overfitting — a different story.

4. **Replay buffer ablation:** remove the replay buffer from streaming → does it
   collapse toward batch? If yes, replay is the key ingredient; if streaming without
   replay still beats batch, the update cadence (gentle, incremental) is the key
   ingredient.

## C.4 Theoretical scaffolding (optional, high-impact)

Connect to known frameworks:
- **Continual learning / catastrophic forgetting (Kirkpatrick et al., 2017):** streaming
  as a form of elastic weight consolidation — small updates near the base, replay as
  rehearsal.
- **Sharpness-aware minimization (Foret et al., 2020):** hypothesis that streaming finds
  flatter minima → test with Hessian eigenvalue spectrum at convergence.
- **Online convex optimization:** can we prove that under causal distribution shift,
  streaming SGD with replay converges to a neighborhood of the population optimum while
  batch SGD (trained on warmup-only, evaluated on held-out from a *different* distribution)
  cannot? A toy linear model analogue would strengthen the empirical result massively.

## C.5 Existing code to reuse

| File | What it already does |
|:--|:--|
| `eval/eval_online_vs_offline.py` | Streaming vs batch comparison on causal data — the baseline harness |
| `online_learner.py` | `OnlineLearnerConfig` (lr, accum, replay_ratio, l2, cusum), `_update()`, replay buffer |
| `adapter_manager.py` | Per-drone adapter activation/deactivation, checkpoint save/load |
| `eval/eval_online_learning.py` | Single-flight streaming eval (baseline for streaming-only sweeps) |
| `eval/eval_online_learning_high.py` | HIGH streaming eval — expand the sweep to HIGH too |

## C.6 Deliverables

**Minimum:** multi-regime sweep (grid of 50+ experiments), phase diagram, 2–3
mechanistic analyses (gradient coherence + weight trajectory + forgetting curve).
**Target:** above + at least one theoretical connection with empirical evidence
(e.g. gradient-coherence plot showing streaming's implicit regularization).

## C.7 New files / changes

| File | Purpose |
|:--|:--|
| `eval/eval_streaming_sweep.py` | Grid-search harness for warmup/LR/accum/rank sweep |
| `analysis/mechanism_gradient.py` | Gradient coherence, weight trajectory, forgetting curve |
| `analysis/phase_diagram.py` | Phase diagram visualization: streaming vs batch win regions |
| `analysis/toy_linear_model.py` | Theoretical intuition: streaming vs batch under causal shift |

---

# TRACK D — Edge Deployment (Quantization & Distillation)

> **Goal:** shrink the model from ~5.2 MB FP32 / 1.37M params (~50 fps on RTX 3060)
> to under 1 MB INT8 / ~50 fps on a Jetson Nano or similar edge platform, without
> losing more than 10% of FDE accuracy.

## D.1 Why this matters — the hardware gap

The current system runs comfortably on a laptop RTX 3060 (6 GB, ~20 ms inference).
Real drones carry embedded platforms: NVIDIA Jetson Orin Nano (~40 TOPS INT8),
Raspberry Pi + Hailo, or STM32-class MCUs. The gap from 5.2 MB FP32 to an embedded
budget is ~5–10× in size and ~10–100× in computational throughput.

The 1.37M-parameter model is already *small* by deep-learning standards — this is
an advantage. Many trajectory models are 10–50M parameters and can't even consider
edge deployment. EMAM can plausibly hit an embedded chip with standard tooling.

## D.2 Quantization (first pass, highest leverage)

PyTorch provides `torch.quantization` (static post-training INT8). The Mamba SSM
ops (`SelectiveSSM` in `emam_se.py`) use element-wise ops that generally quantize
well. The physics model (`KinematicPhysicsModel`) is a closed-form numerical integrator
— it doesn't need quantization and stays in FP32.

**Approach:**
1. Fuse `Conv1d + SiLU/GRLU` in `MambaBlock` into quantizable modules. This may
   require small edits to `emam_se.py` — the current custom activations (`SiLU`,
   `GRLU`) need `torch.ao.quantization` stubs or torch.export rewriting.
2. Calibrate on 500 random windows from the training set (activation range observation).
3. Measure FDE change on the held-out test set. Target: <5% FDE drop.
4. Export via `torch.jit.trace` or `torch.export` to a standalone artifact.

**Bottleneck:** the selective SSM scan (`_selective_ssm_scan`) is a Python loop
(no `mamba-ssm` installed). For edge inference this MUST be replaced with a
vectorized or compiled version — Triton (if the edge chip supports it) or a
precomputed convolution approximation for fixed-length T=40.

**Deliverable:** `deploy_quantized.py` — loads the base model, quantizes, exports
a `model_int8.pt` artifact, and benchmarks FDE + latency on the target chip (or
a simulated embedded environment).

## D.3 Knowledge distillation (if quantization alone isn't enough)

Train a smaller student (e.g. d_model=64 instead of 128, or 1 Mamba layer instead
of 2) to mimic the frozen 40-frame teacher. The student sees the same input, the
teacher provides a soft target (prediction distributions), plus standard Huber on
ground truth.

**Key trick from our existing code:** the base model's `force_predict=True` path
outputs `predictions + uncertainty`. The student can be trained to match both
(uncertainty distillation), giving the edge model calibrated confidence for free.

- **Student candidate:** `TrajectoryPredictor(d_model=64, emam_n_layers=1)` —
  ~350K params, ~1.4 MB FP32, target <0.5 MB INT8.
- **Loss:** MSE(student_pred, teacher_pred) + KL(student_logvar, teacher_logvar)
  + 0.3 * MSE(student_pred, ground_truth). The weighted ground-truth term prevents
  the student from learning the teacher's errors.

## D.4 Evaluation protocol (share with §B & §C where applicable)

- **Accuracy:** FDE / ADE / direction error change vs the full FP32 model on the
  same held-out test set.
- **Latency:** re-run `tests/benchmark_latency.py` on the quantized/distilled model.
  Target <50 ms on the target edge platform at batch=1.
- **Memory:** peak GPU/CPU memory during inference, total model file size.
- **Throughput:** frames/second on the edge platform.

## D.5 New files / changes

| File | Purpose |
|:--|:--|
| `deploy_quantized.py` | PTQ INT8 calibration → export → benchmark |
| `train/train_distillation.py` | Teacher-student distillation (d_model=64, 1-layer) |
| `tests/benchmark_edge.py` | Edge-oriented latency + memory benchmark (TorchScript/ORT) |
| `emam_se.py` (edit) | Fuse Conv+Activation for quantizable graph, optional scan kernel for edge |

---

# TRACK E — Adversarial Robustness

> **Goal:** quantify and improve the model's resilience to input perturbations
> that an attacker could inject — GPS spoofing, sensor noise, adversarial
> position/velocity shifts — and provide a certified robustness baseline.

## E.1 Why this matters — the security gap

UAV trajectory prediction is a safety-critical component. If an adversary can
cause the predictor to output a wrong future (e.g. predict a straight path while
the drone is turning), downstream collision avoidance / air traffic management
makes wrong decisions. The current model has **zero adversarial evaluation** —
no one knows whether it's trivially foolable.

This track produces both: (a) a vulnerability assessment (how easy is it to
fool the model?), and (b) a mitigation strategy (adversarial training or
certified smoothing). The combination is publishable in security/robotics venues.

## E.2 Threat model (start narrow, expand if findings justify it)

**Entry point:** the model receives a history tensor `(B, 40, 6)`. The attacker
can perturb this tensor before inference — real-world analogues: GPS offset,
velocity sensor bias, adversarial RF injection into the navigation filter.

**Attack budget (L-infinity):** bounded perturbation per dimension:
- Position: ±2 m per frame (GPS spoofing)
- Velocity: ±1 m/s per frame (sensor bias)
- Combined: all 6 dims within a joint L-infinity ball

These are conservative — real GPS spoofing can inject larger offsets, but the
model has internal normalization (÷100 for position), so a 2 m perturbation is
0.02 in normalized space, small enough to potentially pass unnoticed.

## E.3 White-box attack (PGD / FGSM)

Projected Gradient Descent: starting from a clean history, take gradient steps
to maximize the FDE of the prediction, projecting back into the L-infinity ball
after each step.

- **Attack loss:** `F.smooth_l1_loss(pred, ground_truth, beta=0.2)` — the
  attacker's goal is to maximize this.
- **Implementation:** wrap `model.forward(perturbed_history, force_predict=True)`
  in a PGD loop. Model is in eval mode but gradients flow through the encoder
  and decoder.
- **Evaluation:** report FDE degradation vs clean input, at multiple attack
  budgets (1m/2m/4m position, 0.5/1.0/2.0 m/s velocity).

**Expected finding:** the model is likely vulnerable — the Mamba encoder has
no built-in robustness, and the physics model amplifies errors (a small velocity
perturbation propagates via the kinematic integrator).

## E.4 Black-box attack (query-based)

If the white-box attack succeeds (which it likely will), evaluate practical
attackability: an attacker without access to model gradients. Use:
- **Random search:** sample perturbations uniformly within the budget, pick the
  one that maximizes FDE.
- **Natural corruptions:** add Gaussian noise, frame-drop, time-warp to simulate
  sensor degradation (which also serves as a non-adversarial robustness baseline).

## E.5 Defense strategies (choose one, both if resources allow)

**Option 1: Adversarial training (standard, effective at the cost of clean accuracy)**
At each training step, generate an adversarial example via 1-step FGSM, and train
on the perturbed input with the clean target. This is standard and works, but
typically costs 2–5% clean FDE. Given our LoRA-based training, add a
`--adversarial` flag to `train_lora_direction.py` that perturbs the history
before the forward pass.

**Option 2: Certified smoothing (harder, more publishable)**
Apply randomized smoothing: add Gaussian noise to the input at inference time,
take the majority-vote prediction over N noisy samples. The resulting prediction
has a provable L2 robustness certificate. This is nontrivial for a regression
model — the standard smoothing theory is for classifiers. Adapting it to trajectory
regression (smooth over the *output displacement*, not a class label) would be
a novel methodological contribution.

**Recommendation:** start with adversarial training (1 week to implement + evaluate);
pursue certified smoothing only if the white-box vulnerability is severe and the
clean-accuracy cost of adversarial training is unacceptable.

## E.6 Evaluation protocol

- **Clean performance:** FDE/ADE/direction error on clean held-out test set.
- **Under attack:** same metrics, white-box PGD (10/20/40 steps) at multiple budgets.
- **Under defense:** same metrics after adversarial training or certified smoothing.
- **Baseline:** the constant-velocity model — how much does an attacker gain vs
  the simplest possible predictor? If attacking EMAM degrades it to const-vel level
  but not below, the attack is successfully "flooring" the predictor. If EMAM under
  attack is *worse* than const-vel, the attack is actively misleading the model.

## E.7 New files / changes

| File | Purpose |
|:--|:--|
| `robustness/adversarial_attack.py` | PGD, FGSM, random-search attacks on the predictor |
| `robustness/adversarial_defense.py` | Adversarial training loop, certified smoothing wrapper |
| `eval/eval_robustness.py` | Composite eval: clean / under-attack / under-defense at multiple budgets |
| `train/train_lora_direction.py` (edit) | Add `--adversarial` flag for adversarial LoRA training |

---

# Cross-Track Dependencies & Composeability

Tracks are designed to be pursued independently, but they compose cleanly:

- **B (long-horizon) + C (streaming→batch deepening):** long-horizon rollout provides
  more frames for streaming personalization, and the mechanistic analysis from C
  explains *why* streaming helps at long horizons (drift compensation).
- **C (streaming→batch) + D (edge deployment):** the streaming personalization runs on
  the edge after the quantized model is deployed — a full "deploy-on-edge + personalize-on-edge"
  pipeline.
- **D (edge) + B (long-horizon) + C (streaming):** a quantized model that predicts long-horizon
  and personalizes per-drone while running on an embedded chip — this is the end-to-end
  deployed vision, and it's genuinely novel as a complete system.
- **E (adversarial) is orthogonal** — add it at any point; the clean → under-attack → under-defense
  eval is the same regardless of whether the underlying model is long-horizon or edge-deployed.

**Recommended first parallel batch:** Track B (cheap rollout eval) + Track D (quantization).
Both are fast wins that produce concrete deliverables. Track C runs in parallel as the
research core (more experiments, longer timeline). Track E can start anytime — the attack
code is independent of all other changes.
