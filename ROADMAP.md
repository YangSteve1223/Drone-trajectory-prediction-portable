# ROADMAP — Interactive Multi-Agent / Multi-Drone Prediction

> Status: **planning only, not implemented.** This document is the design brief for the
> next research phase. It is written so a future engineer (or agent) can build the system
> incrementally on top of the current single-agent EMAM codebase without a rewrite.
>
> Current system = **single-agent**: predict one drone's future from its own history.
> Target system = **interaction-aware multi-agent**: predict N drones jointly, where each
> prediction is conditioned on the others.

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
  learning stack. Reused for **per-swarm** adaptation (see §4).

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

## 1. Problem definition & scene assumptions (do this first, on paper)

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

## 2. Data & scene representation (THE gating bottleneck)

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

## 3. Interaction module design (the core novelty)

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

## 4. Multi-agent decoding strategy

- **Marginal → Joint.** Start **marginal**: each agent decodes independently, but its features already
  carry neighbor context (simple to implement, reuses `ua_pgd` as-is per agent). Then advance to
  **joint**: predict scene-consistent futures so two predicted trajectories don't pass through each other.
- **Reuse the K=5 WTA head at scene level.** Extend `MultiHeadNeuralDecoder` so the K hypotheses are
  **joint** scene futures (K self-consistent scenes) rather than K independent per-agent guesses.
- **Collision-aware loss (new supervision unique to multi-agent):** add a min-inter-agent-distance
  penalty so predicted trajectories don't overlap. This is the signal single-agent training can't have.

---

## 5. Training strategy (maximize reuse of existing assets)

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

## 6. Evaluation & benchmarking (closes our current biggest gap vs SOTA)

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

## 7. Risks & priority

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

## 8. Minimum viable path (recommended first slice)

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

## 9. New files / changes a future agent will create

| File | Purpose |
|:--|:--|
| `utils/multi_agent_loader.py` | Scene tensors `(B,N,T,6)` + `mask` + `global_pos`; interaction-density stat |
| `emam_model/interaction.py` | `InteractionModule(mode='pool'|'gat'|'st')`, inserted encoder→decoder |
| `emam_model/model.py` (edit) | Add multi-agent axis: reshape `B*N` into encoder, call interaction, reshape back |
| loss module (edit `ua_pgd`/new) | Collision / min-inter-agent-distance penalty; joint-WTA option |
| `train/train_multi_agent.py` | Two-stage: freeze encoder → train interaction (+ optional full FT) |
| `eval/eval_multi_agent.py` | Joint ADE/FDE, minJointFDE@K=20, collision rate, ablations, public-set compare |

---

## 10. One-line summary

Reuse EMAM's per-agent Mamba encoder, insert a **distance-gated sparse graph-attention** interaction
layer on the encoded features, move decoding from marginal to joint with a collision-aware loss, keep the
"frozen base + LoRA increment + online personalization" training recipe (extending online learning from
per-drone to **per-swarm**), and finally validate on a public benchmark. **The make-or-break first step is
obtaining concurrent multi-drone data — treat §2 as the go/no-go gate.**
