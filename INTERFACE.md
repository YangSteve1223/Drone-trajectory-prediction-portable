# INTERFACE SPEC — EMAM Drone Trajectory Prediction

System-integration interface control document (ICD) for embedding the predictor into a
larger system. Covers the two public entry points, their exact I/O contracts, units,
coordinate/timing conventions, error handling, and performance envelope.

> Scope: this is the **integration contract**, not a tutorial. For architecture and
> results see `README.md`; for the multi-agent roadmap see `ROADMAP.md`.
> All shapes/fields below are verified against the current code.

---

## 1. Conventions (apply to both entry points)

| Item | Value |
|:--|:--|
| Input dtype | `torch.float32` |
| Position units | **meters** |
| Velocity units | **meters / second** |
| Input feature order | `[pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]` (6 dims) |
| Output | **future displacement** relative to the last history position, in meters |
| Coordinate frame | caller-defined but **consistent**; the model normalizes internally (pos÷100, vel÷10). Feed a metric frame (e.g. local ENU). Do not pre-normalize. |
| Batch | leading `B` dim; `DronePredictor` supports `B≥1`, `DeployedLowPredictor` is single-stream (`B=1`) |
| Device | auto-selects CUDA if available; override via `device=` |

**Frequencies (must match training):** LOW model expects **5 Hz** samples, HIGH model **1 Hz**.
Feeding the wrong rate silently degrades accuracy — resample upstream if needed.

**Intent classes:** LOW = 6 classes, HIGH = 4 classes, returned as raw **logits** (apply
softmax yourself). Only the HIGH `class 3 = DESCEND` semantic is fixed in code (used by Z
correction); other class-index meanings follow the training label set (`fix_labels.py` /
training data), not a hardcoded enum — treat indices as opaque unless you own the label map.

---

## 2. Entry point A — `DronePredictor` (general short-range)

`predictor.py :: DronePredictor` — dual-model soft fusion + Z-axis correction. Use for
general real-time inference.

### Constructor
```python
DronePredictor(threshold: float = 5.0,      # speed (m/s) fusion midpoint region
               device: str | None = None,    # 'cuda' | 'cpu' | None(auto)
               use_finetuned: bool = False)   # keep False (finetuned LOW overfits)
```

### predict
```python
predict(hist: torch.Tensor) -> dict
```
| | Spec |
|:--|:--|
| **Input** | `hist` shape **`(B, 20, 6)`** — 20 history frames |
| **Output** | `predictions (B, 20, 3)` future displacement (m) |
| | `intent_logits (B, 6)` LOW-model intent logits |
| | `speed (B,)` current speed (m/s) |
| | `route (B,)` list of `'LOW'` \| `'HIGH'` route per sample |

### Contract / errors
- Shape is **validated**: wrong `(B,20,6)` → raises `ValueError`.
- `NaN` in input → warns and replaces with 0 (does not crash). Callers should clean input upstream.
- Stateless: no memory between calls; safe to call concurrently on separate instances.

---

## 3. Entry point B — `DeployedLowPredictor` (long-trajectory + online learning)

`deploy.py :: DeployedLowPredictor` — 40-frame base + global LoRA + per-drone online LoRA,
behind three gates. Use for a persistently-flying low-speed drone that should personalize
over time. **Single-stream (B=1).**

### Constructor
```python
DeployedLowPredictor(device: str | None = None,
                     use_global: bool = True,          # enable shared global LoRA
                     global_max_speed: float = 4.0,     # speed gate (m/s): global off above this
                     online_min_frames: int = 60,       # length gate: personalize only after N frames
                     checkpoint_dir: str = 'weights/online_adapters',
                     accumulation_steps: int = 10)      # online update cadence
```

### predict
```python
predict(hist: torch.Tensor,
        drone_id: str | None = None,       # set -> enable per-drone online LoRA
        ground_truth: torch.Tensor = None, # (1,20,3) prev-step truth, feeds online learning
        frames_seen: int | None = None,    # total frames streamed by this drone (length gate)
        timestep: int = 0) -> dict
```
| | Spec |
|:--|:--|
| **Input** | `hist` shape **`(1, 40, 6)`** — 40 history frames |
| | `drone_id` — omit for plain inference; set to enable per-drone personalization |
| | `ground_truth (1,20,3)` — optional; supply to let the drone learn online |
| | `frames_seen` — if `None`, inferred from `hist` length (gate treated as open) |
| **Output** | `predictions (1, 20, 3)` future displacement (m) |
| | `speed (float)` current speed (m/s) |
| | `base (str)` `'global'` \| `'plain'` — which base was routed |
| | `global_active (bool)` global LoRA in effect this call |
| | `online_active (bool)` per-drone adaptation in effect |
| | `length_gate_open (bool)` frames_seen ≥ online_min_frames |
| | `updated (bool)` an online LoRA update fired this call |

### Three gates (behavior)
1. **Length gate** (`online_min_frames`): below threshold → plain 40-frame base, no LoRA.
2. **Global toggle** (`use_global`): master on/off for the shared global LoRA.
3. **Speed gate** (`global_max_speed`): global LoRA disabled above this speed (trained on 0–3 m/s).

### Session / state contract (important for integrators)
- **Stateful.** The per-drone adapter is bound to one base for the whole drone session; do
  **not** flip `use_global`/speed mid-session for the same `drone_id` (would wipe in-memory learning).
- Call `save()` to persist learned per-drone adapters to `checkpoint_dir`.
- One instance serves one stream. For multiple drones concurrently, either interleave calls
  with distinct `drone_id` (adapters are keyed by id) or run separate instances.
- Online learning triggers every `accumulation_steps` observations — expect a periodic
  latency spike on the update call (see §4).

---

## 4. Performance envelope (RTX 3060 Laptop, batch=1)

| Path | mean | p95 | throughput |
|:--|:--:|:--:|:--:|
| `DronePredictor.predict` (20f) | 35.3 ms | 44.2 ms | 28 fps |
| `DeployedLowPredictor.predict` (40f, inference) | 20.1 ms | 27.3 ms | 50 fps |
| `DeployedLowPredictor.predict` (40f, online) | 30.3 ms | 117.9 ms | 33 fps |

- Real-time headroom: LOW 5 Hz (200 ms budget) ~10×; HIGH 1 Hz (1000 ms budget) ~50×.
- The online path's p95 spike is the LoRA update every `accumulation_steps` frames — still far
  under the frame budget, but size your worst-case timing against **p95/p99**, not mean.
- Re-measure on target hardware with `python tests/benchmark_latency.py`.

---

## 5. Integration checklist

- [ ] Upstream resamples to 5 Hz (LOW) / 1 Hz (HIGH) before feeding history.
- [ ] Input is a consistent metric frame in meters & m/s, feature order `[x,y,z,vx,vy,vz]`.
- [ ] Consumer adds `predictions` to the **last history position** to get absolute future points.
- [ ] Consumer applies softmax to `intent_logits` if class probabilities are needed.
- [ ] NaN/dropout handling done upstream (predictor only zero-fills as a last resort).
- [ ] For `DeployedLowPredictor`: `drone_id` is stable per physical drone; `save()` called on shutdown.
- [ ] Worst-case timing budget checked against p95/p99, not mean.
- [ ] **fail-safe (recommended, not built-in):** if predicted step-to-step jump or uncertainty
      exceeds a sane bound, fall back to a constant-velocity extrapolation. Not currently in the
      library — the integrating system should own this safety layer.

---

## 6. Known limitations affecting integration

- No built-in confidence-based rejection / fail-safe (see checklist — caller's responsibility).
- HIGH model is 20-frame only (40-frame expansion was verified negative for 1 Hz cruise).
- Global LoRA is validated in-dataset (cross-flight +7.3% FDE); for out-of-distribution
  airspace, disable it (`use_global=False`) until re-validated.
- Single-agent only — no interaction/collision modeling (see `ROADMAP.md`).
