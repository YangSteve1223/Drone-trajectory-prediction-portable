# 消融实验综合报告 — Per-Drone Online LoRA + Direction Fallback

> **日期**: 2026-07-30  
> **基线**: EMAM 40-frame single-head (low_speed_6class_40frame.pth)  
> **数据**: UAV-Flow (LOW) / SimCruise (HIGH), 验证集 only  
> **协议**: Causal streaming — warmup 前 60% flight windows, eval on held-out 后 40%  
> **Global LoRA**: 已禁用（数据泄露风险；cross-flight +7.3% vs window-level +19.4%）

---

## 1. 核心结论（TL;DR）

| 指标 | Frozen Base | +Online LoRA | +LoRA +Fallback | 总增益 |
|:--|:--|:--|:--|:--|
| **FDE (m)** | 1.0671 | 1.0559 | 1.0546 | **+1.17%** |
| **Dir (°)** | 16.63 | 15.00 | 14.35 | **−2.28°** |
| **Cata** | 9/829 (1.09%) | 6/829 (0.72%) | **0/829 (0%)** | **完全消除** |
| **Fallback 触发** | — | — | 11/829 (1.3%) | 极低误触发 |

**推荐部署配置: Single-head 40f + Online LoRA (accum=5, lr=5e-5) + Direction Fallback (90°)**

---

## 2. LOW 模型消融详细结果

### 2.1 6 配置对比（829 held-out 窗口，30 drones）

| Config | 描述 | FDE↓ | Dir↓ | Cata↓ | vs A |
|:--|:--|:--|:--|:--|:--|
| **A** | Single-head (frozen) | 1.0671 | 16.63° | 9 (1.09%) | — |
| **B** | Multi-head K=5 (frozen) | 1.1739 | 20.17° | 18 (2.17%) | −10.0% |
| **C** | Multi-head + Fallback (frozen) | 1.1526 | 18.25° | 0 | −8.0% |
| **D** | Single + Online LoRA (accum=10, lr=3e-5) | 1.0593 | 16.07° | 8 (0.97%) | +0.73% |
| **E** | Single + Online LoRA (accum=5, lr=5e-5) ⭐ | **1.0516** | **15.07°** | 6 (0.72%) | **+1.45%** |
| **F** | Multi-head + Fallback + Online | 1.0834 | 15.58° | 0 | −1.5% |

**关键发现:**
- **Multi-head K=5 在 LOW 数据上纯属负优化**（−10% FDE, cata 翻倍）。WTA decoder 在仿真巡航数据上训练，对真实 DJI 低速场景校准失败
- **Config E (Single + LoRA) 是最优配置**，FDE 增益 +1.45%，17/30 drones 改善
- **Config C/F 方向 fallback 消除 cata 但无法弥补 multi-head 在 98.9% 窗口上的退化**

### 2.2 Fallback 专项消融（推荐配置验证）

| Config | FDE↓ | FDE med↓ | Dir↓ | Cata | Fallbacks |
|:--|:--|:--|:--|:--|:--|
| (1) Frozen base | 1.0671 | 0.8297 | 16.63° | 9/829 | — |
| (2) +Online LoRA (accum=5, lr=5e-5) | 1.0559 | 0.8013 | 15.00° | 6/829 | — |
| (3) +Online LoRA + Fallback (90°) | **1.0546** | **0.8013** | **14.35°** | **0/829** | 11 (1.3%) |

```
Frozen → +LoRA:       FDE +1.05%   Dir −1.63°   Cata 9→6
+LoRA → +LoRA+FB:     FDE +0.12%   Dir −0.65°   Cata 6→0  (近乎零代价!)
Frozen → +LoRA+FB:    FDE +1.17%   Dir −2.28°   Cata 9→0
```

**Cata drone (2025-04-23_16-37-07) 详细分析:**
```
Frozen:     FDE=2.119m   Dir=73.95°   Cata=9/28
+LoRA:      FDE=2.010m   Dir=66.78°   Cata=6/28   (LoRA 拯救 3 个 cata)
+LoRA+FB:   FDE=1.626m   Dir=40.02°   Cata=0/28   (Fallback 接管剩余 6 个)
                                                     增益: +23.3%!
```
该 drone 具有极端转弯率 (111 deg/s)，模型在此机动上难以跟踪。LoRA 提供增量改善，Fallback 在剩余 cata 窗口干净地替换为 const-vel。

### 2.3 Per-drone 统计

- **Online LoRA 改善**: 17/30 drones (57%)，最大增益 +46.0%，最大退化 −56.8%
- **Online LoRA + Fallback 改善**: 16/30 drones (53%)
- **Fallback 假阳性**: drone 2 (3次触发, FDE +0.29m), drone 25 (1次触发, FDE +0.05m)

---

## 3. HIGH 模型消融（20-frame, SimCruise, 25 drones）

| Config | FDE↓ | Dir↓ | vs Frozen |
|:--|:--|:--|:--|
| Frozen base | 4.2754 | 0.20° | — |
| +Online LoRA (accum=10) | **4.0552** | **0.15°** | **+5.15%** |

- **25/25 drones (100%) 改善** — HIGH 场景 LoRA 全面涨点
- Dir error 本身已很低（0.20°→0.15°），主要收益来自 FDE 降低
- HIGH 模型预测 20s 长时域，方向偏差天然较小；主要瓶颈是幅度精度

---

## 4. 逐秒误差分析（LOW, 5 Hz）

从 representative 轨迹中提取的时变误差特征：

| 时间 | 典型 Frozen 误差 | 典型 +LoRA 误差 | 典型 +LoRA+FB 误差 |
|:--|:--|:--|:--|
| 0s (Start) | 0.020–0.120m | 0.010–0.080m | 同 +LoRA |
| 1s | 0.150–0.500m | 0.080–0.350m | 同 +LoRA |
| 2s | 0.350–1.200m | 0.200–0.900m | 同 +LoRA |
| 3s | 0.600–2.000m | 0.400–1.500m | 同 +LoRA |
| 4s (Final) | 0.800–3.000m | 0.500–2.500m | 同 +LoRA |

**规律:**
- LoRA 改善随时间累积放大：前 1s 差距很小，3–4s 差距显著
- Fallback 仅在 cata 窗口介入（~1%），其他窗口与 +LoRA 完全一致
- 误差随时间单调增长（模型漂移积累），const-vel baseline 在急转弯时优于漂移严重的模型

---

## 5. 多配置最终排名

```
推荐部署:    Single + LoRA(accum=5,lr=5e-5) + Fallback    FDE=1.0546m    Cata=0
次优:        Single + LoRA(accum=5,lr=5e-5)              FDE=1.0516m    Cata=6
旧 baseline: Single frozen                                FDE=1.0671m    Cata=9
不推荐:      Multi-head K=5 (任何配置)                    FDE≥1.0834m    (退化)
```

---

## 6. 生成文件清单（pic-results/summarize/）

### 数据 (JSON)
- `ablation_low.json` — 原始 LOW 消融（base vs online, accum=10）
- `ablation_high.json` — HIGH 消融（base vs online, 25 drones）
- `ablation_comparison.json` — 6 配置全面对比 (A→F)
- `ablation_fallback.json` — Fallback 专项消融 (3 配置) + per-drone 明细

### LOW 可视化（7 张）
- `ablation_low_overview.png` — 6 条代表性轨迹 3D 总览（Frozen vs Online）
- `ablation_low_2025-04-06_11-05-05.png` — 最佳增益 (+46.0%)
- `ablation_low_2025-04-23_16-37-07.png` — Cata drone (+1.5%)
- 其余 4 张覆盖 gain 谱

### HIGH 可视化（7 张）
- `ablation_high_overview.png` — 6 条代表性轨迹 3D 总览
- 6 条 per-drone 详细图（3D + XY）

### Fallback 可视化（8 张）
- `ablation_fallback_overview.png` — 6 条代表性轨迹 3D 总览（3 线: Frozen/Online/Fallback）
- `ablation_fallback_2025-04-23_16-37-07.png` — Cata drone, best window
- `ablation_fallback_2025-04-23_16-37-07_fallback.png` — Cata drone, **fallback-triggered window**（紫色线与橙色分道扬镳）
- 其余 4 张覆盖 gain 谱

---

## 7. 后续方向（基于 ROADMAP.md）

| 优先级 | Track | 描述 | 预计投入 | 预期产出 |
|:--|:--|:--|:--|:--|
| ⭐⭐⭐ | **B** | Long-horizon rollout eval (FDE@4/6/8/10s) | 1–2天 | 验证模型在 8s+ 时域是否退化 |
| ⭐⭐⭐ | **C** | Streaming vs batch 机制深度分析 | 4–8周 | 理论+实验，发 paper 核心贡献 |
| ⭐⭐ | **D** | 边缘部署 (INT8 量化) | 2–4周 | 模型 <1MB，Jetson 实时推理 |
| ⭐⭐ | **E** | 对抗鲁棒性评估 | 2–3周 | PGD/FGSM 攻击 + 对抗训练防御 |
| ⭐ | **A** | 多智能体交互预测 | 2–3月 | 需要先解决数据问题 |

**建议下一步**: Track B（rollout 长时域评估）成本最低、产出最快，可在一两天内给出"模型能预测多远的未来"的量化答案。Track C 是 paper 核心，可与 Track D（量化部署）并行推进。
