# Pic-Results 图表说明

> 数据集：`UAV-Flow-pure/`（真实低速DJI无人机，5Hz）和 `SimCruise/`（仿真巡航高速，1Hz）。LOW + HIGH 均含多假设预测，K=4 有效头部（1 个死头已过滤）。3D 图 Z 轴已按 XY 比例缩放，避免小幅度 Z 误差被视觉放大。

---

## 一、LOW 多假设轨迹可视化 — 40帧模型 (visualize_low_40frame.py) ★ 最新

> 基于 **40帧模型** (`low_speed_6class_40frame.pth` + `low_multihead_K5_40frame.pth`)，
> 从长轨迹 (≥150帧) 自适应窗口中精选。死头自动过滤。取代旧的 20帧 `low_24_*` 系列。

### low40_multihyp_3d.png
**LOW 40帧 — 多假设 K=5 预测 3D 网格**
- 深灰线 = 历史 (40帧自适应步长)，绿线 = 真值，K 条彩色 = 假设头，最粗 = 最佳
- Z 轴已按 XY 比例缩放

### low40_multihyp_xy.png
**LOW 40帧 — 多假设 XY 俯视网格**

### low40_per_second_error.png
**LOW 40帧 — 每秒误差曲线：单假设 vs minFDE_5 (oracle)**

### low40_error_table.png
**LOW 40帧 — 逐样本 single/minADE/minFDE + 每秒误差表**

> 长轨迹上 single-FDE ≈ 0.70m，minFDE_5 ≈ 0.49m。

---

## 二、HIGH 多假设轨迹可视化 (visualize_trajectories.py)

### high_12_multihyp_3d.png
**HIGH 模型 (SimCruise) — 多假设 K=5 预测 (3D)**

- 深灰线 = 历史 20s（20帧@1Hz），绿线 = 地面真值
- 5 种颜色 = 5 个独立预测头，透明度 = 置信度
- 每 5s 时间标记点 (5s/10s/15s/20s)
- Z 轴已按 XY 比例缩放

### high_12_best_trajectory_3d.png
**HIGH 模型 — 最高置信度轨迹 (3D)**

- 只显示置信度最高的轨迹

### high_12_best_trajectory_xy.png
**HIGH 模型 — 最高置信度轨迹 (XY 俯视图)**

### high_12_persecond_error.png
**HIGH 模型 — 每 5s 独立误差柱状图 (12 样本)**

- 横轴 = 预测时间段 (0-5s / 5-10s / 10-15s / 15-20s)

### high_6_adapter_comparison.png
**HIGH 模型 — 单模型 vs 多假设对比 (6 样本 XY)**

- 橙色虚线 = 单模型预测，蓝色实线 = 多假设 K=5 最佳

---

## 三、多假设评估 (eval_multi_head.py)

### eval_multihypothesis.png
**多假设 (K=5) vs 单模型 — 6 合 1 对比面板**

1. FDE 分布叠加 — 单模型(红) vs 多假设(绿)
2. ADE 分布叠加
3. 逐意图 FDE 柱状图
4. DESCEND FDE 分布
5. 逐样本散点图（对角线以下=改善）
6. 全指标汇总表

### eval_multihypothesis_error_growth.png
**逐意图误差时间增长曲线 — 单模型 vs 多假设**

---

## 四、多假设轨迹 (visualize_multihyp.py)

### multihyp_trajectory_grid.png
**多假设 (K=5) 轨迹样本网格 — 16 个样本**

### multihyp_descend_deep.png
**DESCEND 样本深度分析 — 5 条假设全部可见**

### summary_table_updated.png
**更新版核心指标汇总表**

---

## 五、40帧后续工作汇总 (visualize_session2_summary.py) ★ 新

### session2_summary.png
**40帧后续实验 4 合 1 面板**

1. **全局 LoRA on 40帧** — base / +global / +direction 的 FDE 对比
2. **方向误差** — dir-LoRA 把方向误差从 13.8° 降到 12.1°
3. **多假设 + LoRA 叠加** — minFDE_5 及 base/+local/+global+local 叠加对比
4. **Gate-LoRA on 极端转弯** — >60° 子集的 cata% 和方向误差对比

### lora_40frame/ (子目录)
**per-drone LoRA on 40帧** — overview_3d/xy + detail_p1-6 (eval_lora_40frame.py 现场训练)

---

## 六、40帧 LoRA 结果速查 (session 2)

| 实验 | 指标 | base | 结果 | 改善 |
|:--|:--|:--:|:--:|:--:|
| 全局 LoRA (global_lora_40) | FDE (混合留出) | 0.826m | 0.703m | +14.9% |
| **方向 LoRA (dir_lora_40)** ★ | FDE | 0.826m | **0.665m** | **+19.4%** |
| 方向 LoRA | 方向误差 | 13.8° | **12.1°** | +12.3% |
| 多假设 K=5 (40帧) | minFDE_5 vs single | 0.874m | 0.598m | +31.6% |
| **LoRA 叠加** (全局+单机) | per-drone FDE | 0.322m(单机) | **0.290m** | +10.2% vs 单机, 13/19 胜 |
| Gate-LoRA 极端转弯 | cata (>60°子集) | 10.65% | **8.89%** | -1.76pp |

> **负结果**：HIGH 20→40 帧扩展无益 (FDE 11.4→17.5m，1Hz 巡航无输入瓶颈)；
> 物理模型多帧速度种子无益 (UAV-Flow 敏捷机动下最近2帧才是最佳)。

---

## 关键数据速查

### LOW 模型 (UAV-Flow, 真实DJI无人机, 5Hz)

| 指标 | 单模型 | 多假设 K=5 | 改善 |
|:--|:--:|:--:|:--:|
| ADE mean | 0.61m | **0.48m** | **+21.6%** |
| FDE mean | 1.39m | **0.94m** | **+32.3%** |
| FDE median | 1.15m | **0.74m** | **+35.2%** |
| FDE P95 | 3.45m | **2.31m** | **+33.2%** |
| 方向误差 mean | 24.5° | **17.5°** | **+28.5%** |
| 方向误差 median | 16.0° | **9.9°** | **+37.8%** |
| 灾难性失败 (>90°) | 3.35% | **2.94%** | **+12.5%** |
| STRAIGHT FDE | 1.21m | 0.83m | +31.1% |
| TURN_L FDE | 1.38m | 0.94m | +32.5% |
| TURN_R FDE | 1.55m | 1.01m | +34.5% |
| DESCEND FDE | 4.08m | 3.46m | +15.2% |

> 仅训练 48K 参数（总参数 3.5%），5 轮约 14 分钟。minFDE_K=5 测试集全量评估 (24,900 样本)。

### HIGH 模型 (SimCruise, 仿真巡航, 1Hz)

| 指标 | 单模型 | 多假设 K=5 | 改善 |
|:--|:--:|:--:|:--:|
| ADE mean | 1.71m | **0.93m** | **+46%** |
| FDE mean | 5.46m | **1.84m** | **+66%** |
| FDE P95 | 36.97m | **5.22m** | **+86%** |
| DESCEND FDE | 27.26m | **6.13m** | **+78%** |
| 方向误差 | 0.1° | — | — |

> 已清理 21 个旧图表至 `reviewtodelete/`（~15MB），仅保留热力图作为科研图表参考。
