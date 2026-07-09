# Pic-Results 图表说明

> 数据集：`UAV-Flow-pure/`（真实低速DJI无人机）和 `SimCruise/`（仿真巡航高速）。LOW + HIGH 均含多假设预测 (K=5)。

---

## 一、LOW 多假设轨迹可视化 (visualize_low_multihyp.py) ★ 最新

> 从测试集 150+ 样本中多维度评分（精度/平滑度/多样性/置信度分布），精选 12 组。

### low_12_multihyp_3d.png
**LOW 模型 (UAV-Flow) — 多假设 K=5 预测 (3D)**

- 深灰线 = 历史 4s（20帧@5Hz），绿线 = 地面真值
- **5 种颜色 = 5 个独立预测头**，透明度 = 置信度（越不透明越可信）
- 绿色圆点/亮绿方块 = 每秒(1s/2s/3s/4s)时间标记点
- 标题标注意图、速度、minFDE、**每秒独立误差**

### low_12_best_trajectory_3d.png
**LOW 模型 — 最高置信度轨迹 (3D)**

- 只显示置信度最高的那一条轨迹（亮绿虚线）
- 深灰=历史 绿色=真值 亮绿虚线=最佳预测
- 标注 Z 轴误差 + 每秒独立误差

### low_12_best_trajectory_xy.png
**LOW 模型 — 最高置信度轨迹 (XY 俯视图)**

- 同上，从正上方俯视
- 转弯样本的弯道贴合度直观可见
- 标注 Z 轴误差

### low_12_persecond_error.png
**LOW 模型 — 每秒独立误差柱状图 (12 样本)**

- 横轴 = 预测时间段 (0-1s / 1-2s / 2-3s / 3-4s)
- 每个样本一组柱，**每段时间独立计算 L2 误差**

### low_4_multihyp_xy_grid.png
**LOW 模型 — 4 个最佳样本多假设 XY 网格**

- 4 个最高分样本，展示全部 K=5 条假设轨迹
- 5 种颜色 + 置信度标注

---

## 二、HIGH 多假设轨迹可视化 (visualize_trajectories.py)

### high_12_multihyp_3d.png
**HIGH 模型 (SimCruise) — 多假设 K=5 预测 (3D)**

- 深灰线 = 历史 20s（20帧@1Hz），绿线 = 地面真值
- 5 种颜色 = 5 个独立预测头，透明度 = 置信度
- 每 5s 时间标记点 (5s/10s/15s/20s)

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

## 五、其他

### 01_error_heatmap.png
**逐步预测误差热力图**

- 横轴 = 预测时间，纵轴 = 样本（按终点误差排序）
- 颜色 = L2 误差（米），越红越大

### rollout_low.png / rollout_high.png
**自回归外推** — LOW 4s→12s, HIGH 20s→40s

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
