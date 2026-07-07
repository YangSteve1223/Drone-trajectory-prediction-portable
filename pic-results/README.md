# Pic-Results 图表说明

> 所有图表使用 `constrained_layout` 生成，避免文字重叠。数据集：`UAV-Flow-pure/`（真实低速DJI无人机）和 `SimCruise/`（仿真巡航高速）。含多假设预测 (K=5) 完整对比评估。

---

## 一、轨迹可视化 (visualize_trajectories.py) ★ 最新

> 从测试集 80+ 样本中多维度评分（精度/平滑度/多样性/代表性），精选 12 组。

### low_12_trajectories_3d.png
**LOW 模型 (UAV-Flow) — 12 个最佳预测样本 (3D)**

- 深灰线 = 历史 4s（20帧@5Hz），绿线 = 地面真值，橙虚线 = 模型预测
- 绿色圆点/橙色方块 = 每秒(1s/2s/3s/4s)标记点
- 标题标注意图类别、速度、**每秒独立误差（不做时间平均）**
- 12 个样本覆盖 STRAIGHT/TURN_L/TURN_R/HOVER 四类意图

### low_12_trajectories_xy.png
**LOW 模型 — 12 个最佳预测样本 (XY 俯视图)**

- 同上数据，从正上方俯视
- 转弯样本的弯道贴合度直观可见

### low_12_persecond_error.png
**LOW 模型 — 每秒独立误差柱状图 (12 样本)**

- 横轴 = 预测时间段 (0-1s / 1-2s / 2-3s / 3-4s)
- 每个样本一组柱，**每段时间独立计算 L2 误差，不做跨段平均**
- 看误差是否随时间累积增长

### high_12_multihyp_3d.png
**HIGH 模型 — 多假设 K=5 预测 (3D)**

- 深灰线 = 历史 20s（20帧@1Hz），绿线 = 地面真值
- **5 种颜色 = 5 个独立预测头**，透明度 = 置信度（越不透明越可信）
- 标题标注意图、minFDE、**每 5 秒独立误差**
- 绿色圆点/亮绿方块 = 每 5s 标记点

### high_12_best_trajectory_3d.png
**HIGH 模型 — 最高置信度轨迹 (3D)**

- 只显示置信度最高的那一条轨迹（亮绿虚线）
- 简洁清晰，适合展示模型最佳表现

### high_12_best_trajectory_xy.png
**HIGH 模型 — 最高置信度轨迹 (XY 俯视图)**

- 同上，从俯视角度验证

### high_12_persecond_error.png
**HIGH 模型 — 每 5s 独立误差柱状图 (12 样本)**

- 横轴 = 预测时间段 (0-5s / 5-10s / 10-15s / 15-20s)
- 每个样本一组柱，**每段时间独立计算**

### high_6_adapter_comparison.png
**HIGH 模型 — 单模型 vs 多假设对比 (6 样本 XY)**

- 橙色虚线 = 单模型预测，蓝色实线 = 多假设 K=5 最佳
- 标题标注 FDE 改善百分比
- **直观展示多假设带来的精度提升**

---

## 二、Context Adapter 对比 (run_all.py，保留参考)

### 03_high_before_3d.png / 04_high_after_3d.png / 05_high_combined_3d.png / 06_high_combined_xy.png
**HIGH 模型 Context Adapter 前后对比**

- Context Adapter 是之前的创新点（60帧上下文注入解码器），+88-91%
- 与多假设是互补技术，此处保留供参考

---

## 三、科学评估图表 (visualize_final.py)

### 01_error_heatmap.png
**逐步预测误差热力图**

- 横轴 = 预测时间（LOW: 0→4s, HIGH: 0→20s）
- 纵轴 = 样本（按终点误差从小到大排序）
- 颜色 = L2 误差（米），越红越大
- 白色竖虚线 = 25%/50%/75% 时间标记
- **红色集中在右上角 = 远期预测误差更大，符合预期**
- **HIGH 图右侧突然变红 = DESCEND 样本在后期发散**

### 02_intent_confusion.png
**意图混淆矩阵**

- 行 = 真实意图标签，列 = 模型预测意图
- 对角线 = 预测正确，越亮越好
- 每个格子显示：样本数（占比%）
- 标题标注整体准确率
- **LOW 的 STRAIGHT→TURN 混淆已被标签修正消除**

### 03_error_by_speed.png
**预测误差 vs 飞行速度**

- 左图：散点图，按真实意图着色
- 右图：分箱均值 ± 1σ 趋势线
- **如果高速段误差显著更大 = 模型泛化不足**

### 04_error_by_intent.png
**逐意图误差分析（四合一）**

- 左上：ADE/FDE 柱状图（各意图分开）
- 右上：误差随时间增长曲线（不同颜色 = 不同意图，阴影 = ±1σ）
- 左下：X/Y/Z 维度误差占比饼图
- 右下：意图熵分布直方图（0 = 模型非常确定，高值 = 模型纠结）

### 05_uncertainty.png
**模型不确定性校准**

- 上排：不确定性（预测方差）随预测步增长（X/Y/Z 分量分开）
- 下排：校准曲线。横轴 = 模型自报的方差，纵轴 = 实际误差
- **点越靠近虚线 = 模型越有"自知之明"**

### 06_adapter_comparison.png
**Context Adapter 前后对比**

- 每行一条仿真长轨迹，4 列 = 4 个测试窗口
- 左图 XY 轨迹：蓝=历史 / 红=加前 / 橙=加后 / 绿=真值
- 右图逐步误差：红=加前 / 橙=加后 / 阴影=改善幅度
- 彩色菱形/圆形标记 = 5s/10s/15s/20s 标记点
- 标题标注：轨迹编号、起始帧、误差变化（xx→xx m, +xx%）

### 07_trajectory_grid.png
**12 样本轨迹叠加网格**

- 上 6 个 = LOW，下 6 个 = HIGH
- 每类意图各选一个中位数样本
- **展示模型的典型表现（非最优/最差）**
- 标题标注：样本编号、意图、速度、FDE

### 08_summary_table.png
**核心指标汇总表**

- 行 = 所有评估指标（ADE/FDE/方向误差/速度RMSE/意图准确率/逐意图FDE）
- 列 = LOW vs HIGH
- **组会汇报时可直接用这张表做总结页**

---

## 四、诊断图表 (diagnose_failures.py)

### diag_low_summary.png / diag_high_summary.png
**错误模式分析面板（8 合 1）**

1. 误差 vs 速度散点 — 速度是否影响精度
2. 误差 vs 意图熵 — 模型纠结时是否更容易出错
3. 误差 vs 惯性门控 — 过度依赖物理模型会不会导致错误
4. 误差 vs 模型置信度 — 模型是否"自信地犯错"
5. 误差 vs 轨迹曲率 — 转弯越大是否误差越大
6. 意图混淆矩阵（仅最差 30 个样本）— 最差样本的意图错误模式
7. 误差维度分解 — X/Y/Z 哪个维度主导误差
8. Top 10 最差样本条形图 — 红色=意图错配，青色=意图正确

### diag_low_detail.png / diag_high_detail.png
**最差 6 个样本的深度逐帧诊断**

每个样本 8 行面板：
1. **XY 轨迹** — 历史/预测/真值叠加
2. **逐步 L2 误差** — 柱状图 + 均值线
3. **逐维度误差** — X/Y/Z 分开的误差曲线
4. **意图概率分布** — 模型认为当前是什么意图
5. **门控值动态** — Inertia（物理权重）和 Anchor（锚点强度）随时间变化
6. **物理 vs 神经 vs 最终** — 三个分量的位移范数对比
7. **不确定性** — X/Y/Z 的预测方差
8. **Z 轴详情** — 历史 Z / 预测 Z / 真值 Z 叠加

### diag_error_distribution.png
**误差分布直方图**

- LOW (蓝) vs HIGH (红) 的 FDE 分布
- 虚线 = 中位数，点线 = P95
- **HIGH 长尾由 DESCEND 样本造成**

---

## 五、综合评估 (evaluate.py)

### eval_comprehensive.png
**8 合 1 综合评估面板**

1. ADE 分布直方图（LOW vs HIGH 叠加）
2. FDE 分布直方图
3. LOW 逐意图 ADE/FDE 柱状图
4. HIGH 逐意图 ADE/FDE 柱状图
5. 方向误差 CDF 曲线
6. 速度剖面误差分布
7. 不确定性校准散点图
8. 指标汇总表

### eval_metrics.json
**评估指标的 JSON 格式原始数据**

包含 ADE/FDE 的 mean/median/P95、方向误差、速度 RMSE、逐意图误差。可直接导入 Python/Excel 做进一步分析。

---

## 六、自回归外推 (rollout.py)

### rollout_low.png
**LOW 模型自回归外推 4s → 12s**

- 蓝 = 历史 4s（第 0 次）
- 橙 = 第 1 次外推（4-8s）
- 红 = 第 2 次外推（8-12s）
- 紫 = 第 3 次外推
- **展示无重训情况下延长预测时域的效果**

### rollout_high.png
**HIGH 模型自回归外推 20s → 40s**

- 蓝 = 历史 20s
- 橙 = 第 1 次外推（20-40s）
- **2 次推理即可将预测从 20s 延长到 40s**

---

## 七、多假设预测评估 (eval_multi_head.py)

### eval_multihypothesis.png
**多假设 (K=5) vs 单模型 — 6 合 1 对比面板**

1. **FDE 分布叠加** — 单模型(红) vs 多假设(绿) 的 FDE 直方图，虚线标注中位数。**绿色整体左移 = 误差大幅降低**
2. **ADE 分布叠加** — 同上，ADE 维度。**多假设的 ADE 分布更集中在低误差区**
3. **逐意图 FDE 柱状图** — 4 个意图分别对比，柱顶标注改善百分比。**DESCEND 改善最大**
4. **DESCEND FDE 分布** — 单独放大看 DESCEND 类别的改善。**长尾从 60m+ 压缩到 20m 以内**
5. **逐样本散点图** — 横轴=单模型FDE，纵轴=多假设minFDE_5。对角线以下=改善。**绝大多数点在对角线以下**
6. **全指标汇总表** — Single / Multi K=5 / Improvement 三列完整对比

### eval_multihypothesis.json
**多假设评估原始数据 (JSON)**

包含单模型和多假设的 ADE/FDE mean/median/P95、逐意图 minFDE、改善百分比。

### eval_multihypothesis_error_growth.png
**逐意图误差时间增长曲线 — 单模型 vs 多假设**

- 展示 4 个意图类别在 20 步预测时域上的误差增长
- 多假设(绿)在远期步的误差远低于单模型(红)

---

## 八、多假设轨迹可视化 (visualize_multihyp.py)

### multihyp_trajectory_grid.png
**多假设 (K=5) 轨迹样本网格 — 16 个样本**

- 4 行 × 4 列，每行一个意图（STRAIGHT/TURN_L/TURN_R/DESCEND）
- 蓝色 = 历史，绿色 = 真值，红色虚线 = 单模型，**绿色实线 = 多假设最佳**
- 灰色细线 = 其他 4 条假设轨迹（展示覆盖范围）
- 标题标注 minFDE_5
- **看绿色实线是否紧贴绿色真值，灰色覆盖范围是否包含真值**

### multihyp_descend_deep.png
**DESCEND 样本深度分析 — 5 条假设全部可见**

- 4 个 DESCEND 样本，每个展示全部 K=5 条轨迹
- 5 种颜色对应 5 个预测头，标注每条轨迹的 Z 终点位移
- 标题标注置信度 (conf) 和 minFDE_5
- **核心看点：至少一条轨迹命中下降路径，即使单模型完全偏离**

### summary_table_updated.png
**更新版核心指标汇总表**

- 3 列：Single Model / Multi K=5 (minFDE_5) / Improvement
- 包含 ADE/FDE/P95、4 个意图分别的 FDE
- DESCEND 行黄色高亮
- **组会汇报直接用这张表替代旧的 summary_table.png**

---

## 关键数据速查

### 单模型 (确定性预测)

| 指标 | LOW | HIGH |
|:--|:--:|:--:|
| ADE mean | 0.61m | 1.71m |
| FDE mean | 1.38m | 5.46m |
| FDE median | **1.15m** | **1.42m** |
| FDE P95 | 3.45m | 36.97m |
| 方向误差 | 23° | 0.1° |
| 最佳意图 | HOVER 0.19m | STRAIGHT 1.42m |
| 最差意图 | TURN_R 1.55m | DESCEND 27.26m |
| 灾难性失败率 | 0.52% | 0% |

### 多假设预测 (K=5, 5轮 WTA 训练) ★NEW

| 指标 | 单模型 | Multi K=5 | 改善 |
|:--|:--:|:--:|:--:|
| ADE mean | 1.71m | **0.93m** | **+46%** |
| FDE mean | 5.46m | **1.84m** | **+66%** |
| FDE median | 1.42m | **0.95m** | +33% |
| FDE P95 | 36.97m | **5.22m** | **+86%** |
| STRAIGHT FDE | 1.42m | 0.96m | +32% |
| TURN_L FDE | 4.24m | 3.02m | +29% |
| TURN_R FDE | 3.65m | 3.26m | +11% |
| **DESCEND FDE** | **27.26m** | **6.13m** | **+78%** 🔥 |

> **核心突破**: 多假设预测 (Winner-Takes-All) 让模型不需要"猜对"唯一的下降轨迹，只需要 5 条候选轨迹中有一条命中。DESCEND 从 27.26m → 6.13m (-78%)，P95 从 36.97m → 5.22m (-86%)，长尾被大幅压缩。仅训练 48K 参数（总参数 3.5%），5 轮约 50 分钟。测试集 150K 样本全量评估。
