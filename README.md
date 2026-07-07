# EMAM — Drone Trajectory Prediction (Portable)

基于 EMAM (Enhanced Mamba) 架构的无人机轨迹预测系统。双模型速度自适应切换 + 多假设预测 + 在线持续学习。

## 快速开始

```bash
pip install torch numpy tqdm
```

```python
from predictor import DronePredictor

predictor = DronePredictor()

# 输入: (B, 20, 6) — 20帧历史, 每帧 [x,y,z, vx,vy,vz] (米, 米/秒)
history = torch.randn(1, 20, 6)
result = predictor.predict(history)

result['predictions']    # (1, 20, 3) 未来位移 (米)
result['intent_logits']  # (1, 6)    意图分类 logits
result['speed']          # (1,)      当前速度 (m/s)
result['route']          # ['LOW'|'HIGH'] 路由选择
```

## 架构

```
Input (B,20,6) → EMam-SE (SSM编码器) → IA-DTP (意图分类) → UA-PGD (解码器)
                                                                 ├─ Physics Inertia Gate
                                                                 ├─ Neural Decoder (1或K个头)
                                                                 └─ Kinematic Physics Model
```

### 双模型软融合

| 模型 | 数据集 | 类别 | 频率 | 速度域 |
|:--|:--|:--|:--|:--|
| LOW | UAV-Flow (真实DJI) | 6-class | 5Hz | 0–3 m/s |
| HIGH | SimCruise (仿真巡航) | 4-class | 1Hz | 8–28 m/s |

- **软融合**: `α = sigmoid((speed - 5.0) / 1.2)`, 过渡区 [2, 8] m/s 内平滑混合
- 过渡区外硬分配，避免跨域污染
- 意图 logits 同步软融合

### 多假设预测 (Multi-Hypothesis)

K=5 个独立预测头 + 置信度评分头，Winner-Takes-All 训练：

```python
# 训练多假设解码器
python train_multi_head.py --model high --K 5 --epochs 10 --batch_size 64

# 推理时取 K 条轨迹中置信度最高的，或用 minFDE_K 评估
```

### Z轴三段式纠正

基于 DESCEND 概率的三段式过渡 + 多信号融合：
- 信号1: Z历史趋势 (下降倾向检测)
- 信号2: 模型Z轴不确定性 (低方差 → 减弱压制)
- 信号3: 预测Z位移量级 (大位移 → 减弱压制)

## 目录结构

```
├── predictor.py              # DronePredictor — 推理入口 (软融合+Z纠正)
├── emam_model/               # EMAM 模型架构
│   ├── model.py              # TrajectoryPredictor
│   ├── emam_se.py            # Enhanced Mamba with SE
│   ├── ia_dtp.py             # Intent-Aware DTP
│   ├── ua_pgd.py             # UA-PGD + MultiHeadNeuralDecoder
│   └── trigger.py            # 事件触发器
├── utils/                    # 数据加载 & 评估指标
│   ├── fast_data_loader.py   # FastWindowDataset
│   └── metrics.py            # ADE/FDE/MMD 指标
├── weights/                  # 预训练权重 (~48MB)
│   ├── low_speed_6class.pth  # LOW 模型
│   └── high_speed_4class.pth # HIGH 模型
├── train_multi_head.py       # 多假设解码器训练 (WTA loss)
├── evaluate.py               # 综合评估 (ADE/FDE/意图/不确定性)
├── diagnose_failures.py      # 最差样本深度诊断
├── visualize_final.py        # 8 张科研图表
├── run_all.py                # 6 张基础轨迹图
├── rollout.py                # 自回归预测外推
├── fix_labels.py             # UAV-Flow 标签修正
├── context_adapter.py        # ContextAdapterV2 (上下文注入解码器)
├── context_sim.py            # 仿真数据 Adapter 训练
├── pic-results/              # 所有评估图表 + README.md
└── README.md
```

## 性能指标 (2026-07-07)

### 单模型 (确定性预测)

| 指标 | LOW (UAV-Flow) | HIGH (SimCruise) |
|:--|:--:|:--:|
| ADE mean / median | 0.61m / 0.50m | 1.71m / 0.75m |
| FDE mean / median | 1.38m / 1.15m | 5.46m / 1.42m |
| FDE P95 | 3.45m | 36.97m |
| 方向误差 | 23° | 0.1° |
| 灾难性失败 (>90°) | 0.52% | 0% |
| 最佳意图 | HOVER 0.19m | STRAIGHT 1.42m |
| 最差意图 | TURN_R 1.55m | **DESCEND 27.26m** |

### 多假设预测 (K=5, 3轮快速训练)

| 指标 | 单模型 | minFDE_5 | 改善 |
|:--|:--:|:--:|:--:|
| 整体 minADE_5 | 1.71m | **0.87m** | **+49%** |
| 整体 minFDE_5 | 5.46m | **1.87m** | **+66%** |
| STRAIGHT | 1.42m | 0.99m | -55% |
| **DESCEND** | **27.26m** | **7.04m** | **-69%** |
| TURN_L | 4.24m | 3.53m | -25% |
| TURN_R | 3.65m | 3.85m | -3% |

> **关键突破**: DESCEND 贡献 HIGH 模型 76% 的总误差（仅 15% 样本）。多假设预测让模型不需要"猜对"下降，只需要"覆盖到"——5 条轨迹中至少一条命中下降路径。

## 脚本速查

| 脚本 | 用途 | 命令示例 |
|:--|:--|:--|
| `predictor.py` | 推理入口 | `from predictor import DronePredictor` |
| `evaluate.py` | 综合评估 | `python evaluate.py` |
| `train_multi_head.py` | 多假设训练 | `python train_multi_head.py --model high --K 5` |
| `diagnose_failures.py` | 最差样本诊断 | `python diagnose_failures.py` |
| `visualize_final.py` | 科研图表 | `python visualize_final.py` |
| `run_all.py` | 轨迹可视化 | `python run_all.py` |
| `rollout.py` | 预测外推 | `python rollout.py` |
| `fix_labels.py` | 标签修正 | `python fix_labels.py` |

## 模型规格

- 参数量: ~1.4M / 模型 (d_model=128, d_state=16)
- 多假设头: +48K 可训练参数 (K=5, 占总参数 3.5%)
- 输入: 20 帧 × 6 维 [pos, vel] (LOW 4s@5Hz, HIGH 20s@1Hz)
- 输出: 20 帧 × 3 维位移 (或 K×20×3 多假设)
- 硬件: RTX 3060 Laptop (6GB), Windows 11, Python 3.10

## 已知问题

- **HIGH DESCEND**: 历史窗口内 Z 坐标恒定（仿真巡航），模型需从非 Z 特征推断即将下降。多假设已大幅改善（-69%），但无法完全消除不确定性
- **LOW 极端转弯** (>150°): 模型可能预测 STRAIGHT 而非转弯（标签修正已改善）
- **物理模型过于简单**: 2 帧速度估算，不足以支撑复杂机动

## 许可

仅限研究用途。
