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
├── train_multi_head.py       # 多假设解码器训练 (WTA loss + 断点续训)
├── evaluate.py               # 综合评估 (ADE/FDE/意图/不确定性)
├── eval_multi_head.py        # 多假设测试集全量评估
├── diagnose_failures.py      # 最差样本深度诊断
├── visualize_trajectories.py # LOW+HIGH 12 精选样本轨迹
├── visualize_multihyp.py     # 多假设专属图表
├── visualize_final.py        # 8 张科研图表
├── run_all.py                # 6 张基础轨迹图
├── rollout.py                # 自回归预测外推
├── fix_labels.py             # UAV-Flow 标签修正
├── context_adapter.py        # ContextAdapterV2 (上下文注入解码器)
├── context_sim.py            # 仿真数据 Adapter 训练
├── pic-results/              # 所有评估图表 + README.md
└── README.md
```

## 性能指标 (2026-07-09)

### 单模型 (确定性预测)

| 指标 | LOW (UAV-Flow) | HIGH (SimCruise) |
|:--|:--:|:--:|
| ADE mean / median | 0.61m / 0.50m | 1.71m / 0.75m |
| FDE mean / median | 1.39m / 1.15m | 5.46m / 1.42m |
| FDE P95 | 3.45m | 36.97m |
| 方向误差 | 24.5° | 0.1° |
| 灾难性失败 (>90°) | 3.35% | 0% |
| 最佳意图 | HOVER 0.19m | STRAIGHT 1.42m |
| 最差意图 | DESCEND 4.08m | **DESCEND 27.26m** |

### 多假设预测 — LOW (K=5, 5轮 WTA, 24.9K 测试集全量)

| 指标 | 单模型 | minFDE_5 | 改善 |
|:--|:--:|:--:|:--:|
| 整体 minADE_5 | 0.61m | **0.48m** | **+21.6%** |
| 整体 minFDE_5 | 1.39m | **0.94m** | **+32.3%** |
| FDE P95 | 3.45m | **2.31m** | **+33.2%** |
| 方向误差 mean | 24.5° | **17.5°** | **+28.5%** |
| 灾难性失败 | 3.35% | **2.94%** | **+12.5%** |
| STRAIGHT | 1.21m | 0.83m | +31.1% |
| TURN_L | 1.38m | 0.94m | +32.5% |
| TURN_R | 1.55m | 1.01m | +34.5% |

### 多假设预测 — HIGH (K=5, 5轮 WTA, 150K 测试集全量)

| 指标 | 单模型 | minFDE_5 | 改善 |
|:--|:--:|:--:|:--:|
| 整体 minADE_5 | 1.71m | **0.93m** | **+46%** |
| 整体 minFDE_5 | 5.46m | **1.84m** | **+66%** |
| FDE P95 | 36.97m | **5.22m** | **+86%** |
| STRAIGHT | 1.42m | 0.96m | +32% |
| TURN_L | 4.24m | 3.02m | +29% |
| TURN_R | 3.65m | 3.26m | +11% |
| **DESCEND** | **27.26m** | **6.13m** | **+78%** |

> 多假设仅训练 48K 参数（总参数 3.5%），LOW 5 轮约 14 分钟，HIGH 5 轮约 50 分钟。

## 脚本速查

| 脚本 | 用途 | 命令示例 |
|:--|:--|:--|
| `predictor.py` | 推理入口 (软融合+Z纠正) | `from predictor import DronePredictor` |
| `train_multi_head.py` | 多假设 WTA 训练 (支持 LOW+HIGH) | `python train_multi_head.py --model low --K 5 --resume` |
| `evaluate.py` | ADE/FDE/意图/不确定性评估 | `python evaluate.py` |
| `eval_multi_head.py` | 多假设完整测试集评估 | `python eval_multi_head.py` |
| `visualize_low_multihyp.py` | **LOW 多假设 24 样本轨迹 (10张)** ★ | `python visualize_low_multihyp.py` |
| `visualize_trajectories.py` | HIGH 多假设 12 样本轨迹图表 | `python visualize_trajectories.py` |
| `visualize_multihyp.py` | 多假设对比图表 | `python visualize_multihyp.py` |
| `compare_low_models.py` | LOW 单模型 vs 多假设对比 | `python compare_low_models.py` |
| `eval_bidirectional.py` | 双向 Mamba 评估 (已否决) | `python eval_bidirectional.py` |
| `diagnose_failures.py` | 最差样本深度诊断 | `python diagnose_failures.py` |
| `rollout.py` | 自回归预测外推 | `python rollout.py` |
| `fix_labels.py` | UAV-Flow 标签修正 | `python fix_labels.py` |

## 模型规格

- 参数量: ~1.4M / 模型 (d_model=128, d_state=16)
- 多假设头: +48K 可训练参数 (K=5, 占总参数 3.5%)
- 输入: 20 帧 × 6 维 [pos, vel] (LOW 4s@5Hz, HIGH 20s@1Hz)
- 输出: 20 帧 × 3 维位移 (或 K×20×3 多假设)
- 硬件: RTX 3060 Laptop (6GB), Windows 11, Python 3.10

## 后续方向

- **LOW 方向误差**仍偏高（17.5° vs HIGH 0.1°），多假设已改善 28.5%，仍有优化空间
- 极端转弯 (>150°) 处理
- 实时在线学习 (LoRA) 验证
- HOVER 类别在多假设下略有退化（0.19m → 0.40m），可考虑意图感知的 K 值调整

## 已知问题

- **LOW 方向误差 17.5°** vs HIGH 0.1° — 虽已改善但仍是最主要差距
- **LOW 极端转弯** (>150°): 模型可能预测 STRAIGHT
- 物理模型（2 帧速度）过于简单
- Context Adapter 仅验证于仿真数据

## 许可

仅限研究用途。
