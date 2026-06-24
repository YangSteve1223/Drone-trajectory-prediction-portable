# Drone Trajectory Prediction — Portable

可移植的无人机轨迹预测推理包。三模型速度自适应软切换 + LoRA在线持续学习，开箱即用。

## 快速开始

```bash
pip install torch numpy
```

```python
from predictor import DronePredictor

# 加载模型 (首次自动加载三模型权重 ~48MB)
predictor = DronePredictor()

# 输入: (B, 20, 6) — 历史20帧, 每帧 [x, y, z, vx, vy, vz] (米, 米/秒)
history = torch.randn(1, 20, 6)

# 预测
result = predictor.predict(history)
result['predictions']    # (1, 20, 3) 未来3D位移 (米)
result['intent_logits']  # (1, 6)    意图分类logits
result['speed']          # (1,)      检测速度 (m/s)
```

## 三模型速度自适应软切换

```
速度 (m/s):  0 ─────── 2.0 ──────────── 8.0 ─────── 28+
              │          │   过渡带        │          │
模型:        🔵 低速      🟣 三角混合       🔴 高速
              │          │               │          │
场景:        DJI 无人机  过渡/桥接        物流无人机
             慢速高机动                 快速低机动
```

| 权重 | 类别 | 速度域 | RMSE | DistAcc |
|------|:---:|--------|:----:|:-------:|
| `weights/low_speed_6class.pth` | 6 | 0–3 m/s | 0.56 | 89.5% |
| `weights/mixed_6class.pth` | 6 | 桥接 | 0.56 | 89.2% |
| `weights/high_speed_4class.pth` | 4 | 8–28 m/s | 3.10 | 75.5% |

路由准确率 **100%** — 174,900 测试样本零误路由。

## LoRA 在线持续学习

每个无人机独立 LoRA 微调 (~11K参数/drone, ~50KB磁盘)，少量观测即可适配个体飞行特征。

```python
predictor.enable_adaptation()   # 开启在线学习

# 流式推理 + 自适应微调
for timestep, (hist, ground_truth) in enumerate(data_stream):
    out = predictor.predict_with_adaptation(
        hist, drone_id='drone_001',
        ground_truth=ground_truth,
        timestep=timestep,
    )
    # 自动: 累积5帧 → 梯度更新 → 定期持久化

# 查看学习状态
status = predictor.get_drone_status('drone_001')
print(status)  # {num_updates, replay_size, escalated, ...}
```

**特性:**
- 🎯 LoRA rank-4, ~11K 可训练参数, 仅作用于 Mixed 模型
- 📦 每 drone ~50KB 磁盘, 100 drones < 5MB
- 🛡️ Replay Buffer 防灾难性遗忘
- 📈 CUSUM 检测预测退化 → 自动升级学习率
- 💾 定期自动持久化到 `checkpoints/adapters/`

## 目录结构

```
├── predictor.py          # DronePredictor (推理 + 在线学习入口)
├── lora.py               # LoRALinear, LoRAAdapter, merge/unmerge
├── adapter_manager.py    # DroneAdapterManager, ReplayBuffer, CUSUM
├── online_learner.py     # OnlineLearner, 累积/更新/持久化
├── emam_model/           # EMAM 模型架构 (6 files)
│   ├── model.py          # TrajectoryPredictor
│   ├── emam_se.py        # Enhanced Mamba SE (SSM)
│   ├── ia_dtp.py         # Intent-Aware Dynamic Temporal Pyramid
│   ├── ua_pgd.py         # Uncertainty-Aware Physics-Guided Decoder
│   └── trigger.py        # 事件触发器 (Simple/Funnel/EventDriven)
├── utils/
│   ├── fast_data_loader.py
│   └── metrics.py
├── weights/              # 预训练权重 (~48MB)
│   ├── low_speed_6class.pth
│   ├── mixed_6class.pth
│   └── high_speed_4class.pth
└── README.md
```

## 依赖

| 包 | 版本 | 用途 |
|---|------|------|
| Python | >= 3.8 | |
| PyTorch | >= 2.0 | 推理 + 训练 |
| numpy | — | 数据处理 |

## 模型规格

- 架构: EMAM-SE + IA-DTP + UA-PGD
- 参数量: ~1.4M / 模型 (d_model=128)
- 输入: 20帧 (5Hz = 4秒历史) × 6维 [pos, vel]
- 输出: 20帧 (4秒预测) × 3维位移
- 推理速度: ~8ms/batch (RTX 3060, batch=1)

## 评估结果

| 测试集 | 速度 | 最佳单模型 | 软切换 | 路由 |
|--------|------|:---------:|:-----:|:---:|
| UAV-Flow (DJI) | 0–5 m/s | 0.555 | 0.556 | 100% → Low |
| NPZDATA (Delivery) | 8–28 m/s | 3.10 | 3.10 | 100% → High |

## License

Research use only.
