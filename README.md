# Drone Trajectory Prediction — Portable

可移植的无人机轨迹预测完整工具包。推理 + 在线学习 + 训练，开箱即用。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 快速开始

```bash
pip install torch numpy tqdm pyyaml
```

```python
from predictor import DronePredictor

predictor = DronePredictor()

# 输入: (B, 20, 6) 历史20帧 [pos_x,pos_y,pos_z, vx,vy,vz] (米, 米/秒)
history = torch.randn(1, 20, 6)
result = predictor.predict(history)

result['predictions']    # (1, 20, 3) 未来3D位移
result['intent_logits']  # (1, 6)    意图分类
result['speed']          # (1,)      检测速度
```

## 核心能力

| 功能 | 说明 |
|------|------|
| 🎯 **三模型软切换** | 基于速度自适应选择低速/混合/高速模型，路由准确率 100% |
| 🧬 **LoRA 在线学习** | 每台无人机独立微调 (~11K参数)，在线适应个体飞行特征 |
| 🔄 **流式实时推理** | 逐帧输入，滚动窗口，适合机载边缘部署 |
| 🔀 **双向 Mamba 增强** | 前向+后向 SSM 编码，改善长程预测和机动过渡 |
| 📦 **完整训练管线** | 提供训练脚本和示例数据集，可从零训练全部模型 |

## 目录结构

```
├── predictor.py              # DronePredictor 推理入口
├── streaming.py              # StreamingPredictor 流式推理
├── bidirectional.py          # BidirectionalPredictor 双向SSM
├── lora.py                   # LoRA 低秩适配
├── adapter_manager.py        # 多无人机适配器管理
├── online_learner.py         # 在线累积学习引擎
├── emam_model/               # EMAM 模型架构
│   ├── model.py              # TrajectoryPredictor
│   ├── emam_se.py            # Enhanced Mamba SE
│   ├── bidirectional_mamba.py
│   ├── ia_dtp.py             # Intent-Aware DTP
│   ├── ua_pgd.py             # Uncertainty-Aware PGD
│   └── trigger.py
├── utils/                    # 数据加载 + 评估指标
├── weights/                  # 预训练权重 (48MB)
│   ├── low_speed_6class.pth
│   ├── mixed_6class.pth
│   └── high_speed_4class.pth
├── checkpoints/              # 训练产出
│   └── bidir_enhancer.pth    # 双向增强器权重
├── train/                    # 训练管线
│   ├── train.py              # 主训练脚本
│   ├── train_bidir.py        # 双向增强器训练
│   ├── preprocess_uavflow.py # 数据预处理
│   ├── extract_uavflow.py    # 数据下载提取
│   ├── *.ps1                 # 训练启动脚本
│   ├── dataset/              # 示例数据集 (295MB)
│   │   ├── uavflow_low/      #   UAV-Flow 低速 (59 chunks)
│   │   └── npzdata_high/     #   NPZDATA 高速 (14 chunks)
│   └── README.md             # 训练文档
└── README.md
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

```python
predictor.enable_adaptation()

for t, (hist, gt) in enumerate(data_stream):
    out = predictor.predict_with_adaptation(
        hist, drone_id='drone_001',
        ground_truth=gt, timestep=t,
    )
    # 自动累积5帧 → 梯度更新 → 定期持久化
```

| 特性 | 参数 |
|------|------|
| 可训练参数 | ~11K / drone |
| 磁盘占用 | ~50KB / drone |
| LoRA rank | r=4 |
| 防遗忘 | Replay Buffer (20条) |
| 退化检测 | CUSUM + 自动升级 |

## 流式实时推理

```python
from streaming import StreamingPredictor

sp = StreamingPredictor(predictor)
for frame in sensor_stream:
    result = sp.update(frame)      # None until 20-frame buffer full
    if result:
        future = result['predictions']  # (1, 20, 3)
```

## 训练自己的模型

```bash
# 训练低速模型
cd train
python train.py --data_root dataset/uavflow_low --num_intent_classes 6 \
    --d_model 128 --batch_size 128 --epochs 30 --lr 1e-4 \
    --trigger_mode simple --exp_name my_model

# 训练双向增强器
python train_bidir.py --data_dir dataset/uavflow_low --epochs 5
```

详见 [`train/README.md`](train/README.md)。

## 评估结果

| 测试集 | 速度 | 最佳单模型 | 软切换 | 路由 |
|--------|------|:---------:|:-----:|:---:|
| UAV-Flow (DJI) | 0–5 m/s | RMSE=0.555 | 0.556 | 100% → Low |
| NPZDATA (Delivery) | 8–28 m/s | RMSE=3.10 | 3.10 | 100% → High |

## 模型规格

- 架构: EMAM-SE + IA-DTP + UA-PGD
- 参数量: ~1.4M / 模型 (d_model=128)
- 输入: 20帧 (5Hz = 4秒历史) × 6D
- 输出: 20帧 (4秒预测) × 3D位移
- 依赖: Python >= 3.8, PyTorch >= 2.0, numpy

## License

Research use only.
