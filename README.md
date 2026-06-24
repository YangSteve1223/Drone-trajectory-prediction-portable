# 无人机轨迹预测 — 可移植工具包

基于增强状态空间模型(EMAM)的无人机轨迹预测系统，支持多意图识别、速度自适应模型切换、在线持续学习和实时流式推理。

## 快速开始

```bash
pip install torch numpy tqdm pyyaml
```

```python
from predictor import DronePredictor

predictor = DronePredictor()

# 输入: (B, 20, 6) — 20帧历史, 每帧 [x,y,z, vx,vy,vz] (米, 米/秒)
history = torch.randn(1, 20, 6)
result = predictor.predict(history)

result['predictions']    # (1, 20, 3) 未来位移
result['intent_logits']  # (1, 6)    意图分类
result['speed']          # (1,)      速度
```

## 核心功能

**三模型速度自适应软切换。** 基于历史轨迹速度在低速(6-class UAV-Flow)、混合(6-class)和高速(4-class NPZDATA)三个模型之间平滑切换。低速场景(0-3 m/s)由低速模型主导，高速场景(8-28 m/s)由高速模型主导，过渡带(2-8 m/s)三角隶属融合。测试集路由准确率100%。

**LoRA在线持续学习。** 每台无人机独立微调约11K参数，通过低秩适配矩阵在SSM投影层注入无人机专属偏移量。基础模型权重冻结，仅更新LoRA矩阵和末端输出头。累积5条置信轨迹后触发一次梯度更新，配合Replay Buffer(容量20)防止灾难性遗忘。CUSUM检测器监测预测退化并自动升级学习率。

**流式实时推理。** 维护20帧滑动窗口，每收到新帧即输出预测。适合机载边缘部署，逐帧处理延迟约46ms/batch(RTX 3060)。

**空间自适应缩放。** 基于窗口平均速度动态计算归一化尺度，替代模型内部硬编码的固定常数(`_scale_pos=100`)。提供`predict_adaptive()`兼容现有权重，以及`predict_normalized()`供动态归一化重新训练的模型使用。

**安全与风险评估。** 支持圆柱、长方体和球形禁飞区边界定义，计算预测轨迹到边界的最小距离和侵入时间。异常检测器基于预测误差的自适应阈值(mu+k*sigma)和CUSUM累积和检测持续退化。

## 目录结构

```
├── predictor.py              # DronePredictor — 推理和在线学习入口
├── streaming.py              # StreamingPredictor — 逐帧流式推理
├── bidirectional.py          # BidirectionalPredictor — 双向SSM增强
├── lora.py                   # LoRALinear, LoRAAdapter — 低秩适配
├── adapter_manager.py        # DroneAdapterManager — 多机适配器管理
├── online_learner.py         # OnlineLearner — 累积更新引擎
├── safety.py                 # RiskAssessment, AnomalyDetector
├── dynamic_norm.py           # DynamicNormalizer, SI-MSE loss
├── emam_model/               # EMAM模型架构
│   ├── model.py              # TrajectoryPredictor (EMam-SE + IA-DTP + UA-PGD)
│   ├── emam_se.py            # Enhanced Mamba with Squeeze-and-Excitation
│   ├── bidirectional_mamba.py  # BidirectionalSelectiveSSM
│   ├── ia_dtp.py             # Intent-Aware Dynamic Temporal Pyramid
│   ├── ua_pgd.py             # Uncertainty-Aware Physics-Guided Decoder
│   └── trigger.py            # SimpleTrigger / FunnelTrigger / EventDrivenTrigger
├── utils/                    # 数据加载和评估指标
├── weights/                  # 预训练权重 (48MB)
│   ├── low_speed_6class.pth  # 低速模型 (6-class, UAV-Flow, RMSE=0.56)
│   ├── mixed_6class.pth      # 混合模型 (6-class, RMSE=0.56)
│   └── high_speed_4class.pth # 高速模型 (4-class, NPZDATA, RMSE=3.10)
├── checkpoints/              # 训练产出 (双向增强器权重)
├── train/                    # 训练脚本和示例数据集
│   ├── train.py              # 主训练脚本 (含NaN保护、自动回滚)
│   ├── train_bidir.py        # 双向增强器训练
│   ├── dataset/              # 预处理数据集 (295MB)
│   └── README.md             # 训练文档
└── README.md
```

## 三模型软切换架构

速度(m/s): 0 — 2.0 — 8.0 — 28+
低速模型(6-class): DJI无人机, 慢速高机动, RMSE=0.56
混合模型(6-class): 过渡桥接, RMSE=0.56
高速模型(4-class): 物流无人机, 快速低机动, RMSE=3.10

## 在线学习使用

```python
predictor.enable_adaptation()

for t, (hist, gt) in enumerate(data_stream):
    out = predictor.predict_with_adaptation(
        hist, drone_id='drone_001',
        ground_truth=gt, timestep=t,
    )
```

## 流式推理使用

```python
from streaming import StreamingPredictor

sp = StreamingPredictor(predictor)
for frame in sensor_stream:
    result = sp.update(frame)      # 缓冲满20帧后开始输出
    if result:
        future = result['predictions']
```

## 训练

训练脚本和预处理数据集位于`train/`目录。详见`train/README.md`。

```bash
cd train
python train.py --data_root dataset/uavflow_low --num_intent_classes 6 \
    --d_model 128 --batch_size 128 --epochs 30 --lr 1e-4 \
    --trigger_mode simple --exp_name my_model
```

## 模型规格

- 架构: EMAM-SE + IA-DTP + UA-PGD
- 参数量: ~1.4M / 模型 (d_model=128)
- 输入: 20帧 (5Hz = 4秒历史) x 6维 [pos, vel]
- 输出: 20帧 (4秒预测) x 3维位移
- 依赖: Python >= 3.8, PyTorch >= 2.0, numpy

## 评估结果

| 测试集 | 速度范围 | 单模型RMSE(m) | 软切换RMSE(m) | 路由准确率 |
|--------|---------|:---------:|:---------:|:------:|
| UAV-Flow | 0-5 m/s | 0.555 | 0.556 | 100% |
| NPZDATA | 8-28 m/s | 3.10 | 3.10 | 100% |

## 引用

本项目基于EMAM (Enhanced Mamba)架构实现。如使用本工作，请引用原论文。

## 许可

仅限研究用途。
