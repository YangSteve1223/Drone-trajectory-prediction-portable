# Training Guide

训练自己的无人机轨迹预测模型。所有脚本从 `train/` 目录运行。

## 依赖

```bash
pip install torch numpy pyyaml tqdm tensorboard
```

## 数据准备

### 方案 A: UAV-Flow 真实数据 (DJI 无人机, 低速高机动)

```bash
# 1. 下载原始数据
python extract_uavflow.py --output ../UAV-Flow-trajs

# 2. 生成滑动窗口 (6类意图标注)
python preprocess_uavflow.py --traj_dir ../UAV-Flow-trajs --out_dir ../UAV-Flow-windows

# 3. (可选) 混入 SimCruise 仿真数据增强
python ../align_npzdata.py  # 需要先准备好 SimCruise
```

### 方案 B: 自定义数据

数据格式: `.npz` 文件，每个包含:
- `hist`: (N, 20, 6) float32 — 历史轨迹 [pos_x, pos_y, pos_z, vx, vy, vz]
- `pred`: (N, 20, 3) float32 — 未来位移
- `intent`: (N,) int32 — 意图标签
- `maneuver`: (N,) int32 — 机动等级 (可选)

文件命名: `windows_train_chunk*.npz`, `windows_val_chunk*.npz`, `windows_test_chunk*.npz`

## 训练命令

### 纯低速模型 (UAV-Flow, 6-class)

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"

# 从零训练
python train.py --data_root ../UAV-Flow-pure --num_intent_classes 6 `
    --d_model 128 --batch_size 128 --epochs 30 --lr 1e-4 `
    --trigger_mode simple --exp_name my_low_speed `
    --loss_intent_weight 0.3 --num_workers 0

# 或从预训练权重热启动
python train.py --data_root ../UAV-Flow-pure --num_intent_classes 6 `
    --d_model 128 --batch_size 128 --epochs 20 --lr 1e-5 `
    --pretrained ../weights/mixed_6class.pth `
    --trigger_mode simple --exp_name my_low_speed `
    --loss_intent_weight 0.3 --num_workers 0
```

### 纯高速模型 (SimCruise, 4-class)

```powershell
python train.py --data_root ../SimCruise --num_intent_classes 4 `
    --d_model 128 --batch_size 128 --epochs 50 --lr 3e-4 `
    --warmup_epochs 5 --trigger_mode simple --exp_name my_high_speed `
    --loss_intent_weight 0.3
```

### 续训已有模型

```powershell
python train.py --data_root ../SimCruise --num_intent_classes 4 `
    --d_model 128 --batch_size 128 --epochs 50 `
    --resume ../checkpoints/my_model/latest.pth `
    --trigger_mode simple --exp_name my_high_speed
```

### 训练双向Mamba增强器

```powershell
python train_bidir.py --epochs 5 --batch_size 128 --data_dir ../UAV-Flow-pure
```

## 关键参数说明

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `--d_model` | 模型维度 | 128 (6GB显存) / 256 (更好的效果) |
| `--batch_size` | 批大小 | 128 (6GB) / 256 (12GB+) |
| `--lr` | 学习率 | 1e-4 ~ 3e-4 |
| `--warmup_epochs` | 预热轮数 | 5 |
| `--trigger_mode` | 触发模式 | `simple` (始终训练, 推荐) |
| `--pretrained` | 预训练权重路径 | 从已有权重热启动 |
| `--resume` | 续训检查点路径 | 从中断处继续 |
| `--no_amp` | 关闭混合精度 | UAV-Flow数据建议开启 |
| `--num_workers` | 数据加载进程 | 0 (UAV-Flow) / 2-4 (SimCruise) |

## 输出

训练完成后 `../checkpoints/<exp_name>/` 包含:
- `best.pth` — 验证集最优权重
- `latest.pth` — 最新权重（用于续训）
- `epoch_*.pth` — 每轮检查点
- `config.yaml` — 训练配置

## 模型转换

训练好的权重可直接放入 `../weights/` 替换默认权重，或通过 `DronePredictor` 的 `low_ckpt`/`mixed_ckpt`/`high_ckpt` 参数指定路径。
