# EMAM — 无人机轨迹预测系统（便携版）

基于 EMAM（Enhanced Mamba）架构的无人机轨迹预测系统。核心特性：

- **速度自适应双模型软融合** — LOW（真实低速）+ HIGH（仿真巡航）
- **40 帧长历史扩展** — 长轨迹 FDE 从 2.32m 降至 0.87m，灾难性失败近零
- **LoRA 个性化** — 全局 LoRA（锦上添花）+ 逐无人机 LoRA（在线学习）
- **多假设预测** — K=5 Winner-Takes-All，minFDE 显著优于单假设模型
- **在线持续学习** — 逐无人机流式增量适配（LOW 和 HIGH 均验证正收益）

## 快速开始

```bash
pip install -r requirements.txt
```

系统有两个推理入口，按用途选择：

### 1. `DronePredictor` — 通用短程推理（20 帧输入）

带 Z 轴纠正 + 双模型软融合，适合一般实时预测。

```python
import torch
from predictor import DronePredictor

predictor = DronePredictor()

# 输入: (B, 20, 6) — 20 帧历史, 每帧 [x,y,z, vx,vy,vz] (米, 米/秒)
history = torch.randn(1, 20, 6)
result = predictor.predict(history)

result['predictions']    # (1, 20, 3) 未来位移 (米)
result['intent_logits']  # 意图分类 logits
result['speed']          # 当前速度 (m/s)
result['route']          # 'LOW' | 'HIGH' 路由选择
```

### 2. `DeployedLowPredictor` — 长轨迹 + 逐无人机在线学习（40 帧输入）

用于持续飞行的低速无人机，随飞随学，为每架无人机建立个性化 LoRA。

```python
import torch
from deploy import DeployedLowPredictor

deployer = DeployedLowPredictor(use_global=True)  # 全局 LoRA 默认开启

hist = torch.randn(1, 40, 6)          # 40 帧历史
result = deployer.predict(
    hist,
    drone_id='drone_007',             # 设置后启用逐无人机在线 LoRA
    ground_truth=future_disp,         # (1,20,3) 上一步真值，喂给在线学习（可选）
    frames_seen=120,                  # 该无人机累计帧数（长度门控用）
)
result['predictions']  # (1, 20, 3) 未来位移
```

三道门控（详见 `deploy.py` docstring）：

1. **长度门控** (`online_min_frames=60`)：短飞行只用干净 40 帧 base，不加 LoRA。
2. **全局开关** (`use_global`)：一键开/关共享的低速全局 LoRA (`dir_lora_40`)。
3. **速度门控** (`global_max_speed=4.0`)：速度超过 4 m/s 时关闭全局 LoRA（它训练于 0–3 m/s 真实飞行，超域不适用）。

## 架构

```
Input (B,T,6) → EMam-SE (SSM编码器) → IA-DTP (意图分类) → UA-PGD (解码器)
                                                             ├─ Physics Inertia Gate
                                                             ├─ Neural Decoder (1 或 K 个头)
                                                             └─ Kinematic Physics Model
```

### 双模型软融合

| 模型 | 数据集 | 类别 | 频率 | 速度域 | 历史长度 |
|:--|:--|:--|:--|:--|:--|
| LOW | UAV-Flow (真实DJI) | 6-class | 5Hz | 0–3 m/s | 40 帧 (长) / 20 帧 (短) |
| HIGH | SimCruise (仿真巡航) | 4-class | 1Hz | 8–28 m/s | 20 帧 |

- 软融合：`α = sigmoid((speed - 5.0) / 1.2)`，过渡区 [2, 8] m/s 平滑混合，区外硬分配避免跨域污染。
- **HIGH 保持 20 帧**：40 帧扩展在 1Hz 巡航上是负收益（20 帧已 = 20s），已验证否决。

### 40 帧扩展（LOW）

- `low_speed_6class_40frame.pth`：从 20 帧 checkpoint 迁移 147/148 权重，冻结编码器、微调解码器+门控。
- 长轨迹覆盖率 7% → 84%，FDE 0.87m（vs 20 帧 2.32m），灾难性失败近零。
- 自适应步长采样：`stride = max(1, min(4, n//60))`。

### LoRA 策略

LoRA = **锦上添花**，不是雪中送炭。base 模型先把所有长度处理好，LoRA 再做增量个性化。

- **上游 targets（不含 delta_head）**：SSM in/out_proj + `ua_pgd.feat_compress` + `neural_decoder.proj.0`，头部只微调 `anchor_to_pos.2`，约 100K 参数。
- **为何排除 delta_head**：它逐步独立处理 20 个预测步，LoRA 作用其上会放大步间差异 → 锯齿轨迹。
- **全局 LoRA**：`dir_lora_40.pth`（当前最佳，方向加权训练）。
  - window-level 划分：FDE +19.4%，方向误差 13.8°→12.1°（同飞行窗口可能同现于训练/测试，偏乐观）。
  - **跨飞行划分（诚实泛化，20% 飞行完全留出）：FDE +7.3%，方向 +4.6%**（`eval/eval_global_lora_generalization.py`）。真实部署应以此为准。
- **叠加**：40 帧 base → 合并全局 LoRA → 逐无人机在线 LoRA。

### 在线持续学习

逐无人机流式增量适配：replay buffer + CUSUM 漂移检测 + 常驻 adapter，gentle 配置（lr 3e-5, accum 10, l2 0.05）。

- **因果验证结论**：真实部署只能见因果 warmup 数据时，流式在线（+2.4%）远胜离线批量微调（-202% 灾难性过拟合）。
- LOW：FDE +1.6%，21/30 无人机改善（5Hz 每段飞行数据有限，增益温和）。
- HIGH：FDE +5.1%，25/25 无人机改善（巡航系统性偏差更多，无全局层→个性化空间更大）。

### 多假设预测 (Multi-Hypothesis)

K=5 独立预测头 + 置信度评分头，Winner-Takes-All 训练。推理取置信度最高，或用 minFDE_K 评估。

## 目录结构

根目录只放**运行时库**（被到处 import 的核心模块）+ 两个推理入口；脚本按用途分入 `eval/` `train/` `viz/` `tests/`。

```
├── predictor.py              # DronePredictor — 通用推理入口 (软融合 + Z 纠正)
├── deploy.py                 # DeployedLowPredictor — 长轨迹 + 在线学习入口 (三门控)
├── online_config.py          # 在线学习权威配置 (LoRA targets / 权重文件 / 门控阈值)
├── online_learner.py         # OnlineLearner — 流式增量训练
├── adapter_manager.py        # DroneAdapterManager — 逐无人机 adapter 存取
├── streaming.py              # 流式数据缓冲 + 漂移检测
├── lora.py                   # LoRALinear / LoRAAdapter — 低秩适配核心
├── dynamic_norm.py           # 动态归一化
├── fix_labels.py             # UAV-Flow 标签修正
├── emam_model/               # EMAM 模型架构
│   ├── model.py              # TrajectoryPredictor
│   ├── emam_se.py            # Enhanced Mamba + SE (含可选 chunked SSM scan)
│   ├── ia_dtp.py             # Intent-Aware DTP
│   ├── ua_pgd.py             # UA-PGD + 多假设解码器 + 运动学物理模型
│   ├── trigger.py            # 事件触发器
│   └── bidirectional_mamba.py # 双向增强器 (独立实验, 主流程未启用)
├── utils/                    # 数据加载 & 评估指标 & 日志
├── eval/                     # 评估脚本 (evaluate / eval_lora / eval_online_* ...)
├── train/                    # 训练流水线 + LoRA/多假设训练 + 40 帧扩展
├── viz/                      # 可视化 + 诊断 + rollout
├── tests/                    # 回归测试 (deploy 门控 / SSM scan) + 延迟基准
├── weights/                  # 预训练权重 (见下表)
├── pic-results/              # 评估图表 + README.md
└── reviewtodelete/           # 弃置文件暂存 (gitignore, 待人工删除)
```

> 从根目录运行，例如 `python eval/evaluate.py`、`python tests/test_deploy_gates.py`。

**文档：** `README.md`（英文版，架构与结果）、`README_CN.md`（本文，中文版）、`INTERFACE.md`（系统集成接口规范 / ICD）、`ROADMAP.md`（多无人机交互预测下一步规划）。

## 权重文件 (`weights/`)

| 文件 | 说明 |
|:--|:--|
| `low_speed_6class.pth` | LOW 20 帧原始模型（短/中程） |
| `low_speed_6class_40frame.pth` | LOW 40 帧扩展（长轨迹默认 base） |
| `high_speed_4class.pth` | HIGH 20 帧模型 |
| `dir_lora_40.pth` | **当前最佳全局 LoRA**（方向加权，跨飞行 +7.3% / window-level +19.4% FDE） |
| `global_lora_40.pth` | 早期全局 LoRA（+14.9%，已被 dir_lora_40 取代） |
| `gate_lora_40.pth` | 门控 LoRA（改善极端转弯灾难率 -1.76pp） |
| `low_multihead_K5_40frame.pth` | LOW K=5 多假设头 |
| `high_multihead_K5.pth` | HIGH K=5 多假设头 |

## 脚本速查

| 脚本 | 用途 |
|:--|:--|
| `predictor.py` | 通用推理入口（根目录） |
| `deploy.py` | 长轨迹 + 在线学习部署入口（根目录，含冒烟自测） |
| `eval/evaluate.py` | ADE/FDE/意图/不确定性综合评估 |
| `eval/eval_multihead.py` | 多假设完整测试集评估 |
| `eval/eval_lora.py` / `eval/eval_lora_stack.py` | LoRA / 全局+局部叠加评估 |
| `eval/eval_online_learning.py` / `eval/eval_online_learning_high.py` | LOW / HIGH 在线学习验证 |
| `eval/eval_online_vs_offline.py` | 因果条件下 在线 vs 离线批量 对比 |
| `train/expand_model_low.py` / `train/expand_model_high.py` | 20→40 帧扩展实验（LOW 正 / HIGH 负） |
| `train/train_lora_direction.py` / `train/train_lora_gate.py` / `train/train_lora_global.py` | 各类全局 LoRA 训练 |
| `train/train_multihead_low.py` / `train/train_multihead.py` | 多假设解码器训练 (WTA) |
| `viz/visualize_low.py` / `viz/visualize_multihead.py` / `viz/visualize_trajectories.py` / `viz/visualize_summary.py` | 图表生成 |
| `viz/diagnose_failures.py` | 最差样本深度诊断 |
| `viz/rollout.py` | 自回归预测外推 |
| `fix_labels.py` | UAV-Flow 标签修正（根目录） |
| `eval/eval_global_lora_generalization.py` | 全局 LoRA 跨飞行泛化验证（飞行级不相交划分） |
| `tests/test_deploy_gates.py` | 部署三门控 + 在线学习常驻 adapter 回归测试 |
| `tests/test_ssm_scan.py` | chunked SSM scan 数值等价性测试（独立运行 + pytest） |
| `tests/benchmark_latency.py` | 推理延迟基准（两个入口 + 在线更新） |

> 所有脚本都从**项目根目录**运行（如 `python eval/evaluate.py`），脚本内部会自动定位根目录以加载 `weights/` 与数据集。

## 性能指标

### LOW — 完整改进链（长轨迹）

| 阶段 | 长轨迹 FDE | 方向误差 | 灾难率 | 覆盖率 |
|:--|:--:|:--:|:--:|:--:|
| 原始 20 帧 | 2.32m | 54° | 20.0% | 7% |
| 40 帧扩展 | 0.87m | ~15° | ~0% | 84% |
| 40 帧 + LoRA | 0.23m | ~5° | 0% | 84% |

### 多假设 (K=5, minFDE_5 vs 单模型)

| | LOW | HIGH |
|:--|:--:|:--:|
| minFDE_5 | 0.60m (+31.6%) | 1.84m (+66%) |
| minADE_5 | 0.41m (+18.2%) | 0.93m (+46%) |
| FDE P95 | 1.57m | 5.22m |

### 在线学习（因果验证）

| | 冻结 | 在线流式 | 改善 |
|:--|:--:|:--:|:--:|
| LOW FDE | 0.648m | 0.638m | +1.6% (21/30 drones) |
| HIGH FDE | 4.28m | 4.06m | +5.1% (25/25 drones) |

### 全局 LoRA 泛化性（诚实评估）

| 划分方式 | FDE 增益 | 方向增益 | 说明 |
|:--|:--:|:--:|:--|
| window-level（旧） | +19.4% | +12.3% | 同飞行窗口可能同现于训练/测试，偏乐观 |
| **跨飞行（20% 飞行留出）** | **+7.3%** | **+4.6%** | held-out 飞行从未见过，**部署以此为准** |

### 推理延迟（RTX 3060 Laptop, batch=1, 单无人机流）

| 调用路径 | mean | p50 | p95 | 吞吐 |
|:--|:--:|:--:|:--:|:--:|
| `DronePredictor.predict` (20f) | 35.3 ms | 34.2 ms | 44.2 ms | 28 fps |
| `DeployedLowPredictor.predict` (40f, 仅推理) | 20.1 ms | 18.9 ms | 27.3 ms | 50 fps |
| `DeployedLowPredictor.predict` (40f, 含在线更新) | 30.3 ms | 19.9 ms | 117.9 ms | 33 fps |

> 实时裕量：LOW 5Hz（200ms/帧）有 ~10× 裕量，HIGH 1Hz（1000ms/帧）有 ~50× 裕量。在线更新的 p95 尖峰来自每 10 帧触发一次的 LoRA 更新，仍远低于帧预算。

## 模型规格

- 参数量：~1.4M / 模型 (d_model=128, d_state=16)
- 逐无人机在线 LoRA：~100K 参数
- 多假设头：+48K 参数 (K=5)
- 输入：20 或 40 帧 × 6 维 [pos, vel]；输出：20 帧 × 3 维位移
- 硬件：RTX 3060 Laptop (6GB), Windows 11, Python 3.10, CUDA 11.8

## 已知问题与后续方向

- **LOW 方向误差**仍高于 HIGH，多假设 + dir_lora 已改善，仍有空间。
- **极端转弯 (>150°)**：需架构级改动，gate_lora 只能缓解。
- **物理模型 2 帧速度种子**：曾试多帧最小二乘种子，验证为负收益（已否决，勿重试）。
- **全局 LoRA 泛化性**：已用跨飞行划分诚实验证，held-out 飞行 FDE +7.3%（window-level +19.4% 偏乐观）；架构支持一键关闭以应对 OOD。
- 测试覆盖：门控/在线学习/SSM 有回归测试 + 延迟基准，predictor 软融合与 40 帧扩展尚无专门回归测试。

## 许可

仅限研究用途。
