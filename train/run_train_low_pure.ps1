# EMAM 纯低速模型训练 — UAV-Flow-pure (无高速数据掺入)
# Run from gitcommit/ directory
#
# 从现有低速权重 (emam_uavflow_6class_ft/best.pth) 继续 fine-tune
# 使用主 train.py 的全部安全机制: NaN保护、自动回滚、显存管理
#
# 关键参数:
#   --pretrained: 部分加载权重 (非 --resume, 从头开始 epoch 计数)
#   --lr 1e-5: 微调学习率
#   --no_amp: 关闭 AMP (UAV-Flow 数据上 AMP 容易 NaN)
#   --num_workers 0: UAV-Flow 数据量小, 避免多进程开销

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.7"

python train.py `
    --data_root ../UAV-Flow-pure `
    --num_intent_classes 6 `
    --d_model 128 `
    --batch_size 128 `
    --epochs 30 `
    --lr 1e-5 `
    --warmup_epochs 0 `
    --trigger_mode simple `
    --exp_name emam_uavflow_6class_low_pure `
    --checkpoint_dir ./checkpoints `
    --pretrained checkpoints/emam_uavflow_6class_ft/best.pth `
    --loss_intent_weight 0.3 `
    --loss_disp_weight 1.0 `
    --loss_unc_weight 0.05 `
    --no_amp `
    --num_workers 0
