# EMAM 纯高速模型续训 — NPZDATA (raw, 100% >8 m/s)
# Run from gitcommit/ directory
#
# 从 emam_4class_d128_safe/latest.pth (epoch 10) 继续训练
# NPZDATA: 11 train chunks, 165万窗口, 全量 >8 m/s, 无低速污染
#
# 当前状态: epoch 10/50, train_loss=3.37, best RMSE=2.32 (epoch 9)
# 目标: RMSE < 1.0m by epoch 20-25

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.7"

python train.py `
    --data_root ../NPZDATA `
    --num_intent_classes 4 `
    --d_model 128 `
    --batch_size 128 `
    --epochs 50 `
    --lr 3e-4 `
    --warmup_epochs 5 `
    --trigger_mode simple `
    --exp_name emam_4class_d128_safe `
    --checkpoint_dir ./checkpoints `
    --resume checkpoints/emam_4class_d128_safe/latest.pth `
    --loss_intent_weight 0.3 `
    --loss_disp_weight 1.0 `
    --loss_unc_weight 0.05 `
    --num_workers 2
