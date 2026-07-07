# EMAM 6GB-safe training — RTX 3060 Laptop (6 GB VRAM)
# Run from gitcommit/ directory
#
# 关键改动 vs 原始配置:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True → 消除碎片化 (根因修复)
#   2. cudnn.benchmark=OFF → 限制 workspace 膨胀 (train.py 已内置)
#   3. batch_size=128 保持不变 (expandable_segments 修的是碎片化，不是峰值)
#   4. lr=3e-4 (比原来低一点，减少 Infinity 风险)
#   5. num_workers=2 (减少锁页内存压力)
#   6. 更频繁的 empty_cache (train.py 已内置)

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.7"

# 续训模式 (从 latest checkpoint 恢复)
python train.py --data_root ../SimCruise --num_intent_classes 4 --d_model 128 --batch_size 128 --grad_accum 1 --epochs 50 --lr 3e-4 --loss_intent_weight 0.3 --trigger_mode simple --exp_name emam_4class_d128_safe --checkpoint_dir ./checkpoints --num_workers 2 --resume checkpoints/emam_4class_d128_safe/latest.pth
