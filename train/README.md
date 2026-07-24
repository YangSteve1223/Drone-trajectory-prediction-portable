# Training Guide

Run scripts from project root, e.g. `python train/train.py ...`.

## Data Preparation

### UAV-Flow (real DJI, low-speed, 5Hz)

```bash
python train/extract_uavflow.py --output ../UAV-Flow-trajs
python train/preprocess_uavflow.py --traj_dir ../UAV-Flow-trajs --out_dir ../UAV-Flow-windows
```

### Custom Data

`.npz` files, each containing:
- `hist`: (N, 20, 6) float32 — history [pos_x, pos_y, pos_z, vx, vy, vz]
- `pred`: (N, 20, 3) float32 — future displacement
- `intent`: (N,) int32 — intent label
- `maneuver`: (N,) int32 — maneuver level (optional)

File naming: `windows_train_chunk*.npz`, `windows_val_chunk*.npz`, `windows_test_chunk*.npz`

## Training

### LOW (UAV-Flow, 6-class)

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"
python train/train.py --data_root ../UAV-Flow-pure --num_intent_classes 6 `
    --d_model 128 --batch_size 128 --epochs 30 --lr 1e-4 `
    --trigger_mode simple --exp_name my_low_speed `
    --loss_intent_weight 0.3 --num_workers 0
```

### HIGH (SimCruise, 4-class)

```powershell
python train/train.py --data_root ../SimCruise --num_intent_classes 4 `
    --d_model 128 --batch_size 128 --epochs 50 --lr 3e-4 `
    --warmup_epochs 5 --trigger_mode simple --exp_name my_high_speed `
    --loss_intent_weight 0.3
```

### Resume / Warm-start

```powershell
# From pretrained weights
python train/train.py ... --pretrained ../weights/low_speed_6class.pth

# Resume from checkpoint
python train/train.py ... --resume ../checkpoints/my_model/latest.pth
```

### Bidirectional Mamba Enhancer

```powershell
python train/train_bidir.py --epochs 5 --batch_size 128 --data_dir ../UAV-Flow-pure
```

## Key Parameters

| Param | Purpose | Suggested |
|:--|:--|:--|
| `--d_model` | Model dim | 128 (6GB VRAM) / 256 |
| `--batch_size` | Batch size | 128 (6GB) / 256 (12GB+) |
| `--lr` | Learning rate | 1e-4 ~ 3e-4 |
| `--warmup_epochs` | Warmup epochs | 5 |
| `--trigger_mode` | Trigger mode | `simple` (always on) |
| `--pretrained` | Pretrained weights | Path to .pth |
| `--resume` | Resume checkpoint | Path to latest.pth |
| `--no_amp` | Disable mixed precision | Recommended for UAV-Flow |
| `--num_workers` | Data loader workers | 0 (UAV-Flow) / 2-4 (SimCruise) |

## Output

`../checkpoints/<exp_name>/`:
- `best.pth` — best validation weights
- `latest.pth` — latest weights (for resume)
- `epoch_*.pth` — per-epoch checkpoint
- `config.yaml` — training config

Trained weights can be placed in `../weights/` or specified via `DronePredictor` constructor args.
