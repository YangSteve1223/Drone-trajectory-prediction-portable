#!/usr/bin/env python3
"""
UAV-Flow 轨迹数据预处理: 生成滑动窗口 + 6类意图标注 (含 HOVER)。

从 extract_uavflow.py 输出的轨迹 .npz 文件生成训练窗口。

意图类别 (6类):
  0: STRAIGHT   — 直线飞行 (航向变化<15°, 垂直速率<2 m/s)
  1: TURN_LEFT  — 左转 (航向变化<-30°)
  2: TURN_RIGHT — 右转 (航向变化>30°)
  3: ASCEND     — 上升 (垂直速率>2 m/s)
  4: DESCEND    — 下降 (垂直速率<-2 m/s)
  5: HOVER      — 悬停 (速度<0.3 m/s, 位移<0.5m)

输出: NPZ chunks (兼容 FastWindowDataset)
  hist:     (N, 20, 6) float32  [x, y, z, vx, vy, vz]
  pred:     (N, 20, 3) float32  [dx, dy, dz] 相对位移
  intent:   (N,)      int32     意图标签 (0-5)
  maneuver: (N,)      int32     机动等级 (0=平稳, 1=普通, 2=剧烈)

用法:
  python preprocess_uavflow.py --traj_dir ./UAV-Flow-trajs --out_dir ./UAV-Flow-windows
  python preprocess_uavflow.py --traj_dir ./UAV-Flow-trajs --out_dir ./UAV-Flow-windows --val_ratio 0.1 --test_ratio 0.1
"""

import numpy as np
import argparse
import json
import time
import shutil
from pathlib import Path

# 窗口参数 (与 baseline 保持一致)
HIST_LEN = 20
PRED_LEN = 20
MIN_LEN = HIST_LEN + PRED_LEN  # 40
DEFAULT_STRIDE = 3  # UAV-Flow 数据量较小, 使用更小的 stride 增加窗口数
CHUNK_SIZE = 5000  # 每个 chunk 的窗口数 (较小以确保有足够 chunk 分配给 train/val/test)


def classify_intent(hist_pos, hist_vel, pred_pos):
    """
    6类意图分类 (向量化版本用于单窗口)。

    Args:
        hist_pos: (20, 3) 历史位置 [x, y, z] (已中心化)
        hist_vel: (20, 3) 历史速度 [vx, vy, vz]
        pred_pos: (20, 3) 预测位置 [x, y, z] (相对 hist_pos[0])

    Returns:
        intent: 0-5
    """
    dt = 0.2  # UAV-Flow is 5Hz

    # 1. 检查 HOVER: 平均速度很低 且 总位移很小
    avg_speed = np.linalg.norm(hist_vel, axis=1).mean()
    total_displacement = np.linalg.norm(pred_pos[-1] - pred_pos[0])

    if avg_speed < 0.3 and total_displacement < 0.5:
        return 5  # HOVER

    # 2. 计算航向角变化
    hist_heading = np.arctan2(hist_vel[:, 1], hist_vel[:, 0])

    if len(pred_pos) > 1:
        pred_vel = np.diff(pred_pos, axis=0) / dt
        pred_heading = np.arctan2(pred_vel[:, 1], pred_vel[:, 0])
    else:
        pred_heading = np.array([])

    all_heading = np.concatenate([hist_heading, pred_heading])

    n_avg = min(5, len(all_heading) // 2)
    heading_start = all_heading[:n_avg].mean()
    heading_end = all_heading[-n_avg:].mean()
    heading_change = heading_end - heading_start
    # 归一化到 [-π, π]
    heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))
    heading_change_deg = np.degrees(heading_change)

    # 3. 计算垂直速率
    if len(pred_pos) > 1:
        vert_rate = np.diff(pred_pos[:, 2]) / dt
        avg_vert_rate = vert_rate.mean()
    else:
        avg_vert_rate = 0.0

    # 4. 分类 (优先级: 垂直 > 转向 > 直线)
    if avg_vert_rate > 0.5:
        return 3  # ASCEND
    elif avg_vert_rate < -0.5:
        return 4  # DESCEND
    elif heading_change_deg < -45:
        return 1  # TURN_LEFT
    elif heading_change_deg > 45:
        return 2  # TURN_RIGHT
    else:
        return 0  # STRAIGHT


def classify_intent_batch(hist, pred):
    """
    批量向量化意图分类 (比逐窗口循环快 ~100x)。

    Args:
        hist: (N, 20, 6) [x, y, z, vx, vy, vz]
        pred: (N, 20, 3) [dx, dy, dz]

    Returns:
        intents: (N,) int32
    """
    N = len(hist)
    dt = 0.2

    # 历史速度
    hv = hist[:, :, 3:6]  # (N, 20, 3)
    hist_speed = np.linalg.norm(hv, axis=2).mean(axis=1)  # (N,) avg speed

    # 总位移
    total_disp = np.linalg.norm(pred[:, -1, :] - pred[:, 0, :], axis=1)  # (N,)

    # HOVER mask
    hover_mask = (hist_speed < 0.3) & (total_disp < 0.5)

    # 航向角
    hist_heading = np.arctan2(hv[:, :, 1], hv[:, :, 0])  # (N, 20)

    # 预测速度
    pred_vel = np.diff(pred, axis=1) / dt  # (N, 19, 3)
    pred_heading = np.arctan2(pred_vel[:, :, 1], pred_vel[:, :, 0])  # (N, 19)

    all_heading = np.concatenate([hist_heading, pred_heading], axis=1)  # (N, 39)

    n_avg = 5
    h_start = all_heading[:, :n_avg].mean(axis=1)
    h_end = all_heading[:, -n_avg:].mean(axis=1)
    heading_change = h_end - h_start
    heading_change_deg = np.degrees(np.arctan2(np.sin(heading_change),
                                                np.cos(heading_change)))

    # 垂直速率
    avg_vert_rate = np.diff(pred[:, :, 2], axis=1).mean(axis=1) / dt  # (N,)

    # 分类 (HOVER已预先标记, 其他按优先级)
    intents = np.full(N, 0, dtype=np.int32)  # 默认 STRAIGHT

    # HOVER
    intents[hover_mask] = 5

    # 非HOVER样本的其他分类
    non_hover = ~hover_mask

    # ASCEND
    ascend_mask = non_hover & (avg_vert_rate > 0.5)
    intents[ascend_mask] = 3

    # DESCEND
    descend_mask = non_hover & ~ascend_mask & (avg_vert_rate < -0.5)
    intents[descend_mask] = 4

    # TURN_LEFT
    left_mask = non_hover & ~ascend_mask & ~descend_mask & (heading_change_deg < -45)
    intents[left_mask] = 1

    # TURN_RIGHT
    right_mask = non_hover & ~ascend_mask & ~descend_mask & ~left_mask & (heading_change_deg > 45)
    intents[right_mask] = 2

    # 其余保持 STRAIGHT (0)

    return intents


def generate_windows(traj, stride=DEFAULT_STRIDE):
    """
    从单条轨迹生成滑动窗口。

    Args:
        traj: (T, 6) float32  [x, y, z, vx, vy, vz]
        stride: 窗口步长

    Returns:
        hists:   (W, 20, 6)
        preds:   (W, 20, 3)
        intents: (W,)
        maneuvers: (W,)
    """
    T = traj.shape[0]
    if T < MIN_LEN:
        return [], [], [], []

    n_windows = (T - MIN_LEN) // stride + 1

    hists = np.zeros((n_windows, HIST_LEN, 6), dtype=np.float32)
    preds = np.zeros((n_windows, PRED_LEN, 3), dtype=np.float32)
    intents = np.zeros(n_windows, dtype=np.int32)
    maneuvers = np.zeros(n_windows, dtype=np.int32)

    for w in range(n_windows):
        start = w * stride
        end_hist = start + HIST_LEN
        end_pred = end_hist + PRED_LEN

        # 历史窗口
        hist_seg = traj[start:end_hist].copy()  # (20, 6)
        hist_pos = hist_seg[:, :3]  # (20, 3)
        origin = hist_pos[0:1, :]

        # 中心化位置
        hist_pos_centered = hist_pos - origin
        hists[w, :, :3] = hist_pos_centered
        hists[w, :, 3:6] = hist_seg[:, 3:6]  # velocity unchanged

        # 预测窗口
        pred_seg = traj[end_hist:end_pred].copy()  # (20, 6)
        pred_pos = pred_seg[:, :3]
        pred_pos_centered = pred_pos - origin
        preds[w] = pred_pos_centered

        # 意图标签
        hist_vel = hist_seg[:, 3:6]
        intents[w] = classify_intent(hist_pos_centered, hist_vel, pred_pos_centered)

        # 机动等级
        avg_speed = np.linalg.norm(hist_vel, axis=1).mean()
        if avg_speed < 0.5:
            maneuvers[w] = 0  # 平稳/悬停
        elif avg_speed < 3:
            maneuvers[w] = 1  # 普通
        else:
            maneuvers[w] = 2  # 剧烈

    return hists, preds, intents, maneuvers


def process_trajectories(traj_dir: Path, out_dir: Path, split_ratios: dict, stride: int = DEFAULT_STRIDE):
    """
    读取所有轨迹 NPZ, 生成窗口, 分配 train/val/test, 保存 chunks。
    """
    traj_files = sorted(traj_dir.glob('*.npz'))
    if not traj_files:
        print(f'Error: No .npz files found in {traj_dir}')
        return

    print(f'Found {len(traj_files)} trajectory files')
    t0 = time.time()

    # 临时目录
    tmp_dir = out_dir / '.tmp_windows'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # 预分配缓冲区
    buf_hist = np.zeros((CHUNK_SIZE, HIST_LEN, 6), dtype=np.float32)
    buf_pred = np.zeros((CHUNK_SIZE, PRED_LEN, 3), dtype=np.float32)
    buf_intent = np.zeros(CHUNK_SIZE, dtype=np.int32)
    buf_maneuver = np.zeros(CHUNK_SIZE, dtype=np.int32)
    buf_fill = 0
    chunk_idx = 0
    total_windows = 0
    skipped_traj = 0
    all_window_intents = []  # 收集用于统计

    def save_chunk():
        nonlocal buf_fill, chunk_idx
        if buf_fill == 0:
            return
        out_path = tmp_dir / f'c{chunk_idx}.npz'
        np.savez_compressed(
            out_path,
            hist=buf_hist[:buf_fill].copy(),
            pred=buf_pred[:buf_fill].copy(),
            intent=buf_intent[:buf_fill].copy(),
            maneuver=buf_maneuver[:buf_fill].copy(),
        )
        sz_mb = out_path.stat().st_size / 1024 / 1024
        print(f'  chunk {chunk_idx}: {buf_fill:,} windows, {sz_mb:.1f} MB')
        chunk_idx += 1
        buf_fill = 0

    # 处理每条轨迹
    for i, tf in enumerate(traj_files):
        try:
            data = np.load(tf, allow_pickle=True)
            traj = data['traj']  # (T, 6)
        except Exception as e:
            print(f'  Warning: skipping {tf.name}: {e}')
            skipped_traj += 1
            continue

        if traj.shape[0] < MIN_LEN:
            skipped_traj += 1
            continue

        hists, preds, intents, maneuvers = generate_windows(traj, stride)

        if len(hists) == 0:
            skipped_traj += 1
            continue

        # 写入缓冲区
        n = len(hists)
        if buf_fill + n > CHUNK_SIZE:
            save_chunk()

        buf_hist[buf_fill:buf_fill + n] = hists
        buf_pred[buf_fill:buf_fill + n] = preds
        buf_intent[buf_fill:buf_fill + n] = intents
        buf_maneuver[buf_fill:buf_fill + n] = maneuvers
        buf_fill += n
        total_windows += n
        all_window_intents.append(intents)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1:5d}/{len(traj_files)}] {total_windows:,} windows, '
                  f'{elapsed:.0f}s, ~{total_windows/elapsed:.0f} w/s')

    save_chunk()
    elapsed = time.time() - t0
    print(f'\n[1] Window generation: {total_windows:,} windows from '
          f'{len(traj_files)-skipped_traj} trajectories ({elapsed:.0f}s)')
    if skipped_traj:
        print(f'    Skipped {skipped_traj} trajectories (too short or errors)')

    # 统计意图分布
    all_intents = np.concatenate(all_window_intents)
    intent_names = ['STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'ASCEND', 'DESCEND', 'HOVER']
    print(f'\n[2] Intent distribution:')
    for k in range(6):
        count = (all_intents == k).sum()
        pct = count / len(all_intents) * 100
        print(f'    {k} {intent_names[k]:12s}: {count:>8,} ({pct:5.1f}%)')

    # 分配 train/val/test
    print(f'\n[3] Splitting into train/val/test...')
    chunks = sorted(tmp_dir.glob('c*.npz'))

    # 计算每个 chunk 的窗口数
    chunk_sizes = []
    for cp in chunks:
        with np.load(cp, mmap_mode='r') as ch:
            chunk_sizes.append(len(ch['hist']))

    total = sum(chunk_sizes)
    val_ratio = split_ratios.get('val', 0.1)
    test_ratio = split_ratios.get('test', 0.1)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)
    n_train = total - n_val - n_test

    # 简单顺序分配 (已随机打乱的话, 也可以先 shuffle chunks)
    rng = np.random.RandomState(42)
    chunk_order = rng.permutation(len(chunks))

    split_map = {'train': [], 'val': [], 'test': []}
    split_counts = {'train': 0, 'val': 0, 'test': 0}

    for ci in chunk_order:
        n = chunk_sizes[ci]
        if split_counts['train'] < n_train:
            split_map['train'].append((ci, n))
            split_counts['train'] += n
        elif split_counts['val'] < n_val:
            split_map['val'].append((ci, n))
            split_counts['val'] += n
        else:
            split_map['test'].append((ci, n))
            split_counts['test'] += n

    # 重命名和移动到输出目录
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, chunk_list in split_map.items():
        for i, (ci, n) in enumerate(chunk_list):
            src = chunks[ci]
            dst = out_dir / f'windows_{split_name}_chunk{i}.npz'
            shutil.move(str(src), str(dst))

        total_mb = sum(
            (out_dir / f'windows_{split_name}_chunk{j}.npz').stat().st_size
            for j in range(len(chunk_list))
        ) / 1024 / 1024
        print(f'  {split_name}: {len(chunk_list)} chunks, '
              f'{split_counts[split_name]:,} windows, {total_mb:.0f} MB')

    # 元信息
    meta = {
        'hist_len': HIST_LEN,
        'pred_len': PRED_LEN,
        'stride': stride,
        'min_len': MIN_LEN,
        'num_intent_classes': 6,
        'intent_names': intent_names,
        'intent_distribution': {intent_names[k]: int((all_intents == k).sum())
                                for k in range(6)},
        'total_windows': total,
    }
    for name in ['train', 'val', 'test']:
        files = sorted(out_dir.glob(f'windows_{name}_chunk*.npz'))
        n_w = sum(np.load(f, mmap_mode='r')['hist'].shape[0] for f in files)
        meta[f'{name}_windows'] = int(n_w)
        meta[f'{name}_chunks'] = len(files)

    with open(out_dir / 'windows_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # 清理
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f'\nDone. Total: {total_windows:,} windows in {out_dir}')
    print(f'  Config: hist={HIST_LEN}, pred={PRED_LEN}, stride={stride}, '
          f'classes={6}, dt=0.2s (5Hz)')


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess UAV-Flow trajectories into training windows')
    parser.add_argument('--traj_dir', type=str, required=True,
                        help='Directory of extracted trajectory .npz files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for window chunks')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE,
                        help=f'Window stride (default: {DEFAULT_STRIDE})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split')
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)

    if not traj_dir.exists():
        print(f'Error: {traj_dir} does not exist. Run extract_uavflow.py first.')
        return

    np.random.seed(args.seed)

    process_trajectories(
        traj_dir,
        out_dir,
        {'val': args.val_ratio, 'test': args.test_ratio},
        stride=args.stride
    )


if __name__ == '__main__':
    main()
