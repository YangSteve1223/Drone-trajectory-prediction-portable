#!/usr/bin/env python3
"""
从 UAV-Flow parquet 文件中提取轨迹数据，转换为统一格式供后续预处理。

UAV-Flow parquet schema:
  id:        string      — trajectory timestamp ID
  frame_idx: int32       — frame index within trajectory
  image:     struct      — RGB image bytes+path (skipped)
  log:       string(JSON) — {
      raw_logs:          [[pos_x, pos_y, alt, roll, heading, pitch, timestamp], ...]
      preprocessed_logs: [[dx, dy, dz, droll, dheading, dpitch], ...]  (relative)
      instruction:       str — NL flight instruction
      instruction_unified: str — standardized instruction
  }

输出: 每个轨迹一个 .npz 文件，包含:
  traj:      (T, 6) float32  [x, y, z, vx, vy, vz]  (centered, in meters & m/s)
  timestamp: (T,)  float64   Unix timestamps
  instruction: str           自然语言指令
  traj_id:    str            轨迹 ID

用法:
  python extract_uavflow.py --data_dir ./UAV-Flow-data --out_dir ./UAV-Flow-trajs
  python extract_uavflow.py --data_dir ./UAV-Flow-data --out_dir ./UAV-Flow-trajs --max_files 5
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
import pyarrow.parquet as pq
import sys


def parse_raw_logs(raw_logs):
    """
    将 raw_logs 列表转换为轨迹数组。

    raw_logs 每帧 7 个字段:
      0: pos_x    — UTM easting (m)
      1: pos_y    — UTM northing (m)
      2: altitude — 高度 (m)
      3: roll     — 滚转角 (deg)
      4: heading  — 偏航角 (deg, 0=North, 90=East)
      5: pitch    — 俯仰角 (deg, 负=前倾)
      6: timestamp — Unix 时间戳 (s)

    Returns:
      traj: (T, 6) float32  [x, y, z, vx, vy, vz] centered on first frame
      timestamps: (T,) float64
    """
    raw = np.array(raw_logs, dtype=np.float64)  # (T, 7)
    T = raw.shape[0]

    # 提取 position (centered on first frame)
    pos = raw[:, 0:3].copy()                    # (T, 3)
    pos0 = pos[0:1, :]                          # (1, 3) first frame
    pos_centered = pos - pos0                   # (T, 3) relative to start

    # 提取 timestamps
    timestamps = raw[:, 6].copy()

    # 计算速度 (中心差分, 边界用单侧)
    dt = np.diff(timestamps)                    # (T-1,)
    dt = np.maximum(dt, 1e-6)                   # 防止除零
    vel = np.zeros((T, 3), dtype=np.float64)

    # 中心差分 (内部点)
    if T >= 3:
        dt_center = (timestamps[2:] - timestamps[:-2])  # (T-2,)
        dt_center = np.maximum(dt_center, 1e-6)
        vel[1:-1, 0] = (pos[2:, 0] - pos[:-2, 0]) / dt_center
        vel[1:-1, 1] = (pos[2:, 1] - pos[:-2, 1]) / dt_center
        vel[1:-1, 2] = (pos[2:, 2] - pos[:-2, 2]) / dt_center

    # 边界: 前向/后向差分
    vel[0, :] = (pos[1, :] - pos[0, :]) / dt[0]
    vel[-1, :] = (pos[-1, :] - pos[-2, :]) / dt[-1]

    # 合并为 (T, 6): [x, y, z, vx, vy, vz]
    traj = np.concatenate([pos_centered, vel], axis=1).astype(np.float32)

    return traj, timestamps


def extract_file(parquet_path: Path, out_dir: Path, stats: dict):
    """
    从单个 parquet 文件提取所有轨迹。

    关键: 同一个 trajectory ID 在表中重复多次（每帧一行），
    log 列包含完整轨迹的 JSON，所以只需每个 ID 取第一行即可。
    """
    pf = pq.ParquetFile(str(parquet_path))
    # 只读 id 和 log 列, 跳过嵌套的 image 列
    table = pf.read(columns=['id', 'log'])

    ids = table.column('id').to_pylist()
    logs = table.column('log').to_pylist()

    # 去重: 每个 trajectory ID 只处理一次
    seen = set()
    n_extracted = 0

    for row_idx, (tid, log_str) in enumerate(zip(ids, logs)):
        if tid in seen:
            continue
        seen.add(tid)

        try:
            log_data = json.loads(log_str)
        except (json.JSONDecodeError, TypeError) as e:
            stats['json_errors'] += 1
            continue

        raw_logs = log_data.get('raw_logs')
        if not raw_logs or len(raw_logs) < 5:  # 至少5帧才有意义
            stats['too_short'] += 1
            continue

        try:
            traj, timestamps = parse_raw_logs(raw_logs)
        except Exception as e:
            stats['parse_errors'] += 1
            continue

        # 过滤异常轨迹: 位置跳跃过大 (>100m/s 的瞬时速度)
        max_speed = np.max(np.linalg.norm(traj[:, 3:6], axis=1))
        if max_speed > 100:
            stats['anomalous'] += 1
            continue

        # 保存
        instruction = log_data.get('instruction', '')
        safe_id = tid.replace(':', '-').replace('/', '-').replace('\\', '-')
        out_path = out_dir / f'{safe_id}.npz'
        np.savez_compressed(
            out_path,
            traj=traj,
            timestamp=timestamps,
            instruction=np.array(instruction, dtype=object),
            traj_id=np.array(tid, dtype=object),
        )
        n_extracted += 1
        stats['extracted'] += 1

    return n_extracted


def main():
    parser = argparse.ArgumentParser(
        description='Extract trajectories from UAV-Flow parquet files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing *.parquet files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for trajectory .npz files')
    parser.add_argument('--max_files', type=int, default=0,
                        help='Max parquet files to process (0=all)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob('*.parquet'))
    if not parquet_files:
        print(f'Error: No .parquet files found in {data_dir}')
        sys.exit(1)

    if args.max_files > 0:
        parquet_files = parquet_files[:args.max_files]

    print(f'Processing {len(parquet_files)} parquet files...')
    print(f'Output directory: {out_dir}')
    print('=' * 60)

    stats = {
        'extracted': 0, 'json_errors': 0, 'too_short': 0,
        'parse_errors': 0, 'anomalous': 0, 'total_frames': 0,
    }

    for i, pf in enumerate(parquet_files):
        n = extract_file(pf, out_dir, stats)
        print(f'  [{i+1:3d}/{len(parquet_files)}] {pf.name}: {n} trajectories')
        stats['total_frames'] += n

    print('=' * 60)
    print(f'Done. {stats["extracted"]} trajectories saved to {out_dir}')
    print(f'  Skipped: {stats["json_errors"]} json errors, '
          f'{stats["too_short"]} too short, '
          f'{stats["parse_errors"]} parse errors, '
          f'{stats["anomalous"]} anomalous')


if __name__ == '__main__':
    main()
