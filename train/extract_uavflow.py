#!/usr/bin/env python3
"""Extract trajectory data from UAV-Flow parquet files into a unified format.

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

Output: one .npz per trajectory with:
  traj:      (T, 6) float32  [x, y, z, vx, vy, vz]  (centered, meters & m/s)
  timestamp: (T,)  float64   Unix timestamps
  instruction: str
  traj_id:    str

Usage:
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
    """Convert raw_logs list to trajectory array.

    raw_logs fields per frame (7):
      0: pos_x    — UTM easting (m)
      1: pos_y    — UTM northing (m)
      2: altitude — (m)
      3: roll     — (deg)
      4: heading  — yaw (deg, 0=North, 90=East)
      5: pitch    — (deg, negative=nose down)
      6: timestamp — Unix time (s)

    Returns:
      traj: (T, 6) float32  [x, y, z, vx, vy, vz] centered on first frame
      timestamps: (T,) float64
    """
    raw = np.array(raw_logs, dtype=np.float64)  # (T, 7)
    T = raw.shape[0]

    # Position (centered on first frame)
    pos = raw[:, 0:3].copy()                    # (T, 3)
    pos0 = pos[0:1, :]                          # (1, 3) first frame
    pos_centered = pos - pos0                   # (T, 3) relative to start

    # Timestamps
    timestamps = raw[:, 6].copy()

    # Velocity (central difference, one-sided at boundaries)
    dt = np.diff(timestamps)                    # (T-1,)
    dt = np.maximum(dt, 1e-6)                   # avoid div-by-zero
    vel = np.zeros((T, 3), dtype=np.float64)

    # Central difference (interior points)
    if T >= 3:
        dt_center = (timestamps[2:] - timestamps[:-2])  # (T-2,)
        dt_center = np.maximum(dt_center, 1e-6)
        vel[1:-1, 0] = (pos[2:, 0] - pos[:-2, 0]) / dt_center
        vel[1:-1, 1] = (pos[2:, 1] - pos[:-2, 1]) / dt_center
        vel[1:-1, 2] = (pos[2:, 2] - pos[:-2, 2]) / dt_center

    # Boundary: forward/backward difference
    vel[0, :] = (pos[1, :] - pos[0, :]) / dt[0]
    vel[-1, :] = (pos[-1, :] - pos[-2, :]) / dt[-1]

    # Combine into (T, 6): [x, y, z, vx, vy, vz]
    traj = np.concatenate([pos_centered, vel], axis=1).astype(np.float32)

    return traj, timestamps


def extract_file(parquet_path: Path, out_dir: Path, stats: dict):
    """Extract all trajectories from a single parquet file.

    Note: each trajectory ID repeats (one row per frame), but the log column
    holds the full trajectory JSON, so only the first row per ID is needed.
    """
    pf = pq.ParquetFile(str(parquet_path))
    # Read only id and log columns, skip nested image column
    table = pf.read(columns=['id', 'log'])

    ids = table.column('id').to_pylist()
    logs = table.column('log').to_pylist()

    # Dedup: process each trajectory ID once
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
        if not raw_logs or len(raw_logs) < 5:  # need at least 5 frames
            stats['too_short'] += 1
            continue

        try:
            traj, timestamps = parse_raw_logs(raw_logs)
        except Exception as e:
            stats['parse_errors'] += 1
            continue

        # Filter anomalous trajectories (instant speed >100 m/s)
        max_speed = np.max(np.linalg.norm(traj[:, 3:6], axis=1))
        if max_speed > 100:
            stats['anomalous'] += 1
            continue

        # Save
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
