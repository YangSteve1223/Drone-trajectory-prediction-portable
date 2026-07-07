#!/usr/bin/env python3
"""
Fix UAV-Flow label noise: trajectories with >40 deg turns labeled as STRAIGHT.
Corrects STRAIGHT -> TURN_L or TURN_R based on cumulative heading change.

INTENT_NAMES_6 = ['STRAIGHT', 'TURN_L', 'TURN_R', 'ASCEND', 'DESC', 'HOVER']
STRAIGHT=0, TURN_L=1, TURN_R=2
"""

import numpy as np
from pathlib import Path
import shutil, sys

TURN_THRESHOLD_DEG = 40  # cumulative heading change to classify as turn
DT = 0.2  # 5Hz

def compute_total_turn(hist, pred_delta):
    """
    Compute cumulative heading change over full trajectory (hist + pred).
    Returns (total_turn_deg, direction): positive=right, negative=left
    """
    # Reconstruct full absolute positions
    last_pos = hist[-1, :3].copy()
    pred_abs = np.zeros((len(pred_delta) + 1, 3))
    pred_abs[0] = last_pos
    for i in range(len(pred_delta)):
        pred_abs[i + 1] = pred_abs[i] + pred_delta[i]

    # Full trajectory: history positions + predicted positions
    full_pos = np.concatenate([hist[:, :3], pred_abs[1:]], axis=0)

    # Compute velocity directions (heading)
    vel = np.diff(full_pos, axis=0)
    headings = np.arctan2(vel[:, 1], vel[:, 0])
    headings_unwrapped = np.unwrap(headings)

    # Total absolute turn (for threshold)
    heading_diffs = np.abs(np.diff(headings_unwrapped))
    total_turn_deg = np.degrees(heading_diffs.sum())

    # Net heading change (for direction: positive=right, negative=left)
    net_turn_deg = np.degrees(headings_unwrapped[-1] - headings_unwrapped[0])

    return total_turn_deg, net_turn_deg


def correct_chunk(filepath, backup_dir, threshold_deg=TURN_THRESHOLD_DEG):
    """Correct labels in a single chunk file."""
    data = np.load(filepath)
    hist = data['hist']
    pred = data['pred']
    intent = data['intent'].copy()

    n = len(hist)
    corrections = 0
    correction_details = []  # (idx, old_label, new_label, turn_deg)

    for i in range(n):
        if intent[i] != 0:  # Only fix STRAIGHT (0) labels
            continue

        total_turn, net_turn = compute_total_turn(hist[i], pred[i])

        if total_turn > threshold_deg:
            # Use net direction for left/right, but require net > 15 deg
            if abs(net_turn) > 15:
                new_label = 2 if net_turn > 0 else 1  # TURN_R / TURN_L
            else:
                new_label = 2 if net_turn >= 0 else 1  # ambiguous: use sign
            corrections += 1
            correction_details.append((i, 0, new_label, total_turn, net_turn))
            intent[i] = new_label

    if corrections > 0:
        # Backup original
        backup_path = backup_dir / filepath.name
        shutil.copy2(filepath, backup_path)

        # Save corrected (preserve all original arrays)
        save_dict = {}
        for key in data.keys():
            if key == 'intent':
                save_dict[key] = intent
            else:
                save_dict[key] = data[key]
        np.savez_compressed(filepath, **save_dict)

    return corrections, correction_details


def main():
    data_dir = Path('../UAV-Flow-pure')
    backup_dir = data_dir / 'label_backup'
    backup_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob('windows_*.npz'))
    print(f'Found {len(all_files)} chunk files')
    print(f'Turn threshold: {TURN_THRESHOLD_DEG} deg')
    print(f'Backup directory: {backup_dir}')
    print()

    total_corrected = 0
    total_samples = 0

    for f in all_files:
        split = 'unknown'
        if '_train_' in f.name:
            split = 'train'
        elif '_val_' in f.name:
            split = 'val'
        elif '_test_' in f.name:
            split = 'test'

        corrections, details = correct_chunk(f, backup_dir)
        data = np.load(f)
        n = len(data['intent'])
        data.close()
        total_samples += n
        total_corrected += corrections

        if corrections > 0:
            # Show distribution of corrections
            to_left = sum(1 for d in details if d[2] == 1)
            to_right = sum(1 for d in details if d[2] == 2)
            turns = [d[3] for d in details]
            print(f'{f.name:40s} [{split:5s}] {n:6d} samples -> {corrections:4d} corrected '
                  f'(L={to_left} R={to_right}) turn range=[{min(turns):.0f}-{max(turns):.0f}]deg')
        else:
            print(f'{f.name:40s} [{split:5s}] {n:6d} samples -> no corrections needed')

    print()
    print(f'Total: {total_corrected}/{total_samples} samples corrected '
          f'({total_corrected/total_samples*100:.2f}%)')
    print(f'Backups saved to: {backup_dir}')
    print('Done!')


if __name__ == '__main__':
    main()
