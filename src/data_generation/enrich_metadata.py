# src/data_generation/enrich_metadata.py
"""
Backfills system property columns into existing metadata CSVs.

Reads each saved .npy signal, smooths the noisy output with a Savitzky-Golay
filter, then computes:  steady_state_gain, rise_time, settling_time, overshoot_pct

Run from the project root:
    python src/data_generation/enrich_metadata.py
    python src/data_generation/enrich_metadata.py --data data/
"""

import os
import csv
import argparse
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

N_POINTS = 2000
T_FINAL  = 100.0
DT       = T_FINAL / N_POINTS

SMOOTH_WINDOW = 51   # ~2.5 s at DT=0.05 — removes noise, preserves shape
SMOOTH_POLY   = 3

NEW_COLUMNS = ['steady_state_gain', 'rise_time', 'settling_time', 'overshoot_pct']


def compute_system_properties(y):
    n = len(y)
    t = np.arange(n) * DT

    y_ss = float(np.mean(y[int(0.9 * n):]))
    if abs(y_ss) < 1e-9:
        return {'steady_state_gain': 0.0, 'rise_time': T_FINAL,
                'settling_time': T_FINAL, 'overshoot_pct': 0.0}

    y10, y90 = 0.1 * y_ss, 0.9 * y_ss
    if y_ss > 0:
        idx10 = int(np.argmax(y >= y10))
        idx90 = int(np.argmax(y >= y90))
    else:
        idx10 = int(np.argmax(y <= y10))
        idx90 = int(np.argmax(y <= y90))
    rise_time = float(t[idx90] - t[idx10]) if idx90 > idx10 else float(T_FINAL)

    outside = np.where(np.abs(y - y_ss) > 0.02 * abs(y_ss))[0]
    settling_time = float(t[outside[-1]]) if len(outside) > 0 else 0.0

    if y_ss > 0:
        overshoot_pct = float(max(0.0, (np.max(y) - y_ss) / y_ss * 100))
    else:
        overshoot_pct = float(max(0.0, (y_ss - np.min(y)) / abs(y_ss) * 100))

    return {
        'steady_state_gain': round(y_ss, 4),
        'rise_time':         round(rise_time, 4),
        'settling_time':     round(settling_time, 4),
        'overshoot_pct':     round(overshoot_pct, 4),
    }


def enrich_split(data_dir, split):
    meta_path = os.path.join(data_dir, f'metadata_{split}.csv')
    if not os.path.exists(meta_path):
        print(f'  [{split}] metadata not found, skipping.')
        return

    with open(meta_path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f'  [{split}] empty metadata, skipping.')
        return

    already_done = all(col in rows[0] for col in NEW_COLUMNS)
    if already_done:
        print(f'  [{split}] columns already present — overwriting with fresh values.')

    errors = 0
    for row in tqdm(rows, desc=f'  {split}', unit='sample'):
        sig_path = os.path.join(
            data_dir, 'signals', split, row['class'], row['filename'] + '.npy'
        )
        if not os.path.exists(sig_path):
            for col in NEW_COLUMNS:
                row[col] = ''
            errors += 1
            continue

        sig = np.load(sig_path)
        y_noisy = sig[1].astype(np.float64)

        # Smooth before computing — removes measurement noise
        y_smooth = savgol_filter(y_noisy, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLY)

        props = compute_system_properties(y_smooth)
        row.update(props)

    # Write updated CSV preserving original column order + new columns at end
    original_cols = list(rows[0].keys())
    for col in NEW_COLUMNS:
        if col not in original_cols:
            original_cols.append(col)

    with open(meta_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=original_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f'  [{split}] {len(rows)} rows updated  ({errors} signal files missing)')


def main(data_dir):
    print(f'\nEnriching metadata in: {data_dir}\n')
    for split in ('train', 'val', 'test'):
        enrich_split(data_dir, split)
    print('\nDone.')


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(project_root, 'data'))
    args = parser.parse_args()
    main(args.data)
