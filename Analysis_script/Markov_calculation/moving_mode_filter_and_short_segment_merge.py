#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moving_Mode_Filter.py  -- Perform WIN-point moving mode filtering on the first NUM_OF_NITRO molecules
Usage:  python Moving_Mode_Filter.py input.txt
"""
import sys, os
import numpy as np
from scipy.ndimage import generic_filter

FRAME_DT     = 0.001     # ns per frame
NUM_OF_NITRO = 95        # Number of nitrogen molecules to process    
LMIN_FRAMES  = 200       # Minimum duration frames
WIN          = 61        # Window size (odd)
targeted_ID  = 1         # Target cluster molecule ID

def mode_filter(x):
    """Return mode in window; if tie, take minimum (deterministic)."""
    vals, counts = np.unique(x, return_counts=True)
    return vals[counts.argmax()]

def moving_mode_file(in_path, out_path):
    """Generate 61-point moving mode filtered file"""
    data = np.loadtxt(in_path, skiprows=1)
    t = data[:, 0]
    traj = data[:, 1:NUM_OF_NITRO+1].astype(int)
    filt = np.empty_like(traj)
    for col in range(NUM_OF_NITRO):
        filt[:, col] = generic_filter(traj[:, col], mode_filter, size=WIN, mode='mirror')
    with open(out_path, 'w') as fo:
        header_cols = '\t'.join(map(str, range(1, NUM_OF_NITRO+1)))
        fo.write(f'#time(ns)\t{header_cols}\n')
        for i in range(len(t)):
            fo.write(f'{t[i]:.3f}\t' + '\t'.join(map(str, filt[i, :])) + '\n')
    print(f'[info] 61-point filtering complete -> {out_path}')
    return out_path

def merge_short_segments(traj, lmin=LMIN_FRAMES):
    """
    Perform short segment merging on single trajectory (1D int array)
    Returns merged new trajectory
    """
    n = len(traj)
    out = traj.copy()
    start = 0
    while start < n:
        clu = traj[start]
        end = start
        while end < n and traj[end] == clu:
            end += 1
        seg_len = end - start
        if seg_len < lmin:
            # Need to merge
            if start == 0:               # First segment insufficient, align with next
                new_clu = traj[end] if end < n else clu
            else:                        # Otherwise align with previous
                new_clu = out[start - 1]
            out[start:end] = new_clu
        start = end
    return out

def process(in_path, do_filter=False):
    base, ext = os.path.splitext(in_path)
    mode_file = f'{base}_mode{str(WIN)}_{targeted_ID}{ext}'
    out_file  = f'{base}_mode{str(WIN)}_merge{str(LMIN_FRAMES)}_{targeted_ID}{ext}'

    if do_filter:
        moving_mode_file(in_path, mode_file)
    else:
        mode_file = in_path

    data = np.loadtxt(mode_file, skiprows=1)
    t = data[:, 0]
    traj = data[:, 1:NUM_OF_NITRO+1].astype(int)

    merged = np.empty_like(traj)
    for col in range(NUM_OF_NITRO):
        merged[:, col] = merge_short_segments(traj[:, col], LMIN_FRAMES)

    with open(out_file, 'w') as fo:
        header_cols = '\t'.join(map(str, range(1, NUM_OF_NITRO+1)))
        fo.write(f'#time(ns)\t{header_cols}\n')
        for i in range(len(t)):
            fo.write(f'{t[i]:.3f}\t' + '\t'.join(map(str, merged[i, :])) + '\n')
    print(f'[info] Short segment merge complete -> {out_file}')

def main():
    do_filter = '--filter' in sys.argv
    in_file = sys.argv[1] if len(sys.argv) > 1 else 'raw_cluster.txt'
    if not os.path.isfile(in_file):
        sys.exit(f'Input file not found: {in_file}')
    process(in_file, do_filter=do_filter)

if __name__ == '__main__':
    main()