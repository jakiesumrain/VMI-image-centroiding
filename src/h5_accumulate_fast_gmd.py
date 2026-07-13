#!/usr/bin/env python3
"""
Accumulate all frames with threshold + morphological denoising + optional GMD scaling.

GMD scaling: each frame's contribution is weighted by mean_gmd / gmd_of_that_frame,
so frames from weaker FEL pulses get proportionally higher weight.

This uses the pre-computed gmd_map_all.txt (frame_idx → GMD value).
"""

import h5py
import numpy as np
import cv2
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- config ---
h5_dir       = r"H:\Xe_sig"
output_path  = r"H:\Xe_sig\accumulated_result.txt"
im_threshold = 5
do_denoise   = True
kernel       = np.ones((2, 2), dtype=np.uint8)
max_workers  = 6

# GMD scaling (set to False to skip)
gmd_map_path = r"E:\work\projects\VMI-FEL\cpvmi\data_20260703\gmd_map_all.txt"
enable_gmd   = False

# GMD weight range: only include frames with weight in [gmd_w_min, gmd_w_max]
# Frames outside this range are discarded (weight = 0).
# Set both to 0 to disable filtering (use all frames).
gmd_w_min = 0.0
gmd_w_max = 4.0

# Dark frame (.npy file from centroiding/compute_dark.py)
# Set to empty string to skip dark subtraction
dark_path    = r"E:\work\projects\VMI-FEL\cpvmi\data_20260703\centroiding\dark_mean.npy"
# --------------

# worker-process globals — set once via initializer, never pickled per-task
_dark_frame  = None
_gmd_weights = None

def _worker_init(dark_frame, gmd_weights):
    global _dark_frame, _gmd_weights
    _dark_frame  = dark_frame
    _gmd_weights = gmd_weights


def load_gmd_weights(path, n_total_frames):
    """Load GMD map and return weight per global frame index.

    weight = mean_gmd / gmd_value  (linear normalization)
    Frames not in the map get weight = 1.0.
    """
    print(f"Loading GMD map: {path}")
    map_data = np.loadtxt(path, comments='#', usecols=(0, 4))
    frame_idx = map_data[:, 0].astype(int)
    gmd_vals = map_data[:, 1]

    mean_gmd = gmd_vals.mean()
    # Clamp near-zero GMD values (measurement noise floor ~ ±1e-11)
    # 57 frames in method 1, 626 in method 2 have slightly negative values
    gmd_safe = np.maximum(gmd_vals, 1e-12)
    weights = np.ones(n_total_frames, dtype=np.float64)
    weights[frame_idx] = mean_gmd / gmd_safe
    # Filter by weight range: zero out frames outside [gmd_w_min, gmd_w_max]
    if gmd_w_min > 0 or gmd_w_max > 0:
        if gmd_w_max > 0:
            weights[weights > gmd_w_max] = 0.0
        if gmd_w_min > 0:
            weights[weights < gmd_w_min] = 0.0
        n_discarded = int((weights == 0).sum())
        n_kept = n_total_frames - n_discarded
        print(f"  Weight filter [{gmd_w_min}, {gmd_w_max}]: "
              f"kept {n_kept}/{n_total_frames} frames ({n_discarded} discarded)")
    n_nan = np.isnan(weights).sum()
    if n_nan:
        weights[np.isnan(weights)] = 1.0
    print(f"  {len(frame_idx)} frames mapped, mean_gmd={mean_gmd:.3e}, "
          f"weight range [{weights[frame_idx].min():.3f}, {weights[frame_idx].max():.3f}]")
    return weights


def process_h5(args):
    """Accumulate all frames in one h5 file with dark subtraction + threshold + denoising + GMD weight."""
    fpath, base_frame_idx = args

    with h5py.File(fpath, 'r') as hf:
        data = hf['raw/data'][:]                   # (N, H, W) uint16

    n = data.shape[0]
    if n == 0:
        return np.zeros(data.shape[1:], dtype=np.float64), 0

    # --- dark subtraction + ensure float64 for in-place ops ---
    if _dark_frame is not None:
        data = data.astype(np.float64) - _dark_frame
    else:
        data = data.astype(np.float64)

    # --- threshold (vectorized) — works on float64, negatives are < 1 ---
    mask = data >= im_threshold                    # (N, H, W) bool

    # --- morphological denoise ---
    if do_denoise:
        mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
        for i in range(n):
            mask_u8[i] = cv2.morphologyEx(mask_u8[i], cv2.MORPH_OPEN, kernel, iterations=1)
        mask = mask_u8.astype(bool)

    # --- zero below-threshold pixels in place (no large temporary array) ---
    data[~mask] = 0.0
    if _gmd_weights is not None:
        w = _gmd_weights[base_frame_idx:base_frame_idx + n].reshape(-1, 1, 1)
        data *= w  # in-place multiply, no extra full-array allocation
    acc = data.sum(axis=0, dtype=np.float64)
    return acc, n


def main():
    files = sorted([
        os.path.join(h5_dir, f)
        for f in os.listdir(h5_dir) if f.endswith('.h5')
    ])
    n_files = len(files)
    workers = min(max_workers, n_files) if n_files else 1
    print(f"Found {n_files} h5 files, processing with {workers} workers (processes)...")

    # Pre-count total frames for GMD weight array
    n_total = 0
    for f in files:
        with h5py.File(f, 'r') as hf:
            n_total += hf['raw/data'].shape[0]

    # Load dark frame
    dark_frame = np.load(dark_path) if dark_path and os.path.exists(dark_path) else None
    if dark_frame is not None:
        print(f"Dark frame loaded: mean={dark_frame.mean():.1f}, std={dark_frame.std():.1f}")
    else:
        print(f"Dark subtraction DISABLED (dark_path={dark_path})")

    # Load GMD weights once
    gmd_weights = load_gmd_weights(gmd_map_path, n_total) if enable_gmd else None
    if gmd_weights is not None:
        print(f"GMD weighting ENABLED")
    else:
        print(f"GMD weighting DISABLED")

    # Build args with base frame indices
    batch_args = []
    idx = 0
    for f in files:
        with h5py.File(f, 'r') as hf:
            n = hf['raw/data'].shape[0]
        batch_args.append((f, idx))
        idx += n

    t_start = time.time()
    total_frames = 0
    total_acc = None

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init,
                             initargs=(dark_frame, gmd_weights)) as executor:
        futures = {executor.submit(process_h5, args): args[0] for args in batch_args}
        with tqdm(total=n_files, desc="Processing H5 files") as pbar:
            for fut in as_completed(futures):
                try:
                    acc, n_frames = fut.result()
                    total_frames += n_frames
                    if total_acc is None:
                        total_acc = acc
                    else:
                        total_acc += acc
                except Exception as e:
                    print(f"\n[Error] {e}")
                pbar.update(1)

    t_end = time.time()
    if total_acc is None:
        print("No files processed successfully; nothing to save.")
        return
    print(f"\n{total_frames} frames processed in {t_end - t_start:.1f}s")
    print(f"Sum of all counts: {np.sum(total_acc):.0f}")

    np.savetxt(output_path, total_acc)
    print(f"Saved to {output_path}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    plt.imshow(total_acc, origin='lower')
    plt.colorbar(label='weighted counts')
    gmd_label = "ON" if gmd_weights is not None else "OFF"
    plt.title(f'Accumulated ({total_frames} frames, GMD={gmd_label})')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()