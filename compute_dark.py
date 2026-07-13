#!/usr/bin/env python3
"""
Compute a dark frame from all dark HDF5 files and save to disk.

Reads configuration from centroiding/config.toml (same as main analysis scripts).

Two methods (set dark_method in config.toml):
  mean:   Streaming running sum — uses all dark files, ~8 MB memory
  median: Parallel subsample — uses dark_frame_count frames, ~4 GB memory

Output: dark_{method}.npy (1024×1024 float64) — can be loaded instantly
        in the main analysis script via np.load().

Usage:
  uv run python compute_dark.py
  # Then in main analysis scripts, the dark frame can be loaded directly
  # instead of recomputing each time.
"""

import os
import sys
import glob
import time
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Load config from centroiding/config.toml
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python 3.6-3.10
    except ImportError:
        print("[ERROR] TOML parser not available! Install tomli.")
        sys.exit(1)

config_path = os.path.join(os.path.dirname(__file__), "config.toml")
if not os.path.exists(config_path):
    print(f"[ERROR] Config not found: {config_path}")
    sys.exit(1)

with open(config_path, "rb") as f:
    config = tomllib.load(f)

vmi_cfg = config["vmi_picasso"]
proc_cfg = vmi_cfg["processing"]

DARK_DATA_DIR = vmi_cfg.get("dark_data_dir", "")
H5_PATTERN = vmi_cfg.get("h5_pattern", "RAW-*.h5")
H5_DATASET = vmi_cfg.get("h5_dataset", "raw/data")
METHOD = proc_cfg.get("dark_method", "median")
DARK_FRAME_COUNT = proc_cfg.get("dark_frame_count", 2000)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

if not DARK_DATA_DIR:
    print("[ERROR] dark_data_dir not set in config.toml")
    sys.exit(1)


def _sum_h5_file(fpath):
    """Worker: return (sum_of_frames, n_frames) for one H5 file."""
    with h5py.File(fpath, 'r') as hf:
        data = hf[H5_DATASET][:]
    return data.sum(axis=0, dtype=np.float64), data.shape[0]


def _read_h5_data(fpath):
    """Worker: return full data array from one H5 file."""
    with h5py.File(fpath, 'r') as hf:
        return hf[H5_DATASET][:]


def compute_mean(dark_files):
    """Compute mean dark frame from all files (streaming, low memory)."""
    n_files = len(dark_files)
    print(f"[Mean] Processing {n_files} files...")
    running_sum = None
    n_total = 0
    t0 = time.time()
    for idx, fpath in enumerate(dark_files, 1):
        try:
            with h5py.File(fpath, 'r') as hf:
                data = hf[H5_DATASET][:]
            if running_sum is None:
                running_sum = data.sum(axis=0, dtype=np.float64)
            else:
                running_sum += data.sum(axis=0, dtype=np.float64)
            n_total += data.shape[0]
            if idx % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx}/{n_files}] {n_total} frames, "
                      f"{n_total/elapsed:.0f} frames/s")
        except Exception as e:
            print(f"  WARNING: {os.path.basename(fpath)}: {e}")
    dark = running_sum / n_total
    print(f"[Mean] Done: {n_total} frames in {time.time()-t0:.1f}s")
    return dark, n_total


def compute_median(dark_files):
    """Compute median of subsampled dark frames (parallel read)."""
    with h5py.File(dark_files[0], 'r') as hf:
        frames_per_file = hf[H5_DATASET].shape[0]
    n_needed = int(np.ceil(DARK_FRAME_COUNT / frames_per_file))
    selected = dark_files[:min(n_needed, len(dark_files))]
    n_workers = min(len(selected), cpu_count())
    print(f"[Median] Reading {len(selected)} files ({n_workers} workers)...")
    t0 = time.time()
    all_frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_read_h5_data, f): f for f in selected}
        for fut in as_completed(futures):
            try:
                all_frames.append(fut.result())
            except Exception as e:
                print(f"  WARNING: {e}")
    if not all_frames:
        raise RuntimeError("No dark frames could be read!")
    stack = np.concatenate(all_frames, axis=0)[:DARK_FRAME_COUNT]
    dark = np.median(stack, axis=0).astype(np.float64)
    print(f"[Median] Done: {stack.shape[0]} frames in {time.time()-t0:.1f}s")
    return dark, stack.shape[0]


def main():
    t0 = time.time()
    pattern = os.path.join(DARK_DATA_DIR, H5_PATTERN)
    print(f"Searching: {pattern}")
    dark_files = sorted(glob.glob(pattern))
    if not dark_files:
        print(f"ERROR: No HDF5 files found in {DARK_DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(dark_files)} dark HDF5 files (method={METHOD})")

    if METHOD == "median":
        dark, n_used = compute_median(dark_files)
    else:
        dark, n_used = compute_mean(dark_files)

    out_name = f"dark_{METHOD}.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.save(out_path, dark)
    print(f"Saved: {out_path}")
    print(f"  shape={dark.shape}  dtype={dark.dtype}  "
          f"mean={dark.mean():.1f}  std={dark.std():.1f}")

    info_path = os.path.join(OUTPUT_DIR, f"dark_{METHOD}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"method={METHOD}\n")
        f.write(f"n_frames={n_used}\n")
        f.write(f"mean={dark.mean():.6f}\n")
        f.write(f"std={dark.std():.6f}\n")
        f.write(f"min={dark.min():.0f}\n")
        f.write(f"max={dark.max():.0f}\n")
        f.write(f"computed_seconds={time.time()-t0:.1f}\n")
    print(f"Saved: {info_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
