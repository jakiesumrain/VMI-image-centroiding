#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMI Photoelectron Analysis - PICASSO V5 (Batch HDF5, LQ/MLE)
==============================================================
Batch-optimized version of V4 with configurable fitting method:
  lq:  Least Squares (fast, robust, no divergence issues)
  mle: Maximum Likelihood (Poisson optimal, slower, needs tuning)

Key difference from V3:
- Single movie (N_frames, H, W) → Picasso identify/get_spots
  instead of per-frame (1, H, W) → Picasso
- Eliminates 125× redundant Python→numba boundary overhead per file
- ~3-5× speedup for large datasets

Data model:
  HDF5 file structure:
    raw/data  -> (N_frames, height, width)  uint16
    raw/meta  -> (N_frames, N_columns)       uint64

Author: TAO Jianfei (jakiesumrain@163.com)
Date: 2026-07-03 (v5: LQ/MLE configurable fitting)
"""

from __future__ import annotations
import os
import sys
import glob
import pandas as pd
import numpy as np

# Load configuration from TOML file
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python 3.6-3.10
    except ImportError:
        print("[ERROR] TOML parser not available!")
        print("Install with: pip install tomli")
        sys.exit(1)

# Set Picasso CPU utilization BEFORE importing Picasso to suppress warning
os.environ['PICASSO_CPU_UTILIZATION'] = '0.8'

# Set matplotlib backend
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# For per-frame background estimation when no dark frame available
from astropy.stats import sigma_clipped_stats

# Suppress Picasso CRLB sqrt warnings (negative CRLBs from unconverged fits)
warnings.filterwarnings('ignore', message='invalid value encountered in sqrt')

# Picasso imports
try:
    from picasso import localize
    from picasso import gaussmle
    from picasso import gausslq
    from picasso import io as picasso_io
    picasso_available = True
except ImportError:
    picasso_available = False
    print("[ERROR] Picasso not installed!")
    print("Install with: pip install picassosr")
    sys.exit(1)

# ==============================================================================
# LOAD CONFIGURATION FROM TOML
# ==============================================================================

def load_config(config_path="config.toml"):
    """Load configuration from TOML file"""
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found: {config_path}")
        print(f"[ERROR] Please create config.toml with [vmi_picasso] section")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "vmi_picasso" not in config:
        print(f"[ERROR] Missing [vmi_picasso] section in {config_path}")
        sys.exit(1)

    return config

# Load configuration
CONFIG = load_config()
vmi_cfg = CONFIG["vmi_picasso"]
proc_cfg = CONFIG["vmi_picasso"]["processing"]
hist_cfg = CONFIG["vmi_picasso"]["histogram"]
gmd_cfg = CONFIG["vmi_picasso"]["gmd"]

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- INPUT/OUTPUT (HDF5) ---
H5_DIR = vmi_cfg["h5_dir"]
H5_PATTERN = os.path.join(H5_DIR, vmi_cfg.get("h5_pattern", "*.h5"))
H5_DATASET = vmi_cfg.get("h5_dataset", "raw/data")
H5_META = vmi_cfg.get("h5_meta", "raw/meta")
META_COLUMN_GMD = vmi_cfg.get("meta_column_gmd", 1)  # DEPRECATED — see GMD_MAP_PATH

# Pre-computed frame-to-GMD map file
GMD_MAP_PATH = vmi_cfg.get("gmd_map_path", "")
GMD_MAP_PATH = os.path.abspath(GMD_MAP_PATH)

OUTPUT_DIR = vmi_cfg["output_dir"]
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "vmi_results.h5")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vmi_results.csv")
CHECKPOINT_FILE = vmi_cfg.get("checkpoint_file", "vmi_results_raw.h5")

# --- DARK/BACKGROUND DATA ---
DARK_DATA_DIR = vmi_cfg.get("dark_data_dir", "")
if DARK_DATA_DIR:
    DARK_H5_PATTERN = os.path.join(DARK_DATA_DIR, vmi_cfg.get("dark_h5_pattern", "*.h5"))
else:
    DARK_H5_PATTERN = ""

# --- CAMERA/DETECTOR ---
SATURATION_LEVEL = 65535  # 16-bit camera

# --- PICASSO LOCALIZATION PARAMETERS ---
BOX_SIZE = vmi_cfg["box_size"]
MIN_NET_GRADIENT = vmi_cfg["min_net_gradient"]

# Camera parameters for MLE fitting
CAMERA_BASELINE = vmi_cfg["camera_baseline"]
CAMERA_SENSITIVITY = vmi_cfg["camera_sensitivity"]
CAMERA_GAIN = vmi_cfg["camera_gain"]
CAMERA_QE = vmi_cfg["camera_qe"]

# MLE convergence parameters
MLE_CONVERGENCE = vmi_cfg["mle_convergence"]
MLE_MAX_ITERATIONS = vmi_cfg["mle_max_iterations"]
MLE_METHOD = vmi_cfg.get("mle_method", "sigmaxy")
FITTING_METHOD = vmi_cfg.get("fitting_method", "mle")

# Tuning plot histogram bins
TUNING_BINS_PHOTONS = vmi_cfg.get("tuning_bins_photons", 50)
TUNING_BINS_NET_GRADIENT = vmi_cfg.get("tuning_bins_net_gradient", 50)
TUNING_BINS_PRECISION = vmi_cfg.get("tuning_bins_precision", 50)
TUNING_FRAME_INDEX = vmi_cfg.get("tuning_frame_index", 10000)

# --- PARALLEL PROCESSING ---
ENABLE_PARALLEL = proc_cfg["enable_parallel"]
NUM_CORES = proc_cfg["num_cores"]

# --- DARK FRAME SUBTRACTION ---
DARK_FILE = vmi_cfg.get("dark_file", "")
if DARK_FILE and not os.path.isabs(DARK_FILE):
    DARK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DARK_FILE)
DARK_METHOD = proc_cfg.get("dark_method", "median")
DARK_FRAME_COUNT = proc_cfg["dark_frame_count"]
ENABLE_DARK_SUBTRACTION = proc_cfg["enable_dark_subtraction"]

# --- FLAT FIELD CORRECTION ---
ENABLE_FLAT_FIELD = proc_cfg["enable_flat_field"]
FLAT_FIELD_FILE = proc_cfg.get("flat_field_file", None)

# --- CIRCULAR MASK ---
ENABLE_CIRCULAR_MASK = proc_cfg["enable_circular_mask"]
MASK_CENTER_X = proc_cfg.get("mask_center_x", None)
MASK_CENTER_Y = proc_cfg.get("mask_center_y", None)
MASK_RADIUS = proc_cfg.get("mask_radius", None)

# --- HISTOGRAM PARAMETERS ---
HISTOGRAM_RESOLUTION_FACTOR = hist_cfg["resolution_factor"]

# --- PROCESSING MODE ---
TUNING_ONLY = proc_cfg["tuning_only"]

# --- GMD NORMALIZATION ---
ENABLE_GMD_WEIGHTING = gmd_cfg["enable_gmd_weighting"]
GMD_WEIGHT_MIN = gmd_cfg.get("gmd_weight_min", 0.0)
GMD_WEIGHT_MAX = gmd_cfg.get("gmd_weight_max", 0.0)

# Optional: Enable Abel transform
try:
    import abel
    abel_available = hist_cfg.get("enable_abel", False) and True
except ImportError:
    abel_available = False
    if hist_cfg.get("enable_abel", False):
        print("[Warning] PyAbel not installed. Abel transform disabled.")
        print("Install with: pip install pyabel")

# ==============================================================================
# SETUP
# ==============================================================================

def create_output_dir():
    """Create output directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_frames_and_metadata():
    """Scan HDF5 files and build frame index.

    Returns:
        h5_files: sorted list of HDF5 file paths
        frame_info: list of (h5_path, frame_in_file) per global frame
        img_shape: (height, width) from first file
        n_total_frames: total frames across all files
    """
    print(f"[Load] Searching: {H5_PATTERN}")
    h5_files = sorted(glob.glob(H5_PATTERN))

    if len(h5_files) == 0:
        print(f"[ERROR] No HDF5 files found matching pattern!")
        sys.exit(1)

    # Determine image shape and count total frames
    with h5py.File(h5_files[0], 'r') as hf:
        dset = hf[H5_DATASET]
        img_shape = dset.shape[1:]  # (height, width)
        print(f"[Load] Image size: {img_shape[1]}×{img_shape[0]} pixels")

    # Build flat frame index
    frame_info = []  # index = global_frame_idx -> (h5_path, frame_in_h5)
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as hf:
            n_frames = hf[H5_DATASET].shape[0]
        for i in range(n_frames):
            frame_info.append((h5_path, i))

    n_total_frames = len(frame_info)
    print(f"[Load] Found {len(h5_files)} HDF5 files, {n_total_frames} total frames")
    print(f"[Load] Frames per file: {n_total_frames // len(h5_files)}")

    return h5_files, frame_info, img_shape, n_total_frames

def calculate_dark_frame():
    """Calculate dark frame from HDF5 files in dark data directory.

    Priority:
      1. dark_file: load pre-computed .npy file (fastest)
      2. dark_data_dir: compute from H5 files
      3. Neither: return None (per-frame sigma-clipped stats)

    Uses method specified by DARK_METHOD config ("median" or "mean").
    """
    # Priority 1: pre-computed dark file
    if DARK_FILE and os.path.exists(DARK_FILE):
        print(f"[Dark] Loading pre-computed dark frame from: {DARK_FILE}")
        dark_frame = np.load(DARK_FILE)
        if dark_frame.shape != (1024, 1024):
            print(f"[Dark] WARNING: unexpected shape {dark_frame.shape}")
        print(f"[Dark] Loaded: mean={dark_frame.mean():.1f}, "
              f"std={dark_frame.std():.1f}")
        return dark_frame

    # Priority 2: compute from H5 files
    if not DARK_DATA_DIR or not DARK_H5_PATTERN:
        print("[Dark] No dark data directory configured, skipping dark frame")
        return None

    print(f"[Dark] Loading background HDF5 files from: {DARK_H5_PATTERN}")

    dark_h5_files = sorted(glob.glob(DARK_H5_PATTERN))

    if len(dark_h5_files) == 0:
        print(f"[Dark] WARNING: No background HDF5 files found! Using sigma-clipped stats instead.")
        return None

    print(f"[Dark] Found {len(dark_h5_files)} dark HDF5 files, method={DARK_METHOD}")

    if DARK_METHOD == "mean":
        dark_frame, n_used = _calc_dark_mean(dark_h5_files)
    else:
        dark_frame, n_used = _calc_dark_median(dark_h5_files)

    if dark_frame is None:
        print(f"[Dark] WARNING: No dark frames could be processed! "
              f"Using sigma-clipped stats instead.")
        return None

    print(f"[Dark] Used {n_used} frames, method={DARK_METHOD}")
    print(f"[Dark] Dark level: mean={dark_frame.mean():.1f}, "
          f"std={dark_frame.std():.1f}")

    return dark_frame

def _read_h5_data(fpath):
    """Worker: return raw data from one H5 file."""
    with h5py.File(fpath, 'r') as hf:
        return hf[H5_DATASET][:]  # (N, H, W) uint16

def _sum_h5_file(fpath):
    """Worker: return (sum_of_frames, n_frames) for one H5 file."""
    with h5py.File(fpath, 'r') as hf:
        data = hf[H5_DATASET][:]  # (N, H, W) uint16
    return data.sum(axis=0, dtype=np.float64), data.shape[0]

def _select_dark_files(dark_h5_files):
    """Select enough files to cover DARK_FRAME_COUNT frames. Returns list of paths."""
    with h5py.File(dark_h5_files[0], 'r') as hf:
        frames_per_file = hf[H5_DATASET].shape[0]
    n_needed = int(np.ceil(DARK_FRAME_COUNT / frames_per_file))
    return dark_h5_files[:min(n_needed, len(dark_h5_files))]

def _calc_dark_median(dark_h5_files):
    """Calculate dark frame as median of subsampled frames (parallel read)."""
    selected = _select_dark_files(dark_h5_files)
    n_workers = min(len(selected), cpu_count())
    print(f"[Dark] Median method: reading {len(selected)} H5 files ({n_workers} workers)...")

    all_frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_read_h5_data, f): f for f in selected}
        for fut in as_completed(futures):
            fname = os.path.basename(futures[fut])
            try:
                all_frames.append(fut.result())
            except Exception as e:
                print(f"[Dark] WARNING: Failed to read {fname}: {e}")

    if not all_frames:
        return None, 0

    stack = np.concatenate(all_frames, axis=0)[:DARK_FRAME_COUNT]
    dark_frame = np.median(stack, axis=0).astype(np.float64)
    return dark_frame, stack.shape[0]

def _calc_dark_mean(dark_h5_files):
    """Calculate dark frame as mean of frames from selected files (parallel).

    Uses enough files to cover DARK_FRAME_COUNT frames. All frames from
    those selected files are used (no cropping).
    """
    selected = _select_dark_files(dark_h5_files)
    n_workers = min(len(selected), cpu_count())
    print(f"[Dark] Mean method: processing {len(selected)} H5 files in parallel "
          f"({n_workers} workers)...")

    running_sum = None
    n_total = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_sum_h5_file, f): f for f in selected}
        for fut in as_completed(futures):
            fname = os.path.basename(futures[fut])
            try:
                file_sum, n_frames = fut.result()
                if running_sum is None:
                    running_sum = file_sum
                else:
                    running_sum += file_sum
                n_total += n_frames
            except Exception as e:
                print(f"[Dark] WARNING: Failed to read {fname}: {e}")

    if running_sum is None or n_total == 0:
        return None, 0

    dark_frame = running_sum / n_total
    return dark_frame, n_total

def load_flat_field():
    """Load flat field correction (still from TIFF -- single image)."""
    if not ENABLE_FLAT_FIELD or FLAT_FIELD_FILE is None:
        return None
    try:
        import tifffile
    except ImportError:
        print("[Flat] WARNING: tifffile not installed, skipping flat field")
        return None
    print(f"[Flat] Loading: {FLAT_FIELD_FILE}")
    flat_field = tifffile.imread(FLAT_FIELD_FILE)
    flat_field_normalized = flat_field / flat_field.mean()
    print(f"[Flat] Flat field correction loaded")
    return flat_field_normalized

# ==============================================================================
# GMD EXTRACTION (from pre-computed timestamp-based map)
# ==============================================================================

def load_gmd_map(map_path):
    """Load the pre-computed frame-to-GMD correspondence map.

    Returns:
        gmd_for_frame: numpy array of length n_total_frames with GMD values (method 1)
        or None if the map file cannot be loaded.
    """
    if not os.path.exists(map_path):
        print(f"[GMD] WARNING: GMD map file not found: {map_path}")
        print(f"[GMD] Falling back: all events will have weight = 1.0")
        return None

    print(f"[GMD] Loading frame-to-GMD map from: {map_path}")
    try:
        # Columns: frame_idx, h5_file(str!), frame_in_file, gmd_idx, gmd_val1, gmd_val2, time_diff_ms
        # Skip column 1 (string filename) — load only numeric columns
        map_data = np.loadtxt(map_path, comments='#', usecols=(0, 2, 3, 4, 5, 6))
        if map_data.ndim == 1:
            map_data = map_data[np.newaxis, :]

        frame_indices = map_data[:, 0].astype(int)
        gmd_values = map_data[:, 3]  # column 4 in original = index 3 after dropping string col

        # Clamp near-zero GMD values (measurement noise can produce tiny negatives)
        n_neg = (gmd_values < 0).sum()
        if n_neg:
            print(f"[GMD] Clamping {n_neg} negative GMD values to 1e-12")
            gmd_values = np.maximum(gmd_values, 1e-12)

        n_total = int(frame_indices.max()) + 1
        gmd_arr = np.full(n_total, np.nan)
        gmd_arr[frame_indices] = gmd_values

        n_valid = np.isfinite(gmd_arr).sum()
        print(f"[GMD] Map loaded: {n_valid}/{n_total} frames have GMD data")
        print(f"[GMD] GMD range: [{gmd_values.min():.3e}, {gmd_values.max():.3e}]")
        return gmd_arr

    except Exception as e:
        print(f"[GMD] ERROR loading map file: {e}")
        return None

# ==============================================================================
# PICASSO LOCALIZATION -- BATCH VERSION
# ==============================================================================
# Processes all N frames from one H5 file through Picasso in a single call:
#   movie (N, H, W) -> identify() -> get_spots() -> gaussmle() -> locs_from_fits()
#
# The old per-frame loop called this chain 126× per file; here it is 1×.

def _camera_info():
    return {
        'Baseline': CAMERA_BASELINE,
        'Sensitivity': CAMERA_SENSITIVITY,
        'Gain': CAMERA_GAIN,
        'Qe': CAMERA_QE,
    }

def _run_picasso_batch(movie, box_size, min_net_gradient):
    """Run identify + fit on a multi-frame movie in one batch call.

    Fitting method (controlled by config fitting_method):
      lq:  Least Squares (gausslq) — fast, no divergence
      mle: Maximum Likelihood (gaussmle) — slower, Poisson-optimal

    Args:
        movie: (N, H, W) uint16 — corrected frames

    Returns:
        locs DataFrame, or None if no spots found.
    """
    # 1. Identify spots across ALL frames simultaneously
    try:
        ids = localize.identify(movie, min_net_gradient, box_size, threaded=False)
    except Exception:
        return None
    if len(ids) == 0:
        return None

    # 2. Extract spot patches + convert to photons
    try:
        spots = localize.get_spots(movie, ids, box_size, _camera_info())
    except Exception:
        return None

    # 3. Fitting (LQ or MLE)
    if FITTING_METHOD == "lq":
        try:
            theta = gausslq.fit_spots(spots)
            locs = gausslq.locs_from_fits(ids, theta, box_size, em=False)
        except Exception:
            return None
    else:
        try:
            theta, CRLBs, likelihoods, iterations = gaussmle.gaussmle(
                spots, MLE_CONVERGENCE, MLE_MAX_ITERATIONS, method=MLE_METHOD
            )
            locs = gaussmle.locs_from_fits(
                ids, theta, CRLBs, likelihoods, iterations, box_size
            )
        except Exception:
            return None

    # 4. Filter degenerate fits (sx or sy near zero -- failed convergence)
    #    (LQ is less prone to this, but guard anyway)
    min_sigma = 0.3
    valid = (locs['sx'].values > min_sigma) & (locs['sy'].values > min_sigma)
    if not valid.any():
        return None
    if (~valid).any():
        locs = locs[valid].reset_index(drop=True)

    return locs

def _batch_to_trackpy(locs, base_frame_idx):
    """Convert batch Picasso output to trackpy-style DataFrame.

    locs_from_fits already has a 'frame' column (0-indexed within movie).
    We adjust it to global frame numbers and add derived columns.

    Args:
        locs: DataFrame from _run_picasso_batch
        base_frame_idx: global index of the first frame in this batch

    Returns:
        DataFrame with columns: x, y, mass, size, ecc, signal, frame, net_gradient
    """
    if locs is None or len(locs) == 0:
        return None

    df = pd.DataFrame()
    df['x'] = locs['x'].values
    df['y'] = locs['y'].values
    df['mass'] = locs['photons'].values
    df['size'] = (locs['sx'].values + locs['sy'].values) / 2.0

    # Eccentricity: guard against zero width
    sx = locs['sx'].values
    sy = locs['sy'].values
    minor = np.minimum(sx, sy)
    major = np.maximum(sx, sy)
    major = np.maximum(major, 0.3)  # prevent 0/0
    df['ecc'] = 1.0 - (minor / major)

    # Signal: guard against zero size
    size_safe = np.maximum(df['size'].values, 0.3)
    df['signal'] = locs['photons'].values / (np.pi * size_safe**2)

    # Global frame number (Picasso's frame is 0-indexed within the movie)
    df['frame'] = locs['frame'].values + base_frame_idx
    df['net_gradient'] = locs['net_gradient'].values

    return df

def process_h5_file_batch(args):
    """Worker: process ALL frames from one H5 file in a single batch.

    Instead of looping 126× through (identify + get_spots + gaussmle),
    we apply corrections to all frames at once then pass the full
    (N, H, W) movie through the Picasso pipeline once.

    Args:
        args: (h5_path, base_frame_idx, dark_frame, flat_field_normalized,
               box_size, min_net_gradient)

    Returns:
        DataFrame with all detections from this file, or None.
    """
    h5_path, base_frame_idx, dark_frame, flat_field_normalized, box_size, min_net_gradient = args

    # Read all frames from this HDF5 file
    try:
        with h5py.File(h5_path, 'r') as hf:
            data = hf[H5_DATASET][:]  # (N, H, W) uint16
    except Exception as e:
        print(f"[Warning] Failed to read {h5_path}: {e}")
        return None

    n_frames = data.shape[0]

    # --- Apply corrections to ALL frames at once (vectorized where possible) ---
    if dark_frame is not None:
        # Single vectorized subtraction: (N, H, W) - (H, W) broadcasts
        corrected = data.astype(np.float64) - dark_frame
    else:
        # Per-frame sigma-clipped stats (still needs loop, but fast numpy)
        corrected = data.astype(np.float64)
        for i in range(n_frames):
            _, median_bkg, _ = sigma_clipped_stats(corrected[i], sigma=3.0)
            corrected[i] -= median_bkg

    if flat_field_normalized is not None:
        corrected = corrected / flat_field_normalized

    # Pass directly as float64 — Picasso converts to float32 internally.
    # No clamping: preserving negative values gives correct zero-centered
    # background statistics for the MLE/LQ fit.
    # No uint16 round-trip: removes unnecessary precision loss.
    locs = _run_picasso_batch(corrected, box_size, min_net_gradient)
    if locs is None or len(locs) == 0:
        return None

    # --- Convert to trackpy format with global frame numbers ---
    features = _batch_to_trackpy(locs, base_frame_idx)
    return features

# ==============================================================================
# BATCH PROCESSING (HDF5 files)
# ==============================================================================

def run_batch_processing_picasso(h5_files, frame_info, dark_frame, flat_field_normalized):
    """Run Picasso localization on all frames, one HDF5 file per task (batch mode).

    Streams per-file results directly to an HDF5 checkpoint to avoid holding
    all per-file DataFrames in memory simultaneously. Peak memory is bounded
    to ~1 per-file result + the final concatenated copy, rather than
    (#files × per-file result) + concatenated copy.

    Args:
        h5_files: sorted list of HDF5 file paths
        frame_info: list of (h5_path, frame_in_file) for each global frame
        dark_frame: dark frame array or None
        flat_field_normalized: flat field array or None

    Returns:
        Nothing. Results are streamed to vmi_results_raw.h5 on disk.
    """
    print("=" * 80)
    print("BATCH PROCESSING (V5)")
    print("=" * 80)

    n_cores = NUM_CORES if NUM_CORES is not None else cpu_count()
    print(f"[Batch] Parallel processing: {n_cores} cores")

    # Build batch arguments -- one per HDF5 file
    h5_batch_args = []
    idx = 0
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as hf:
            n = hf[H5_DATASET].shape[0]
        h5_batch_args.append((h5_path, idx, dark_frame, flat_field_normalized,
                              BOX_SIZE, MIN_NET_GRADIENT))
        idx += n

    n_files = len(h5_batch_args)
    print(f"[Batch] Processing {n_files} HDF5 files ({len(frame_info)} frames) "
          f"with Picasso batch mode (MLE + Net Gradient)...")

    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Stream results directly to HDF5 — never accumulate a list of DataFrames.
    # Each per-file result is freed as soon as it's written to disk.
    n_total = 0
    files_with_detections = 0

    with pd.HDFStore(checkpoint_path, mode='w', complevel=1) as store:
        if ENABLE_PARALLEL:
            with Pool(n_cores) as pool:
                for result in tqdm(
                    pool.imap(process_h5_file_batch, h5_batch_args),
                    total=n_files,
                    desc="Processing HDF5 files",
                ):
                    if result is not None:
                        store.append("localizations", result, index=False)
                        n_total += len(result)
                        files_with_detections += 1
        else:
            for args in tqdm(h5_batch_args):
                result = process_h5_file_batch(args)
                if result is not None:
                    store.append("localizations", result, index=False)
                    n_total += len(result)
                    files_with_detections += 1

    if n_total == 0:
        print("[ERROR] No events detected!")
        sys.exit(1)

    total_frames = len(frame_info)
    print(f"[Batch] Total detections: {n_total} events")
    print(f"[Batch] Events/frame: {n_total / total_frames:.1f} avg")
    print(f"[Batch] Files with detections: {files_with_detections}/{n_files}")

    # Data is in the checkpoint on disk — caller reads it back in chunks
    # via build_vmi_histogram_from_checkpoint.

# ==============================================================================
# VMI HISTOGRAM (chunked from HDF5 checkpoint — memory-bounded)
# ==============================================================================

def build_vmi_histogram_from_checkpoint(
    checkpoint_path, img_shape,
    gmd_array=None, gmd_mean=None,
    gmd_weight_min=0.0, gmd_weight_max=0.0,
    chunksize=500_000,
):
    """Build 2D VMI histogram by reading events from HDF5 checkpoint in chunks.

    Eliminates the need to hold all detected events in RAM simultaneously.
    For each chunk:
      1. Read events from the checkpoint
      2. Apply GMD weighting (vectorised per-chunk)
      3. Accumulate into the 2D histogram
      4. Discard the chunk

    Peak memory ≈ chunksize × 10 columns × 8 bytes ≈ 40 MB at default chunksize,
    plus the accumulated histogram itself (~8 MB for 2048×2048 float64).

    Args:
        checkpoint_path: Path to vmi_results_raw.h5 written by run_batch_processing_picasso.
        img_shape: (height, width) of the raw camera frames.
        gmd_array: Preprocessed GMD array (NaNs filled with mean), or None.
                   When None all weights default to 1.0.
        gmd_mean: Scalar mean GMD value for normalisation (required if gmd_array
                  is not None).
        gmd_weight_min: Weights below this are zeroed (0 = no clamp).
        gmd_weight_max: Weights above this are zeroed (0 = no clamp).
        chunksize: Number of event rows to read per chunk.

    Returns:
        vmi_2d: 2D histogram array of shape (hist_ny, hist_nx).
        n_events: Total number of events processed.
        max_frame: Maximum global frame index that contained any detection.
    """
    print("=" * 80)
    print("VMI HISTOGRAM (chunked from checkpoint)")
    print("=" * 80)

    ny, nx = img_shape
    hist_nx = nx * HISTOGRAM_RESOLUTION_FACTOR
    hist_ny = ny * HISTOGRAM_RESOLUTION_FACTOR
    print(f"[Histogram] Resolution: {hist_nx}×{hist_ny}, "
          f"chunksize: {chunksize:,} events")

    vmi_2d = np.zeros((hist_ny, hist_nx), dtype=np.float64)
    n_events = 0
    max_frame = 0
    n_zeroed = 0
    use_gmd = gmd_array is not None and gmd_mean is not None

    # Pre-count total rows for an accurate progress bar.
    with h5py.File(checkpoint_path, "r") as hf:
        total_rows = hf["localizations/table"].shape[0]

    with tqdm(total=total_rows, desc="Building histogram",
              unit=" events", mininterval=1) as pbar:
        for chunk in pd.read_hdf(checkpoint_path, "localizations",
                                 chunksize=chunksize):
            x = chunk["x"].values * HISTOGRAM_RESOLUTION_FACTOR
            y = chunk["y"].values * HISTOGRAM_RESOLUTION_FACTOR

            if use_gmd:
                frame_idx = chunk["frame"].values.astype(np.int64)
                gmd_energy = gmd_array[frame_idx]
                weight = gmd_mean / gmd_energy
                if gmd_weight_max > 0:
                    weight[weight > gmd_weight_max] = 0.0
                if gmd_weight_min > 0:
                    weight[weight < gmd_weight_min] = 0.0
                if gmd_weight_max > 0 or gmd_weight_min > 0:
                    n_zeroed += int((weight == 0).sum())
                weights = weight
                # Log confirmation on the first chunk only
                if n_events == 0:
                    print(f"[Histogram] GMD weighting active: mean={gmd_mean:.3e}, "
                          f"weight range [{weight.min():.4f}, {weight.max():.4f}]")
            else:
                weights = None

            chunk_hist, _, _ = np.histogram2d(
                y, x,
                bins=[hist_ny, hist_nx],
                range=[[0, hist_ny], [0, hist_nx]],
                weights=weights,
            )
            vmi_2d += chunk_hist
            n_events += len(chunk)
            max_frame = max(max_frame, chunk["frame"].max())
            pbar.update(len(chunk))

    print(f"[Histogram] Total events: {n_events:,}")
    print(f"[Histogram] Total weighted counts: {vmi_2d.sum():.0f}")
    print(f"[Histogram] Max frame with detections: {max_frame}")

    if use_gmd and (gmd_weight_min > 0 or gmd_weight_max > 0):
        print(f"[Histogram] Weight filter [{gmd_weight_min}, {gmd_weight_max}]: "
              f"{n_zeroed:,}/{n_events:,} events zeroed")

    return vmi_2d, n_events, max_frame

# ==============================================================================
# ABEL TRANSFORM
# ==============================================================================

def perform_abel_transform(vmi_2d):
    """Perform inverse Abel transform to get 3D distribution"""

    if not abel_available:
        print("[Abel] Skipped (PyAbel not installed)")
        return None

    print("=" * 80)
    print("ABEL TRANSFORM")
    print("=" * 80)

    try:
        vmi_3d = abel.transform.basex_transform(
            vmi_2d,
            direction='inverse',
            basis_dir=None,
            verbose=False
        )
        print(f"[Abel] Method: BASEX inverse transform")
        print(f"[Abel] Reconstructed 3D distribution")
        return vmi_3d
    except Exception as e:
        print(f"[Abel] Error: {e}")
        return None

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(vmi_2d, vmi_3d):
    """Save 2D histogram and optional Abel transform to disk.
    Raw detections are already saved as checkpoint (vmi_results_raw.h5)."""
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    vmi_2d_path = os.path.join(OUTPUT_DIR, "vmi_2d_raw.npy")
    np.save(vmi_2d_path, vmi_2d)
    print(f"[Save] 2D histogram: {vmi_2d_path}")

    if vmi_3d is not None:
        vmi_3d_path = os.path.join(OUTPUT_DIR, "vmi_3d_abel.npy")
        np.save(vmi_3d_path, vmi_3d)
        print(f"[Save] Abel inverted: {vmi_3d_path}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_final_summary(vmi_2d, vmi_3d, n_events, n_frames):
    """Create final summary plot (scalar stats, no DataFrame needed)."""

    print("=" * 80)
    print("FINAL VISUALIZATION")
    print("=" * 80)

    fig = plt.figure(figsize=(15, 5), facecolor='white')
    fig.patch.set_facecolor('white')

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(vmi_2d, origin='lower', cmap='hot', aspect='equal', interpolation='nearest')
    ax1.set_title(f"2D VMI Image (Raw)\n{n_events} events", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(False)
    plt.colorbar(im1, ax=ax1, label='Counts')

    ax2 = plt.subplot(1, 3, 2)
    if vmi_3d is not None:
        im2 = ax2.imshow(vmi_3d, origin='lower', cmap='hot')
        ax2.set_title("3D VMI (Abel Inverted)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        plt.colorbar(im2, ax=ax2, label='Counts')
    else:
        ax2.text(0.5, 0.5, "Abel transform\nnot available",
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("3D VMI (Abel Inverted)", fontsize=12, fontweight='bold')

    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')

    ev_per_frame = n_events / max(n_frames, 1)
    summary_text = f"""PICASSO ANALYSIS COMPLETE (V5 CHUNKED)

Input:
    Frames: {n_frames}
    Resolution: {vmi_2d.shape[1]}×{vmi_2d.shape[0]}

Detection:
    Total events: {n_events:,}
    Events/frame: {ev_per_frame:.1f}

Parameters:
    Box size: {BOX_SIZE} pixels
    Min Net Gradient: {MIN_NET_GRADIENT}

Corrections:
    Dark frame: {'✓' if ENABLE_DARK_SUBTRACTION else '✗'}
    Flat field: {'✓' if ENABLE_FLAT_FIELD else '✗'}
    Circular mask: {'✓' if ENABLE_CIRCULAR_MASK else '✗'}

Output:
    VMI histogram: {vmi_2d.shape[1]}×{vmi_2d.shape[0]}
    Abel transform: {'✓' if vmi_3d is not None else '✗'}
"""

    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "04_final_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved: {output_path}")
    plt.show()
    plt.close()

# ==============================================================================
# TUNING MODE (single frame -- unchanged from V3)
# ==============================================================================

def run_tuning_check(frame_info, dark_frame, flat_field_normalized):
    """Run detection on one frame (from HDF5) to tune parameters"""

    print("=" * 80)
    print("TUNING CHECK")
    print("=" * 80)

    test_global_idx = min(TUNING_FRAME_INDEX, len(frame_info) - 1)
    h5_path, frame_in_h5 = frame_info[test_global_idx]

    print(f"[Tuning] Global frame index: {test_global_idx}")
    print(f"[Tuning] HDF5 file: {os.path.basename(h5_path)}")
    print(f"[Tuning] Frame within file: {frame_in_h5}")

    with h5py.File(h5_path, 'r') as hf:
        test_frame = hf[H5_DATASET][frame_in_h5]

    # Apply corrections
    if dark_frame is not None:
        corrected = test_frame.astype(np.float64) - dark_frame
    else:
        corrected = test_frame.astype(np.float64)
        mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(corrected, sigma=3.0)
        corrected = corrected - median_bkg
        print(f"[Tuning] Per-frame background: mean={mean_bkg:.1f}, "
              f"median={median_bkg:.1f}, std={std_bkg:.1f}")

    if flat_field_normalized is not None:
        corrected = corrected / flat_field_normalized

    # Single-frame Picasso call (float64 — no clamp, no uint16 round-trip)
    locs = _run_picasso_batch(
        corrected[np.newaxis, :, :], BOX_SIZE, MIN_NET_GRADIENT
    )

    print(f"[Tuning] Box size: {BOX_SIZE} pixels")
    print(f"[Tuning] Min Net Gradient: {MIN_NET_GRADIENT}")
    print(f"[Tuning] Detected {len(locs) if locs is not None else 0} events")

    if locs is not None and len(locs) > 0:
        print(f"[Tuning] Photon range: {locs['photons'].min():.0f} - {locs['photons'].max():.0f}")
        print(f"[Tuning] Net Gradient range: {locs['net_gradient'].min():.0f} - {locs['net_gradient'].max():.0f}")
        if 'iterations' in locs.columns:
            it = locs['iterations']
            print(f"[Tuning] Iterations: median={it.median():.0f}  max={it.max():.0f}  "
                  f"lt10={((it <= 10).mean()*100):.0f}%  lt50={((it <= 50).mean()*100):.0f}%")

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    ax1 = axes[0]
    ax1.imshow(corrected, cmap='gray', origin='lower', interpolation='none',
               vmin=0, vmax=np.percentile(corrected, 99.9))
    ax1.grid(False)
    if locs is not None and len(locs) > 0:
        ax1.plot(locs['x'], locs['y'], 'ro', markersize=10, fillstyle='none',
                markeredgewidth=0.8)
    ax1.set_title(f"Tuning Check: {len(locs) if locs is not None else 0} events\nPicasso MLE + Net Gradient",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")

    ax2 = axes[1]
    if locs is not None and len(locs) > 0:
        ax2.hist(locs['photons'], bins=TUNING_BINS_PHOTONS, alpha=0.7, edgecolor='black')
        ax2.axvline(np.median(locs['photons']), color='red', linestyle='--',
                   label=f"Median = {np.median(locs['photons']):.0f}")
    ax2.set_title("Photon Distribution")
    ax2.set_xlabel("Photons (integrated intensity)")
    ax2.set_ylabel("Count")
    ax2.legend()

    ax3 = axes[2]
    if locs is not None and len(locs) > 0:
        ax3.hist(locs['net_gradient'], bins=TUNING_BINS_NET_GRADIENT, alpha=0.7, edgecolor='black', color='green')
        ax3.axvline(MIN_NET_GRADIENT, color='red', linestyle='--',
                   label=f"Threshold = {MIN_NET_GRADIENT}")
    ax3.set_title("Net Gradient Distribution\n(Pile-up Detection)")
    ax3.set_xlabel("Net Gradient")
    ax3.set_ylabel("Count")
    ax3.legend()

    ax4 = axes[3]
    if locs is not None and len(locs) > 0:
        precision = (locs['lpx'] + locs['lpy']) / 2.0
        precision_valid = precision[np.isfinite(precision)]
        if len(precision_valid) > 0:
            ax4.hist(precision_valid, bins=TUNING_BINS_PRECISION, alpha=0.7, edgecolor='black', color='orange')
            ax4.axvline(np.median(precision_valid), color='red', linestyle='--',
                       label=f"Median = {np.median(precision_valid):.3f} px")
            ax4.legend()
    ax4.set_title("Localization Precision\n(MLE uncertainty)")
    ax4.set_xlabel("Precision (pixels)")
    ax4.set_ylabel("Count")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "01_tuning_check_picasso.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Tuning] Saved: {output_path}")
    plt.show()
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("VMI PHOTOELECTRON ANALYSIS - PICASSO V4 BATCH (HDF5 Input)")
    print("=" * 80)
    print(f"[Config] Loaded from: config.toml")
    print(f"[Config] HDF5 dir: {H5_DIR}")
    print(f"[Config] HDF5 dataset: {H5_DATASET}")
    print(f"[Config] GMD map: {GMD_MAP_PATH}")
    print(f"[Config] Dark dir: {DARK_DATA_DIR}")
    print(f"[Config] Output dir: {OUTPUT_DIR}")
    print(f"[Config] BOX_SIZE={BOX_SIZE}, MIN_NET_GRADIENT={MIN_NET_GRADIENT}")
    print(f"[Config] Dark subtraction: {'ON' if ENABLE_DARK_SUBTRACTION else 'OFF'}" +
          (f" ({DARK_METHOD})" if ENABLE_DARK_SUBTRACTION else ""))
    print(f"[Config] GMD weighting: {'ON' if ENABLE_GMD_WEIGHTING else 'OFF'}")
    print(f"[Config] Parallel: {'ON' if ENABLE_PARALLEL else 'OFF'} ({NUM_CORES} cores)")
    print(f"[Config] Tuning mode: {'ON' if TUNING_ONLY else 'OFF'}")
    print(f"[Config] Batch mode: ON  Fitting method: {FITTING_METHOD}")
    print("=" * 80)

    # Setup
    create_output_dir()
    h5_files, frame_info, img_shape, n_total_frames = load_frames_and_metadata()

    # Load corrections
    if ENABLE_DARK_SUBTRACTION:
        dark_frame = calculate_dark_frame()
    else:
        dark_frame = None
        print(f"\n[Dark] Dark subtraction DISABLED (ENABLE_DARK_SUBTRACTION = False)")
        print(f"[Dark] Background will be estimated per-frame using sigma-clipped stats")

    flat_field_normalized = load_flat_field()

    # Tuning mode
    if TUNING_ONLY:
        run_tuning_check(frame_info, dark_frame, flat_field_normalized)
        print("\n" + "=" * 80)
        print("TUNING MODE - Batch processing skipped")
        print("=" * 80)
        print("\nAdjust parameters and re-run until satisfied, then set TUNING_ONLY = False")
        return

    # ========================================================================
    # BATCH PROCESSING or RESUME FROM CHECKPOINT
    # ========================================================================
    # run_batch_processing_picasso streams per-file results directly to
    # CHECKPOINT_FILE (never accumulates a list of DataFrames in RAM).
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        print(f"\n[Resume] Found checkpoint: {checkpoint_path}")
    else:
        run_batch_processing_picasso(h5_files, frame_info, dark_frame,
                                     flat_field_normalized)
    # The checkpoint exists on disk — downstream steps read it in chunks.

    # ========================================================================
    # GMD PREPARATION  (preprocess the small GMD array — ~2 MB — only once)
    # ========================================================================
    gmd_array = None   # passed to chunked histogram builder
    gmd_mean = None

    if ENABLE_GMD_WEIGHTING:
        print(f"\n{'=' * 80}")
        print("GMD NORMALIZATION")
        print(f"{'=' * 80}")
        print("[GMD] Loading frame-to-GMD map...")
        gmd_raw = load_gmd_map(GMD_MAP_PATH)

        if gmd_raw is not None:
            valid_mask = np.isfinite(gmd_raw)
            n_valid = valid_mask.sum()
            n_total_gmd = len(gmd_raw)
            print(f"[GMD] Valid GMD values: {n_valid}/{n_total_gmd} frames")
            gmd_mean = gmd_raw[valid_mask].mean()
            gmd_array = np.where(np.isnan(gmd_raw), gmd_mean, gmd_raw)
            print(f"[GMD] Mean: {gmd_mean:.3e}")
        else:
            print(f"[GMD] WARNING: Cannot load GMD map — all weights = 1.0")
    else:
        print(f"\n{'=' * 80}")
        print("GMD NORMALIZATION")
        print(f"{'=' * 80}")
        print("[GMD] Weighting DISABLED — all weights = 1.0")

    # ========================================================================
    # VMI HISTOGRAM  (built from HDF5 checkpoint in chunks — memory bounded)
    # ========================================================================
    vmi_2d, n_events, max_frame = build_vmi_histogram_from_checkpoint(
        checkpoint_path, img_shape,
        gmd_array=gmd_array,
        gmd_mean=gmd_mean,
        gmd_weight_min=GMD_WEIGHT_MIN,
        gmd_weight_max=GMD_WEIGHT_MAX,
    )

    # Abel transform
    vmi_3d = perform_abel_transform(vmi_2d)

    # Save results
    save_results(vmi_2d, vmi_3d)

    # Visualization (scalar stats — no full DataFrame needed)
    plot_final_summary(vmi_2d, vmi_3d, n_events, n_total_frames)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
