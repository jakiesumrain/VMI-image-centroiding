#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMI Photoelectron Analysis - PICASSO V4 (Batch HDF5)
=====================================================
Batch-optimized version of V3: processes all 126 frames from each H5 file
through Picasso in a single call instead of 126 separate calls.

Key difference from V3:
- Single movie (N_frames, H, W) → Picasso identify/get_spots/gaussmle
  instead of per-frame (1, H, W) → Picasso
- Eliminates 125× redundant Python→numba boundary overhead per file
- ~3-5× speedup for large datasets

Data model:
  HDF5 file structure:
    raw/data  -> (N_frames, height, width)  uint16
    raw/meta  -> (N_frames, N_columns)       uint64

Author: TAO Jianfei (jakiesumrain@163.com)
Date: 2026-07-03 (v4: batch optimization)
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
    """Run identify + MLE fit on a multi-frame movie in one batch call.

    Args:
        movie: (N, H, W) uint16 — corrected frames

    Returns:
        locs DataFrame from gaussmle.locs_from_fits, or None if no spots found.
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

    # 3. MLE fitting on all spots from all frames at once
    try:
        theta, CRLBs, likelihoods, iterations = gaussmle.gaussmle(
            spots, MLE_CONVERGENCE, MLE_MAX_ITERATIONS, method=MLE_METHOD
        )
    except Exception:
        return None

    # 4. Convert to localizations
    try:
        locs = gaussmle.locs_from_fits(
            ids, theta, CRLBs, likelihoods, iterations, box_size
        )
    except Exception:
        return None

    # 5. Filter degenerate fits (sx or sy near zero -- failed convergence)
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
    # No clamping: preserving negatives for correct background statistics.
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

    Args:
        h5_files: sorted list of HDF5 file paths
        frame_info: list of (h5_path, frame_in_file) for each global frame
        dark_frame: dark frame array or None
        flat_field_normalized: flat field array or None

    Returns:
        df_all: concatenated DataFrame of all detections
    """
    print("=" * 80)
    print("BATCH PROCESSING (V4)")
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

    # Run workers
    if ENABLE_PARALLEL:
        with Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_h5_file_batch, h5_batch_args),
                total=n_files,
                desc="Processing HDF5 files"
            ))
    else:
        results = [process_h5_file_batch(args) for args in tqdm(h5_batch_args)]

    # Collect non-None results
    all_features = [r for r in results if r is not None]

    if len(all_features) == 0:
        print("[ERROR] No events detected!")
        sys.exit(1)

    df_all = pd.concat(all_features, ignore_index=True)

    total_frames = len(frame_info)
    print(f"[Batch] Total detections: {len(df_all)} events")
    print(f"[Batch] Events/frame: {len(df_all) / total_frames:.1f} avg")
    print(f"[Batch] Files with detections: {len(all_features)}/{n_files}")

    return df_all

# ==============================================================================
# VMI HISTOGRAM
# ==============================================================================

def create_vmi_histogram(df_all, img_shape):
    """Create 2D VMI histogram from localized events (with GMD weighting)"""

    print("=" * 80)
    print("VMI HISTOGRAM")
    print("=" * 80)

    ny, nx = img_shape
    hist_nx = nx * HISTOGRAM_RESOLUTION_FACTOR
    hist_ny = ny * HISTOGRAM_RESOLUTION_FACTOR

    print(f"[Histogram] Resolution: {hist_nx}×{hist_ny}")

    x_scaled = df_all['x'] * HISTOGRAM_RESOLUTION_FACTOR
    y_scaled = df_all['y'] * HISTOGRAM_RESOLUTION_FACTOR

    if 'weight' in df_all.columns:
        weights = df_all['weight'].values
    else:
        weights = None

    vmi_2d, _, _ = np.histogram2d(
        y_scaled, x_scaled,
        bins=[hist_ny, hist_nx],
        range=[[0, hist_ny], [0, hist_nx]],
        weights=weights
    )

    print(f"[Histogram] Total counts: {vmi_2d.sum():.0f}")

    return vmi_2d

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
    """Save 2D histogram and optional Abel transform.
    Raw detections already saved as checkpoint (vmi_results_raw.h5)."""
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

def plot_final_summary(df_all, vmi_2d, vmi_3d):
    """Create final summary plot"""

    print("=" * 80)
    print("FINAL VISUALIZATION")
    print("=" * 80)

    fig = plt.figure(figsize=(15, 5), facecolor='white')
    fig.patch.set_facecolor('white')

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(vmi_2d, origin='lower', cmap='hot', aspect='equal', interpolation='nearest')
    ax1.set_title(f"2D VMI Image (Raw)\n{len(df_all)} events", fontsize=12, fontweight='bold')
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

    summary_text = f"""PICASSO ANALYSIS COMPLETE (V4 BATCH)

Input:
    Frames: {df_all['frame'].max() + 1}
    Resolution: {vmi_2d.shape[1]}×{vmi_2d.shape[0]}

Detection:
    Total events: {len(df_all)}
    Events/frame: {len(df_all)/(df_all['frame'].max() + 1):.1f}

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

    # Single-frame Picasso call (float64 — no clamp)
    locs = _run_picasso_batch(
        corrected[np.newaxis, :, :], BOX_SIZE, MIN_NET_GRADIENT
    )

    print(f"[Tuning] Box size: {BOX_SIZE} pixels")
    print(f"[Tuning] Min Net Gradient: {MIN_NET_GRADIENT}")
    print(f"[Tuning] Detected {len(locs) if locs is not None else 0} events")

    if locs is not None and len(locs) > 0:
        print(f"[Tuning] Photon range: {locs['photons'].min():.0f} - {locs['photons'].max():.0f}")
        print(f"[Tuning] Net Gradient range: {locs['net_gradient'].min():.0f} - {locs['net_gradient'].max():.0f}")
        it = locs['iterations']
        print(f"[Tuning] Iterations: median={it.median():.0f}  max={it.max():.0f}  "
              f"≤10={((it <= 10).mean()*100):.0f}%  ≤50={((it <= 50).mean()*100):.0f}%")

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
    print(f"[Config] Batch mode: ON (all frames per file processed in single Picasso call)")
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
    checkpoint_path = os.path.join(OUTPUT_DIR, "vmi_results_raw.h5")
    if os.path.exists(checkpoint_path):
        print(f"\n[Resume] Found checkpoint, loading: {checkpoint_path}")
        df_all = pd.read_hdf(checkpoint_path, key='localizations')
        print(f"[Resume] Loaded {len(df_all)} events")
    else:
        # Batch processing
        df_all = run_batch_processing_picasso(h5_files, frame_info, dark_frame,
                                              flat_field_normalized)

        # Checkpoint save
        print(f"\n[Checkpoint] Saving raw detections to {checkpoint_path}")
        df_all.to_hdf(checkpoint_path, key='localizations', mode='w', complevel=1)
        print(f"[Checkpoint] Saved {len(df_all)} events")

    # ========================================================================
    # GMD EXTRACTION & NORMALIZATION
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("GMD NORMALIZATION")
    print(f"{'=' * 80}")

    if not ENABLE_GMD_WEIGHTING:
        print(f"[GMD] Weighting DISABLED (ENABLE_GMD_WEIGHTING = False)")
        print(f"[GMD] Skipping GMD extraction - all events will have weight = 1.0")
        df_all['gmd_energy'] = 1.0
        df_all['weight'] = 1.0
        gmd_enabled = False
    else:
        print(f"[GMD] Loading frame-to-GMD map...")
        gmd_array = load_gmd_map(GMD_MAP_PATH)

        if gmd_array is None:
            print(f"[GMD] WARNING: Cannot load GMD map! All weights set to 1.0")
            df_all['gmd_energy'] = 1.0
            df_all['weight'] = 1.0
            gmd_enabled = False
        else:
            valid_mask = np.isfinite(gmd_array)
            n_valid = valid_mask.sum()
            n_total = len(gmd_array)
            print(f"[GMD] Valid GMD values: {n_valid}/{n_total} frames")

            mean_gmd = gmd_array[valid_mask].mean()
            gmd_array = np.where(np.isnan(gmd_array), mean_gmd, gmd_array)
            print(f"[GMD] Mean: {mean_gmd:.3e}")

            # Vectorized lookup — avoid pandas.map lambda on 347M rows
            frame_idx = df_all['frame'].values.astype(int)
            gmd_energy = gmd_array[frame_idx]
            weight = mean_gmd / gmd_energy

            # Zero out weights outside [min, max] — these events contribute
            # nothing to the histogram (weight=0 in np.histogram2d).
            if GMD_WEIGHT_MAX > 0:
                weight[weight > GMD_WEIGHT_MAX] = 0.0
            if GMD_WEIGHT_MIN > 0:
                weight[weight < GMD_WEIGHT_MIN] = 0.0
            if GMD_WEIGHT_MIN > 0 or GMD_WEIGHT_MAX > 0:
                n_zero = (weight == 0).sum()
                print(f"[GMD] Weight filter [{GMD_WEIGHT_MIN}, {GMD_WEIGHT_MAX}]: "
                      f"{n_zero}/{len(weight)} events zeroed")

            df_all['weight'] = weight

            gmd_enabled = True

    # Create VMI histogram
    vmi_2d = create_vmi_histogram(df_all, img_shape)

    # Abel transform
    vmi_3d = perform_abel_transform(vmi_2d)

    # Save results
    save_results(vmi_2d, vmi_3d)

    # Visualization
    plot_final_summary(df_all, vmi_2d, vmi_3d)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
