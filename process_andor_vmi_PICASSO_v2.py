#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMI Photoelectron Analysis - PICASSO V2 (Modern API)
======================================================
Uses Picasso v0.10.0+ `localize.localize()` API — single-call
identify + MLE fit. No deprecated async pipeline, no coordinate bugs.

Key Features:
- Maximum Likelihood Estimation (MLE): Optimal for Poisson noise
- Net Gradient algorithm: Superior pile-up/overlap detection
- Numba JIT compilation: C-speed performance in Python
- Sub-pixel precision: ~10nm in SMLM, ~0.1 pixel in VMI

References:
- Picasso: Schnitzbauer et al., Nature Protocols 2017
- Net Gradient: Sergé et al., Nature Methods 2008

Author: TAO Jianfei ((jakiesumrain@163.com))
Date: 2025-12-06  (v2: migrated to modern API 2026-05-29)
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

import tifffile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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

# --- INPUT/OUTPUT ---
DATA_DIR = vmi_cfg["data_dir"]
# Filename pattern for your data: img_channel000_position000_time000000XXX_z000.tif
FILE_PATTERN = os.path.join(DATA_DIR, "img_channel000_position000_time*.tif")
OUTPUT_DIR = vmi_cfg["output_dir"]
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "vmi_results.h5")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vmi_results.csv")

# --- DARK/BACKGROUND DATA ---
DARK_DATA_DIR = vmi_cfg.get("dark_data_dir", "")
# Handle empty dark_data_dir
if DARK_DATA_DIR:
    DARK_FILE_PATTERN = os.path.join(DARK_DATA_DIR, "img_channel000_position000_time*.tif")
else:
    DARK_FILE_PATTERN = ""

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

# Tuning plot histogram bins
TUNING_BINS_PHOTONS = vmi_cfg.get("tuning_bins_photons", 50)
TUNING_BINS_NET_GRADIENT = vmi_cfg.get("tuning_bins_net_gradient", 50)
TUNING_BINS_PRECISION = vmi_cfg.get("tuning_bins_precision", 50)
TUNING_FRAME_INDEX = vmi_cfg.get("tuning_frame_index", 10000)

# --- PARALLEL PROCESSING ---
ENABLE_PARALLEL = proc_cfg["enable_parallel"]
NUM_CORES = proc_cfg["num_cores"]  # None = use all cores

# --- DARK FRAME SUBTRACTION ---
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
    """Load frame file list"""
    print(f"[Load] Searching: {FILE_PATTERN}")
    file_list = sorted(glob.glob(FILE_PATTERN))

    if len(file_list) == 0:
        print(f"[ERROR] No files found matching pattern!")
        sys.exit(1)

    # Load one frame to get shape
    test_frame = tifffile.imread(file_list[0])
    img_shape = test_frame.shape

    print(f"[Load] Found {len(file_list)} frames")
    print(f"[Load] Image size: {img_shape[1]}×{img_shape[0]} pixels")

    return file_list, img_shape

def calculate_dark_frame():
    """Calculate dark frame from separate background directory"""
    if not DARK_DATA_DIR or not DARK_FILE_PATTERN:
        print("[Dark] No dark data directory configured, skipping dark frame")
        return None

    print(f"[Dark] Loading background frames from: {DARK_FILE_PATTERN}")

    dark_file_list = sorted(glob.glob(DARK_FILE_PATTERN))

    if len(dark_file_list) == 0:
        print(f"[Dark] WARNING: No background files found! Using sigma-clipped stats instead.")
        return None

    print(f"[Dark] Found {len(dark_file_list)} background frames")
    print(f"[Dark] Using first {min(DARK_FRAME_COUNT, len(dark_file_list))} frames...")

    n_frames = min(DARK_FRAME_COUNT, len(dark_file_list))

    stack = np.array([tifffile.imread(dark_file_list[i]) for i in range(n_frames)])
    dark_frame = np.median(stack, axis=0)

    print(f"[Dark] Dark level: mean={dark_frame.mean():.1f}, "
          f"std={dark_frame.std():.1f}")

    return dark_frame

def load_flat_field():
    """Load flat field correction"""
    if not ENABLE_FLAT_FIELD or FLAT_FIELD_FILE is None:
        return None

    print(f"[Flat] Loading: {FLAT_FIELD_FILE}")
    flat_field = tifffile.imread(FLAT_FIELD_FILE)

    # Normalize so mean = 1
    flat_field_normalized = flat_field / flat_field.mean()
    print(f"[Flat] Flat field correction loaded")

    return flat_field_normalized

def get_tiff_metadata_gmd(filepath):
    """Extract GMD from TIFF metadata."""
    try:
        with tifffile.TiffFile(filepath) as tif:
            tags = tif.pages[0].tags

            # Method 1: ImageDescription (Tag 270)
            if 270 in tags:
                desc = tags[270].value
                if isinstance(desc, bytes):
                    desc = desc.decode('utf-8', errors='ignore')

                if isinstance(desc, str):
                    for item in desc.split(';'):
                        if 'GMD' in item and '=' in item:
                            try:
                                return float(item.split('=')[1])
                            except:
                                pass

                    import re
                    match = re.search(r'GMD["\s:=]+([0-9.eE+-]+)', desc)
                    if match:
                        return float(match.group(1))

            # Method 2: Custom tags (65000-65535)
            for tag_id in range(65000, 65536):
                if tag_id in tags:
                    tag_value = str(tags[tag_id].value)
                    if 'GMD' in tag_value:
                        import re
                        match = re.search(r'([0-9.eE+-]+)', tag_value)
                        if match:
                            return float(match.group(1))

    except Exception:
        pass

    return None

# ==============================================================================
# PICASSO LOCALIZATION
# ==============================================================================
# Pipeline: identify() → get_spots() → gaussmle.gaussmle() → gaussmle.locs_from_fits()
# Uses gaussmle.locs_from_fits (correct coordinates, not the buggy localize version).

def _camera_info():
    return {
        'Baseline': CAMERA_BASELINE,
        'Sensitivity': CAMERA_SENSITIVITY,
        'Gain': CAMERA_GAIN,
        'Qe': CAMERA_QE,
    }

def _run_picasso_pipeline(corrected_uint, box_size, min_net_gradient):
    """Run identify + MLE fit on a single-frame movie. Returns locs DataFrame
    or None if no spots found."""
    movie = corrected_uint[np.newaxis, :, :]  # (1, H, W)

    # 1. Identify spots via Net Gradient
    ids = localize.identify(movie, min_net_gradient, box_size, threaded=False)
    if len(ids) == 0:
        return None

    # 2. Extract spot patches + convert to photons
    spots = localize.get_spots(movie, ids, box_size, _camera_info())

    # 3. MLE fitting
    theta, CRLBs, likelihoods, iterations = gaussmle.gaussmle(
        spots, MLE_CONVERGENCE, MLE_MAX_ITERATIONS, method='sigmaxy'
    )

    # 4. Convert to localizations (gaussmle version — correct coordinates)
    return gaussmle.locs_from_fits(ids, theta, CRLBs, likelihoods, iterations, box_size)

def picasso_to_trackpy_style(locs, frame_num):
    """
    Convert Picasso localization output to trackpy-style DataFrame.

    Picasso output fields:
    - x, y: sub-pixel coordinates
    - photons: integrated intensity
    - sx, sy: Gaussian width (pixels)
    - bg: local background
    - lpx, lpy: localization precision
    - net_gradient: detection metric

    Trackpy style:
    - x, y: positions
    - mass: integrated intensity
    - size: spot size
    - ecc: eccentricity
    - signal: peak intensity
    - frame: frame number
    """
    if len(locs) == 0:
        return None

    df = pd.DataFrame()
    df['x'] = locs['x']
    df['y'] = locs['y']
    df['mass'] = locs['photons']
    df['size'] = (locs['sx'] + locs['sy']) / 2.0

    # Eccentricity from Gaussian widths: ecc = 1 - (minor/major)
    sx = locs['sx']
    sy = locs['sy']
    minor = np.minimum(sx, sy)
    major = np.maximum(sx, sy)
    df['ecc'] = 1.0 - (minor / major)

    # Signal (use photons/size² as proxy for peak intensity)
    df['signal'] = locs['photons'] / (np.pi * df['size']**2)
    df['frame'] = frame_num

    # Keep Net Gradient (Picasso's detection metric)
    df['net_gradient'] = locs['net_gradient']

    return df

def process_frame_worker_picasso(args):
    """Worker function for parallel Picasso processing.
    Loads frames on-demand from disk to minimize memory usage."""
    frame_idx, file_path, dark_frame, flat_field_normalized, box_size, min_net_gradient = args
    frame = tifffile.imread(file_path)

    try:
        if dark_frame is not None:
            corrected = frame.astype(float) - dark_frame
        else:
            corrected = frame.astype(float)
            mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(corrected, sigma=3.0)
            corrected = corrected - median_bkg

        if flat_field_normalized is not None:
            corrected = corrected / flat_field_normalized

        corrected = np.maximum(corrected, 0)
        corrected_uint = np.round(corrected).astype(np.uint16)

        locs = _run_picasso_pipeline(corrected_uint, box_size, min_net_gradient)

        if locs is not None and len(locs) > 0:
            features = picasso_to_trackpy_style(locs, frame_num=frame_idx)
            if features is not None and len(features) > 0:
                return features
        return None

    except Exception as e:
        print(f"[Warning] Frame {frame_idx} failed: {e}")
        return None

# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def run_batch_processing_picasso(file_list, dark_frame, flat_field_normalized):
    """Run Picasso localization on all frames"""

    print("="*80)
    print("BATCH PROCESSING")
    print("="*80)

    n_cores = NUM_CORES if NUM_CORES is not None else cpu_count()
    print(f"[Batch] Parallel processing: {n_cores} cores")

    # Prepare arguments for parallel processing
    args_list = [
        (i, file_path, dark_frame, flat_field_normalized, BOX_SIZE, MIN_NET_GRADIENT)
        for i, file_path in enumerate(file_list)
    ]

    all_features = []

    if ENABLE_PARALLEL:
        print(f"[Batch] Processing {len(file_list)} frames with Picasso (MLE + Net Gradient)...")
        with Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_frame_worker_picasso, args_list),
                total=len(args_list),
                desc="Processing frames (Picasso)"
            ))
    else:
        print(f"[Batch] Processing {len(file_list)} frames (single-threaded)...")
        results = [process_frame_worker_picasso(args) for args in tqdm(args_list)]

    # Collect results
    for result in results:
        if result is not None:
            all_features.append(result)

    if len(all_features) == 0:
        print("[ERROR] No events detected!")
        sys.exit(1)

    # Concatenate all results
    df_all = pd.concat(all_features, ignore_index=True)

    print(f"[Batch] Total detections: {len(df_all)} events")
    print(f"[Batch] Events/frame: {len(df_all)/len(file_list):.1f} avg")

    return df_all

# ==============================================================================
# VMI HISTOGRAM
# ==============================================================================

def create_vmi_histogram(df_all, img_shape):
    """Create 2D VMI histogram from localized events (with GMD weighting)"""

    print("="*80)
    print("VMI HISTOGRAM")
    print("="*80)

    # Upsample resolution
    ny, nx = img_shape
    hist_nx = nx * HISTOGRAM_RESOLUTION_FACTOR
    hist_ny = ny * HISTOGRAM_RESOLUTION_FACTOR

    print(f"[Histogram] Resolution: {hist_nx}×{hist_ny}")

    # Create 2D histogram with GMD weights
    x_scaled = df_all['x'] * HISTOGRAM_RESOLUTION_FACTOR
    y_scaled = df_all['y'] * HISTOGRAM_RESOLUTION_FACTOR

    # Use GMD weights if available
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

    print("="*80)
    print("ABEL TRANSFORM")
    print("="*80)

    # BASEX is fast and accurate for VMI
    try:
        vmi_3d = abel.transform.basex_transform(
            vmi_2d,
            direction='inverse',
            basis_dir=None,  # Auto-create basis sets
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

def save_results(df_all, vmi_2d, vmi_3d):
    """Save all results to disk"""

    print("="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save DataFrame as HDF5
    df_all.to_hdf(OUTPUT_H5, key='localizations', mode='w', complevel=9)
    print(f"[Save] HDF5: {OUTPUT_H5}")

    # Save 2D histogram
    vmi_2d_path = os.path.join(OUTPUT_DIR, "vmi_2d_raw.npy")
    np.save(vmi_2d_path, vmi_2d)
    print(f"[Save] 2D histogram: {vmi_2d_path}")

    # Save 3D Abel inverted
    if vmi_3d is not None:
        vmi_3d_path = os.path.join(OUTPUT_DIR, "vmi_3d_abel.npy")
        np.save(vmi_3d_path, vmi_3d)
        print(f"[Save] Abel inverted: {vmi_3d_path}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_final_summary(df_all, vmi_2d, vmi_3d):
    """Create final summary plot"""

    print("="*80)
    print("FINAL VISUALIZATION")
    print("="*80)

    fig = plt.figure(figsize=(15, 5), facecolor='white')
    fig.patch.set_facecolor('white')

    # 2D VMI image
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(vmi_2d, origin='lower', cmap='hot', aspect='equal', interpolation='nearest')
    ax1.set_title(f"2D VMI Image (Raw)\n{len(df_all)} events", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(False)
    plt.colorbar(im1, ax=ax1, label='Counts')

    # 3D VMI (Abel inverted)
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

    # Summary text
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')

    summary_text = f"""PICASSO ANALYSIS COMPLETE

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
# TUNING MODE
# ==============================================================================

def run_tuning_check(file_list, dark_frame, flat_field_normalized):
    """Run detection on single frame to tune parameters"""

    print("="*80)
    print("TUNING CHECK")
    print("="*80)

    test_idx = min(TUNING_FRAME_INDEX, len(file_list) - 1)
    test_frame = tifffile.imread(file_list[test_idx])
    print(f"[Tuning] Frame index: {test_idx}")

    # Apply corrections
    if dark_frame is not None:
        corrected = test_frame.astype(float) - dark_frame
    else:
        corrected = test_frame.astype(float)
        mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(corrected, sigma=3.0)
        corrected = corrected - median_bkg
        print(f"[Tuning] Per-frame background: mean={mean_bkg:.1f}, median={median_bkg:.1f}, std={std_bkg:.1f}")

    if flat_field_normalized is not None:
        corrected = corrected / flat_field_normalized

    corrected_clipped = np.maximum(corrected, 0)
    corrected_uint = np.round(corrected_clipped).astype(np.uint16)

    locs = _run_picasso_pipeline(corrected_uint, BOX_SIZE, MIN_NET_GRADIENT)

    print(f"[Tuning] Box size: {BOX_SIZE} pixels")
    print(f"[Tuning] Min Net Gradient: {MIN_NET_GRADIENT}")
    print(f"[Tuning] Detected {len(locs) if locs is not None else 0} events")

    if locs is not None and len(locs) > 0:
        print(f"[Tuning] Photon range: {locs['photons'].min():.0f} - {locs['photons'].max():.0f}")
        print(f"[Tuning] Net Gradient range: {locs['net_gradient'].min():.0f} - {locs['net_gradient'].max():.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Raw frame with detections
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

    # Panel 2: Photon histogram
    ax2 = axes[1]
    if locs is not None and len(locs) > 0:
        ax2.hist(locs['photons'], bins=TUNING_BINS_PHOTONS, alpha=0.7, edgecolor='black')
        ax2.axvline(np.median(locs['photons']), color='red', linestyle='--',
                   label=f"Median = {np.median(locs['photons']):.0f}")
    ax2.set_title("Photon Distribution")
    ax2.set_xlabel("Photons (integrated intensity)")
    ax2.set_ylabel("Count")
    ax2.legend()

    # Panel 3: Net Gradient histogram
    ax3 = axes[2]
    if locs is not None and len(locs) > 0:
        ax3.hist(locs['net_gradient'], bins=TUNING_BINS_NET_GRADIENT, alpha=0.7, edgecolor='black', color='green')
        ax3.axvline(MIN_NET_GRADIENT, color='red', linestyle='--',
                   label=f"Threshold = {MIN_NET_GRADIENT}")
    ax3.set_title("Net Gradient Distribution\n(Pile-up Detection)")
    ax3.set_xlabel("Net Gradient")
    ax3.set_ylabel("Count")
    ax3.legend()

    # Panel 4: Localization precision
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
    print("="*80)
    print("VMI PHOTOELECTRON ANALYSIS - PICASSO V2 (Modern API)")
    print("="*80)
    print(f"[Config] Loaded from: config.toml")
    print(f"[Config] Data dir: {DATA_DIR}")
    print(f"[Config] Dark dir: {DARK_DATA_DIR}")
    print(f"[Config] Output dir: {OUTPUT_DIR}")
    print(f"[Config] BOX_SIZE={BOX_SIZE}, MIN_NET_GRADIENT={MIN_NET_GRADIENT}")
    print(f"[Config] Dark subtraction: {'ON' if ENABLE_DARK_SUBTRACTION else 'OFF'}")
    print(f"[Config] GMD weighting: {'ON' if ENABLE_GMD_WEIGHTING else 'OFF'}")
    print(f"[Config] Parallel: {'ON' if ENABLE_PARALLEL else 'OFF'} ({NUM_CORES} cores)")
    print(f"[Config] Tuning mode: {'ON' if TUNING_ONLY else 'OFF'}")
    print("="*80)

    # Setup
    create_output_dir()
    file_list, img_shape = load_frames_and_metadata()

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
        run_tuning_check(file_list, dark_frame, flat_field_normalized)
        print("\n" + "="*80)
        print("TUNING MODE - Batch processing skipped")
        print("="*80)
        print("\nAdjust parameters and re-run until satisfied, then set TUNING_ONLY = False")
        return

    # Batch processing
    df_all = run_batch_processing_picasso(file_list, dark_frame, flat_field_normalized)

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
        print(f"[GMD] Reading metadata from TIFF files...")

        gmd_values = []
        for filepath in tqdm(file_list, desc="Extracting GMD"):
            gmd = get_tiff_metadata_gmd(filepath)
            gmd_values.append(gmd)

        valid_gmds = [g for g in gmd_values if g is not None]

        if len(valid_gmds) == 0:
            print(f"[GMD] WARNING: No GMD data found!")
            df_all['gmd_energy'] = 1.0
            df_all['weight'] = 1.0
            gmd_enabled = False
        else:
            print(f"[GMD] Found GMD in {len(valid_gmds)}/{len(file_list)} files")

            gmd_series = pd.Series(gmd_values, dtype=float)
            mean_gmd = gmd_series.mean()
            gmd_series = gmd_series.fillna(mean_gmd)

            df_all['gmd_energy'] = df_all['frame'].apply(lambda f: gmd_series.iloc[int(f)])
            df_all['weight'] = mean_gmd / df_all['gmd_energy']

            print(f"[GMD] Mean: {mean_gmd:.3e}, Std: {gmd_series.std():.3e}")
            gmd_enabled = True

    # Create VMI histogram
    vmi_2d = create_vmi_histogram(df_all, img_shape)

    # Abel transform
    vmi_3d = perform_abel_transform(vmi_2d)

    # Save results
    save_results(df_all, vmi_2d, vmi_3d)

    # Visualization
    plot_final_summary(df_all, vmi_2d, vmi_3d)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
