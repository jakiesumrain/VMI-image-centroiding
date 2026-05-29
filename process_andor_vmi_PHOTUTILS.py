#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMI Photoelectron Analysis - PHOTUTILS VERSION (with Pile-up Recovery)
=======================================================================
Advanced implementation using astronomy community's photutils package
with IterativePSFPhotometry + SourceGrouper for pile-up deblending.

Key Features:
- IterativePSFPhotometry: Iterative PSF fitting for crowded fields
- SourceGrouper: Groups overlapping sources for SIMULTANEOUS fitting
- Pile-up Recovery: Deblends merged multi-electron events
- Sub-pixel precision centroiding

Algorithm (for pile-up):
   Blob → Group overlapping sources → Fit PSF₁+PSF₂+PSF₃ SIMULTANEOUSLY
   (Joint optimization recovers individual electron positions)

Maintains all other features:
- Dark frame subtraction
- Flat field correction
- GMD normalization
- Saturation/cosmic ray filtering
- Abel transform
- Comprehensive diagnostics

Author: VMI Analysis Pipeline (photutils deblending variant)
Date: 2025-12-05
"""

from __future__ import annotations
import os
import sys
import glob
import pims
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifffile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Photutils imports for PSF photometry with deblending
from photutils.detection import DAOStarFinder
from photutils.psf import (
    PSFPhotometry,
    IterativePSFPhotometry,
    SourceGrouper,
    CircularGaussianSigmaPRF,  # Replaces deprecated IntegratedGaussianPRF
)
from photutils.background import LocalBackground, MMMBackground
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- INPUT/OUTPUT ---
DATA_DIR = "F:/data/int5_kr/1216_Kr_ele_deg54_9ev_0d059W_sig/_1/Default"
FILE_PATTERN = os.path.join(DATA_DIR, "img_*.tif")
OUTPUT_DIR = "vmi_analysis_photutils_ele0p812"
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "vmi_results.h5")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vmi_results.csv")

# --- DARK/BACKGROUND DATA ---
DARK_DATA_DIR = "ele0p812_60ev_5ms_20000_B/_1/Default"
DARK_FILE_PATTERN = os.path.join(DARK_DATA_DIR, "img_*.tif")

# --- CAMERA SETTINGS ---
BINNING = 1  # 0=1×1, 1=2×2, 2=3×3, 3=4×4, 4=8×8

# --- CENTROIDING PARAMETERS (PHOTUTILS STYLE) ---
# ============================================================================
# PHOTUTILS uses astronomy terminology - convert from VMI concepts:
#
# FWHM (Full-Width Half-Maximum):
#   - Astronomy: Expected FWHM of star PSF in pixels
#   - VMI equivalent: PSF size of photoelectron blob
#   - Relationship: FWHM ≈ 0.6 × DIAMETER (trackpy)
#   - For 1×1 binning: FWHM ≈ 4 pixels (trackpy DIAMETER=7)
#   - For 2×2 binning: FWHM ≈ 2 pixels (trackpy DIAMETER=3)
#
# THRESHOLD:
#   - Astronomy: Detection threshold in units of background σ (sigma)
#   - VMI equivalent: How many sigma above background to detect
#   - Higher = fewer false detections, but may miss dim events
#   - Typical: 3-5 sigma (very conservative for low-noise VMI data)
#
# SHARPNESS:
#   - Astronomy: Rejects cosmic rays and extended sources
#   - Range: 0.2-1.0 (0.5 = Gaussian, <0.2 = cosmic ray, >1.0 = extended)
#   - VMI equivalent: Blob shape filter (like trackpy's ecc)
#
# ROUNDNESS:
#   - Astronomy: Rejects non-circular sources
#   - Range: -1 to +1 (0 = perfectly circular, ±1 = very elliptical)
#   - VMI equivalent: Another cosmic ray filter
#
# TUNING WORKFLOW:
# 1. Run with defaults
# 2. Check tuning plot - too few detections? Decrease THRESHOLD
# 3. Too many noise dots? Increase THRESHOLD or tighten SHARPNESS/ROUNDNESS
# 4. Compare to trackpy version to verify similar event counts
# ============================================================================
FWHM = 4.0 if BINNING == 0 else 2.5  # Keep at 2.5 (matches actual PSF)

# Detection threshold (sigma above background)
# Lower = more sensitive (may detect noise)
# Higher = more conservative (may miss dim events)
DETECTION_THRESHOLD = 9.0  # Increased to 12σ for very conservative detection

# Shape filters (reject cosmic rays) - used by DAOStarFinder for initial detection
SHARPLO = 0.4   # Minimum sharpness (reject extended blobs) - increased for stricter filtering
SHARPHI = 1.0   # Maximum sharpness (standard Gaussian)
ROUNDLO = -0.4  # Minimum roundness - tightened from -1.0
ROUNDHI = 0.4  # Maximum roundness - tightened from 1.0

# --- PILE-UP RECOVERY (PSF DEBLENDING) ---
# ============================================================================
# THE PILE-UP PROBLEM IN STRONG-FIELD IONIZATION:
# When multiple photoelectrons hit the detector within ~FWHM pixels of each
# other (within same camera frame), simple centroiding merges them into a
# single "event" with wrong position (centroid of multiple electrons).
#
# SOLUTION: ITERATIVE PSF PHOTOMETRY WITH SOURCE GROUPING
# Instead of simple center-of-mass centroiding, we use:
# 1. DAOStarFinder for initial source detection
# 2. SourceGrouper to identify overlapping sources (potential pile-up)
# 3. IterativePSFPhotometry to fit multiple PSFs SIMULTANEOUSLY
#
# HOW SIMULTANEOUS FITTING SOLVES PILE-UP:
# - Two electrons hit side-by-side → Creates "peanut" shaped blob
# - Simple COM centroiding: Reports 1 electron in the middle (WRONG!)
# - PSF fitting with grouping: Fits 2 Gaussians → Reports 2 electrons (CORRECT!)
#
# GROUPING_SEPARATION:
#   - Sources closer than this are grouped for simultaneous fitting
#   - Rule of thumb: 2-3 × FWHM
#   - Too small: Pile-up events not grouped → not deblended
#   - Too large: Distant sources grouped → slower, may cause fitting issues
#
# FIT_SHAPE:
#   - Size of the fitting window around each source (pixels)
#   - Should be large enough to contain the full PSF
#   - Rule of thumb: 2-3 × FWHM, must be odd integer
#
# APERTURE_RADIUS:
#   - Radius for initial flux estimation (before PSF fitting)
#   - Rule of thumb: 1-1.5 × FWHM
# ============================================================================
ENABLE_PILE_UP_RECOVERY = True  # Enable PSF deblending for pile-up recovery

# Grouping: Sources closer than this are fit simultaneously
GROUPING_SEPARATION = 3. * FWHM  # pixels (increased to 3.5× for more aggressive pile-up grouping)

# PSF fitting window size (must be odd)
# Larger window captures more of the PSF but slower fitting
FIT_SHAPE = (11, 11) if BINNING == 0 else (9, 9)  # Increased to (11,11) for better PSF wing capture

# Aperture radius for initial flux estimation
APERTURE_RADIUS = 1.5 * FWHM  # pixels

# Local background estimation annulus (inner, outer radius)
LOCAL_BKG_INNER = 2.0 * FWHM  # Start of background annulus
LOCAL_BKG_OUTER = 3.0 * FWHM  # End of background annulus

# --- FILTERING PARAMETERS ---
SATURATION_LEVEL = 65000  # 16-bit camera
ECC_CUTOFF = 0.7  # Cosmic ray rejection (converted to roundness in photutils)

# --- CIRCULAR MASK (MCP Active Area) ---
# Same as hybrid version - masks events outside MCP active area
ENABLE_CIRCULAR_MASK = False  # Set True after determining geometry
MASK_CENTER_X = None  # e.g., 512.5 for 1024×1024 image
MASK_CENTER_Y = None  # e.g., 512.5 for 1024×1024 image
MASK_RADIUS = None  # e.g., 480 for typical 1024×1024 VMI

# --- PROCESSING OPTIONS ---
DARK_FRAME_COUNT = 100  # Number of background frames to average

# --- PARALLEL PROCESSING ---
# ============================================================================
# PSF photometry is CPU-intensive. Enable parallel processing to use multiple
# cores for simultaneous frame processing.
#
# ENABLE_PARALLEL:
#   - True: Process frames in parallel using multiprocessing
#   - False: Process frames sequentially (lower memory usage)
#
# NUM_CORES:
#   - Number of CPU cores to use for parallel processing
#   - None: Use all available cores (recommended)
#   - Integer: Specify exact number of cores (e.g., 4, 8)
#   - Only used if ENABLE_PARALLEL = True
#
# SPEED IMPROVEMENT:
#   - Sequential: ~8-10 seconds/frame → ~25 hours for 10,248 frames
#   - Parallel (8 cores): ~1-1.5 seconds/frame → ~3-4 hours for 10,248 frames
#   - Parallel (16 cores): ~0.5-1 seconds/frame → ~1.5-2.5 hours for 10,248 frames
# ============================================================================
ENABLE_PARALLEL = True  # Now safe with on-demand frame loading!
NUM_CORES = 4  # None = use all cores, or specify integer (e.g., 8)

# --- DARK FRAME SUBTRACTION ---
# ============================================================================
# Set ENABLE_DARK_SUBTRACTION = True to use separate background frames.
#
# When enabled:
#   - Loads dark frames from DARK_FILE_PATTERN (separate background directory)
#   - First DARK_FRAME_COUNT frames are averaged to create dark frame
#   - Dark frame is subtracted from all signal frames
#   - IMPORTANT: Background directory must contain true darks (no signal!)
#
# When disabled:
#   - No dark subtraction is performed
#   - Background is estimated per-frame using sigma-clipped statistics
#   - This works well if your images have sparse events (typical VMI)
# ============================================================================
ENABLE_DARK_SUBTRACTION = False  # Set True to use separate background frames

ENABLE_FLAT_FIELD = False
FLAT_FIELD_PATH = "masterflat.npy"
ENABLE_ABEL = False
HISTOGRAM_RESOLUTION_FACTOR = 1

# --- TUNING MODE ---
# ============================================================================
# Set TUNING_ONLY = True to run ONLY the tuning diagnostic on a single frame.
# This allows rapid iteration on centroiding parameters (FWHM, DETECTION_THRESHOLD,
# GROUPING_SEPARATION, etc.) without waiting for full batch processing.
#
# WORKFLOW:
# 1. Set TUNING_ONLY = True
# 2. Run script, check tuning plot
# 3. Adjust parameters, repeat until satisfied
# 4. Set TUNING_ONLY = False
# 5. Run full batch processing
# ============================================================================
TUNING_ONLY = True  # Set False to run full batch processing

# --- GMD NORMALIZATION ---
# ============================================================================
# GMD (Gas Monitor Detector) measures FEL pulse energy shot-to-shot.
# Normalization corrects for pulse energy fluctuations.
#
# ENABLE_GMD_WEIGHTING:
#   - True: Weight each event by mean_gmd / gmd_energy (standard for linear processes)
#   - False: All events have weight = 1.0 (no GMD correction)
#
# PHYSICS CAVEAT FOR STRONG-FIELD IONIZATION (SFI):
# The default linear weighting (weight = mean/actual) assumes signal ∝ intensity.
# In SFI, ionization rate often scales as I^N (N = multiphoton order, e.g., 4-6).
# For large pulse fluctuations (>10%), linear normalization may be insufficient.
# Consider:
#   - Binning data by GMD ranges instead of weighting
#   - Using weight = (mean/actual)^N if you know the nonlinearity order
#   - Setting ENABLE_GMD_WEIGHTING = False and analyzing GMD bins separately
# ============================================================================
ENABLE_GMD_WEIGHTING = False  # Set False to disable GMD-based event weighting

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_output_dir():
    """Create output directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_tiff_metadata_gmd(filepath):
    """
    Extract GMD from TIFF metadata.
    (Same implementation as hybrid version)
    """
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

def load_frames_and_metadata():
    """Load frame sequence and extract metadata"""
    print(f"[Load] Searching: {FILE_PATTERN}")
    file_list = sorted(glob.glob(FILE_PATTERN))

    if len(file_list) == 0:
        print(f"[ERROR] No files found matching pattern!")
        sys.exit(1)

    # Pass pattern directly to pims.open() instead of file list
    frames = pims.open(FILE_PATTERN)
    img_shape = frames[0].shape

    print(f"[Load] Found {len(frames)} frames")
    print(f"[Load] Image size: {img_shape[1]}×{img_shape[0]} pixels")

    return frames, file_list, img_shape

def calculate_dark_frame():
    """
    Calculate dark frame from SEPARATE BACKGROUND DIRECTORY.

    Loads background frames from DARK_FILE_PATTERN instead of signal frames.
    This ensures pure background without signal contamination.
    """
    print(f"[Dark] Loading background frames from: {DARK_FILE_PATTERN}")

    dark_file_list = sorted(glob.glob(DARK_FILE_PATTERN))

    if len(dark_file_list) == 0:
        print(f"[Dark] ERROR: No background files found!")
        print(f"[Dark] Check path: {DARK_DATA_DIR}")
        sys.exit(1)

    print(f"[Dark] Found {len(dark_file_list)} background frames")
    print(f"[Dark] Using first {min(DARK_FRAME_COUNT, len(dark_file_list))} frames for dark calculation...")

    # Load first N background frames
    n_frames = min(DARK_FRAME_COUNT, len(dark_file_list))
    dark_frames = pims.open(DARK_FILE_PATTERN)

    stack = np.array([dark_frames[i] for i in range(n_frames)])
    dark_frame = np.median(stack, axis=0)

    print(f"[Dark] Dark level: mean={dark_frame.mean():.1f}, "
          f"std={dark_frame.std():.1f}")

    return dark_frame

def check_saturation(img, threshold=SATURATION_LEVEL):
    """Check for saturated pixels"""
    n_saturated = np.sum(img >= threshold)
    return n_saturated

def daofind_to_trackpy_style(sources_table, frame_num=None):
    """
    Convert DAOStarFinder output table to trackpy-style DataFrame.

    DAOStarFinder columns:
    - xcentroid, ycentroid: Sub-pixel positions (like trackpy x, y)
    - sharpness: Blob shape metric
    - roundness1: Circularity metric
    - flux: Integrated brightness (like trackpy mass)
    - peak: Peak pixel value (like trackpy signal)

    Trackpy-style output for compatibility with rest of pipeline.
    """
    if sources_table is None or len(sources_table) == 0:
        return pd.DataFrame(columns=['x', 'y', 'mass', 'signal', 'ecc', 'frame'])

    df = pd.DataFrame({
        'x': sources_table['xcentroid'],
        'y': sources_table['ycentroid'],
        'mass': sources_table['flux'],  # Integrated brightness
        'signal': sources_table['peak'],  # Peak pixel value
        'sharpness': sources_table['sharpness'],
        'roundness': sources_table['roundness1'],
    })

    # Convert roundness to eccentricity-like metric
    df['ecc'] = np.abs(df['roundness'])

    if frame_num is not None:
        df['frame'] = frame_num

    return df

def psfphot_to_trackpy_style(phot_table, frame_num=None):
    """
    Convert PSFPhotometry/IterativePSFPhotometry output to trackpy-style DataFrame.

    PSFPhotometry columns:
    - x_fit, y_fit: Fitted sub-pixel positions
    - flux_fit: Fitted flux (like trackpy mass)
    - group_id: Which group this source belongs to (for pile-up detection)
    - group_size: Number of sources in the group (>1 = deblended pile-up)
    - iter_detected: Iteration when source was found (for IterativePSFPhotometry)

    Trackpy-style output for compatibility with rest of pipeline.
    """
    if phot_table is None or len(phot_table) == 0:
        return pd.DataFrame(columns=['x', 'y', 'mass', 'signal', 'ecc', 'frame',
                                     'group_id', 'group_size', 'deblended'])

    df = pd.DataFrame({
        'x': np.array(phot_table['x_fit']),
        'y': np.array(phot_table['y_fit']),
        'mass': np.array(phot_table['flux_fit']),
    })

    # Note: Negative flux values from PSF fitting are kept as-is
    # They indicate locations where background > signal and can be filtered later if needed

    # Add group information if available (indicates pile-up recovery)
    if 'group_id' in phot_table.colnames:
        df['group_id'] = np.array(phot_table['group_id'])
    else:
        df['group_id'] = np.arange(len(phot_table))  # Each source is its own group

    if 'group_size' in phot_table.colnames:
        df['group_size'] = np.array(phot_table['group_size'])
    else:
        df['group_size'] = 1

    # Mark sources that were deblended (group_size > 1)
    df['deblended'] = df['group_size'] > 1

    # PSF photometry doesn't provide peak signal directly, estimate from flux
    # For a 2D Gaussian: flux = 2πσ² × peak, therefore peak = flux / (2πσ²)
    sigma = FWHM / 2.355
    df['signal'] = df['mass'] / (2 * np.pi * sigma**2)

    # No direct eccentricity from PSF fitting, set to 0 (circular assumption)
    df['ecc'] = 0.0

    if frame_num is not None:
        df['frame'] = frame_num

    return df

# ==============================================================================
# PARALLEL PROCESSING WORKER
# ==============================================================================

def process_frame_worker(args):
    """
    Worker function for parallel frame processing.

    This function loads frames on-demand from disk to minimize memory usage.

    Args:
        args: Tuple of (frame_index, file_path, dark_frame, flat_field_normalized)

    Returns:
        pd.DataFrame with detected events for this frame, or None if no events found
    """
    frame_idx, file_path, dark_frame, flat_field_normalized = args

    # Load frame from disk (on-demand, not pre-loaded in memory)
    frame = tifffile.imread(file_path)

    try:
        # Apply corrections
        if dark_frame is not None:
            corrected = frame - dark_frame
        else:
            corrected = frame.astype(float)
        if flat_field_normalized is not None:
            corrected = corrected / flat_field_normalized

        # Calculate background for this frame
        mean_i, median_i, std_i = sigma_clipped_stats(corrected, sigma=3.0)
        data_sub_i = corrected - median_i

        # Clip negative values to zero
        data_sub_i = np.clip(data_sub_i, 0, None)

        if ENABLE_PILE_UP_RECOVERY:
            # === PSF PHOTOMETRY WITH DEBLENDING ===
            # Create PSF model
            sigma = FWHM / 2.355
            psf_model = CircularGaussianSigmaPRF(sigma=sigma)

            # Create components
            grouper = SourceGrouper(min_separation=GROUPING_SEPARATION)
            bkgstat = MMMBackground()
            localbkg_estimator = LocalBackground(
                inner_radius=LOCAL_BKG_INNER,
                outer_radius=LOCAL_BKG_OUTER,
                bkg_estimator=bkgstat
            )

            # Create finder
            daofind_i = DAOStarFinder(
                fwhm=FWHM,
                threshold=DETECTION_THRESHOLD * std_i,
                sharplo=SHARPLO,
                sharphi=SHARPHI,
                roundlo=ROUNDLO,
                roundhi=ROUNDHI,
                exclude_border=True
            )

            # Create PSF photometry object
            psfphot_i = IterativePSFPhotometry(
                psf_model=psf_model,
                fit_shape=FIT_SHAPE,
                finder=daofind_i,
                grouper=grouper,
                localbkg_estimator=localbkg_estimator,
                aperture_radius=APERTURE_RADIUS,
            )

            # Estimate error
            error_i = np.full_like(data_sub_i, std_i, dtype=float)

            # Run PSF photometry
            phot_result = psfphot_i(data_sub_i, error=error_i)

            if phot_result is not None and len(phot_result) > 0:
                features = psfphot_to_trackpy_style(phot_result, frame_num=frame_idx)
                return features
            else:
                return None

        else:
            # === SIMPLE DAOFIND (NO DEBLENDING) ===
            daofind_i = DAOStarFinder(
                fwhm=FWHM,
                threshold=DETECTION_THRESHOLD * std_i,
                sharplo=SHARPLO,
                sharphi=SHARPHI,
                roundlo=ROUNDLO,
                roundhi=ROUNDHI,
                exclude_border=True
            )

            sources = daofind_i(data_sub_i)

            if sources is not None and len(sources) > 0:
                features = daofind_to_trackpy_style(sources, frame_num=frame_idx)
                return features
            else:
                return None

    except Exception as e:
        # Silently skip frames with errors (e.g., fitting failures)
        return None

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def main():
    create_output_dir()

    print("=" * 80)
    print("VMI PHOTOELECTRON ANALYSIS - PHOTUTILS VERSION")
    if ENABLE_PILE_UP_RECOVERY:
        print("(with Pile-up Recovery via IterativePSFPhotometry + SourceGrouper)")
    print("=" * 80)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    frames, file_list, img_shape = load_frames_and_metadata()

    # Calculate dark frame from separate background directory
    if ENABLE_DARK_SUBTRACTION:
        dark_frame = calculate_dark_frame()
    else:
        dark_frame = None
        print(f"[Dark] Dark subtraction DISABLED (ENABLE_DARK_SUBTRACTION = False)")
        print(f"[Dark] Background will be estimated per-frame using sigma-clipped stats")

    # Load flat field if enabled (use local variable to avoid scoping issues)
    use_flat_field = ENABLE_FLAT_FIELD
    flat_field_normalized = None
    if use_flat_field:
        print(f"\n[Flat] Loading flat field: {FLAT_FIELD_PATH}")
        try:
            flat_field = np.load(FLAT_FIELD_PATH)
            if flat_field.shape != img_shape:
                print(f"[Flat] ERROR: Shape mismatch!")
                use_flat_field = False
            else:
                flat_field_normalized = flat_field / flat_field.mean()
                print(f"[Flat] Loaded successfully")
        except Exception as e:
            print(f"[Flat] ERROR: {e}")
            use_flat_field = False

    # ========================================================================
    # 2. TUNING CHECK
    # ========================================================================
    test_idx = len(frames) // 2
    raw_test = frames[test_idx]

    # Apply dark subtraction if enabled
    if dark_frame is not None:
        corrected_test = raw_test - dark_frame
    else:
        corrected_test = raw_test.astype(float)  # Work with float for consistency

    if use_flat_field and flat_field_normalized is not None:
        corrected_test = corrected_test / flat_field_normalized

    sat_pixels = check_saturation(raw_test)
    if sat_pixels > 0:
        print(f"\n[WARNING] Test frame has {sat_pixels} saturated pixels!")

    print(f"\n{'=' * 80}")
    print(f"TUNING CHECK (Frame {test_idx})")
    print(f"{'=' * 80}")
    print(f"[Tuning] FWHM: {FWHM} pixels (PSF size)")
    print(f"[Tuning] Detection threshold: {DETECTION_THRESHOLD}σ above background")
    print(f"[Tuning] Pile-up recovery: {'ENABLED' if ENABLE_PILE_UP_RECOVERY else 'DISABLED'}")
    if ENABLE_PILE_UP_RECOVERY:
        print(f"[Tuning] Grouping separation: {GROUPING_SEPARATION:.1f} pixels")
        print(f"[Tuning] Fit shape: {FIT_SHAPE}")

    # Calculate background statistics using sigma-clipped stats
    # (robust to bright sources)
    mean, median, std = sigma_clipped_stats(corrected_test, sigma=3.0)
    print(f"[Tuning] Background: mean={mean:.1f}, median={median:.1f}, std={std:.1f}")

    # Subtract background for source detection
    data_sub = corrected_test - median

    # Clip negative values to zero (physically meaningful: no negative photon counts)
    # This prevents PSF fitting from trying to fit negative "peaks"
    data_sub = np.clip(data_sub, 0, None)
    print(f"[Tuning] Clipped {np.sum(corrected_test - median < 0)} negative pixels to zero")

    # Create DAOStarFinder for initial source detection
    daofind = DAOStarFinder(
        fwhm=FWHM,
        threshold=DETECTION_THRESHOLD * std,
        sharplo=SHARPLO,
        sharphi=SHARPHI,
        roundlo=ROUNDLO,
        roundhi=ROUNDHI,
        exclude_border=True
    )

    if ENABLE_PILE_UP_RECOVERY:
        # === PSF PHOTOMETRY WITH DEBLENDING ===
        print(f"[Tuning] Using IterativePSFPhotometry with SourceGrouper...")

        # Create PSF model (Circular Gaussian for sub-pixel accuracy)
        sigma = FWHM / 2.355  # Convert FWHM to sigma
        psf_model = CircularGaussianSigmaPRF(sigma=sigma)

        # NOTE: We don't set flux.bounds here because initial guesses from aperture
        # photometry can be negative, causing "initial guess outside bounds" errors.
        # Instead, we'll clip negative flux values in post-processing.

        # Create source grouper for simultaneous fitting of overlapping sources
        grouper = SourceGrouper(min_separation=GROUPING_SEPARATION)

        # Create local background estimator
        bkgstat = MMMBackground()
        localbkg_estimator = LocalBackground(
            inner_radius=LOCAL_BKG_INNER,
            outer_radius=LOCAL_BKG_OUTER,
            bkg_estimator=bkgstat
        )

        # Use default fitter (no bounds - we'll clip negative flux in post-processing)
        # Create IterativePSFPhotometry object
        psfphot = IterativePSFPhotometry(
            psf_model=psf_model,
            fit_shape=FIT_SHAPE,
            finder=daofind,
            grouper=grouper,
            localbkg_estimator=localbkg_estimator,
            aperture_radius=APERTURE_RADIUS,
        )

        # Estimate error from background std
        error = np.full_like(data_sub, std, dtype=float)

        # Run PSF photometry
        try:
            phot_result = psfphot(data_sub, error=error)

            if phot_result is None or len(phot_result) == 0:
                print(f"[Tuning] Found 0 events - threshold too high or no signal!")
                test_features = pd.DataFrame(columns=['x', 'y', 'mass', 'signal', 'ecc'])
                n_deblended = 0
            else:
                print(f"[Tuning] Detected {len(phot_result)} events (after deblending)")

                # Count deblended events
                if 'group_size' in phot_result.colnames:
                    n_deblended = np.sum(np.array(phot_result['group_size']) > 1)
                    n_groups_multi = len(set(phot_result['group_id'][phot_result['group_size'] > 1]))
                    print(f"[Tuning] Pile-up recovery: {n_deblended} events from {n_groups_multi} pile-up groups")
                else:
                    n_deblended = 0

                print(f"[Tuning] Flux range: {phot_result['flux_fit'].min():.0f} - "
                      f"{phot_result['flux_fit'].max():.0f}")

                # Convert to trackpy-style DataFrame
                test_features = psfphot_to_trackpy_style(phot_result)

                # Report negative flux if present
                n_negative = np.sum(np.array(phot_result['flux_fit']) < 0)
                if n_negative > 0:
                    print(f"[Tuning] Note: {n_negative} events have negative flux (background > signal)")

        except Exception as e:
            print(f"[Tuning] PSF photometry failed: {e}")
            print(f"[Tuning] Falling back to DAOStarFinder...")
            sources = daofind(data_sub)
            if sources is None or len(sources) == 0:
                test_features = pd.DataFrame(columns=['x', 'y', 'mass', 'signal', 'ecc'])
            else:
                test_features = daofind_to_trackpy_style(sources)
            n_deblended = 0

    else:
        # === SIMPLE DAOFIND (NO DEBLENDING) ===
        print(f"[Tuning] Using DAOStarFinder (no pile-up recovery)...")
        sources = daofind(data_sub)

        if sources is None or len(sources) == 0:
            print(f"[Tuning] Found 0 events - threshold too high or no signal!")
            print(f"[Tuning] Try decreasing DETECTION_THRESHOLD")
            test_features = pd.DataFrame(columns=['x', 'y', 'mass', 'signal', 'ecc'])
        else:
            print(f"[Tuning] Detected {len(sources)} events")
            print(f"[Tuning] Flux range: {sources['flux'].min():.0f} - "
                  f"{sources['flux'].max():.0f}")
            test_features = daofind_to_trackpy_style(sources)
        n_deblended = 0

    # Visualization (3-panel or 4-panel depending on pile-up recovery)
    if ENABLE_PILE_UP_RECOVERY:
        fig = plt.figure(figsize=(20, 6))
        n_panels = 4
    else:
        fig = plt.figure(figsize=(18, 6))
        n_panels = 3

    # Panel 1: Detected sources
    ax1 = plt.subplot(1, n_panels, 1)

    # Clip negative values for better visualization (matches HYBRID style)
    corrected_display = np.clip(corrected_test, 0, None)

    # Use linear scale with better dynamic range (99.5th percentile)
    positive_pixels = corrected_display[corrected_display > 0]
    if len(positive_pixels) > 0:
        vmax = np.percentile(positive_pixels, 99.5)
    else:
        vmax = corrected_display.max()

    im1 = ax1.imshow(corrected_display, cmap='gray', origin='lower',
                     vmin=0, vmax=vmax)

    if len(test_features) > 0:
        # Color-code by deblended status if available (subtle markers)
        if 'deblended' in test_features.columns and ENABLE_PILE_UP_RECOVERY:
            single = test_features[~test_features['deblended']]
            deblend = test_features[test_features['deblended']]
            ax1.scatter(single['x'], single['y'],
                       s=30, facecolors='none', edgecolors='cyan', linewidths=0.6,
                       alpha=0.7, label=f'Single ({len(single)})')
            ax1.scatter(deblend['x'], deblend['y'],
                       s=40, facecolors='none', edgecolors='red', linewidths=0.8,
                       alpha=0.8, label=f'Deblended ({len(deblend)})')
            ax1.legend(loc='upper right', fontsize=8)
        else:
            ax1.scatter(test_features['x'], test_features['y'],
                       s=30, facecolors='none', edgecolors='red', linewidths=0.6,
                       alpha=0.7)

    # Fix axis limits to image size (prevent scatter from expanding axes)
    ax1.set_xlim([0, corrected_display.shape[1]])
    ax1.set_ylim([0, corrected_display.shape[0]])

    title_str = f"Tuning Check: {len(test_features)} events\nFWHM={FWHM}, Threshold={DETECTION_THRESHOLD}σ"
    if ENABLE_PILE_UP_RECOVERY:
        title_str += f"\nPile-up Recovery: ON"
    ax1.set_title(title_str)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    # Panel 2: Quality metrics
    ax2 = plt.subplot(1, n_panels, 2)
    if len(test_features) > 0 and 'sharpness' in test_features.columns:
        scatter = ax2.scatter(test_features['roundness'], test_features['sharpness'],
                           c=test_features['mass'], cmap='viridis', alpha=0.6, s=30)
        ax2.axhline(SHARPLO, color='r', linestyle='--', label=f'Sharp limits')
        ax2.axhline(SHARPHI, color='r', linestyle='--')
        ax2.axvline(ROUNDLO, color='orange', linestyle='--', label=f'Round limits')
        ax2.axvline(ROUNDHI, color='orange', linestyle='--')
        ax2.set_xlabel('Roundness')
        ax2.set_ylabel('Sharpness')
        ax2.set_title('Feature Quality (DAOStarFinder)')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Flux')
    elif len(test_features) > 0:
        # For PSF photometry, show group_size distribution
        if 'group_size' in test_features.columns:
            sizes = test_features['group_size'].value_counts().sort_index()
            ax2.bar(sizes.index, sizes.values, color='steelblue', edgecolor='black')
            ax2.set_xlabel('Group Size (1=single, 2+=pile-up)')
            ax2.set_ylabel('Count')
            ax2.set_title('Source Grouping Distribution')
            ax2.set_xticks(range(1, int(sizes.index.max()) + 1))
        else:
            ax2.text(0.5, 0.5, 'PSF fitting mode\n(no sharpness/roundness)',
                   ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No events detected',
               ha='center', va='center', transform=ax2.transAxes)

    # Panel 3: Mass (Flux) histogram with pile-up check
    ax3 = plt.subplot(1, n_panels, 3)
    if len(test_features) > 0:
        n, bins, patches = ax3.hist(test_features['mass'], bins=50,
                                    color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Flux (Integrated Brightness)')
        ax3.set_ylabel('Count')
        ax3.set_title('Flux Distribution (Pile-up Check)')

        # Estimate single-electron peak using MEDIAN (robust to outliers and low statistics)
        # MEDIAN is more reliable than MODE for:
        #   - Low statistics (few events per bin → Poisson noise)
        #   - Skewed distributions (long tail to high mass)
        #   - Negative flux values (from PSF fitting)
        # Mode would pick the tallest bin (often affected by binning artifacts)
        peak_mass = np.median(test_features['mass'])
        ax3.axvline(peak_mass, color='green', linestyle=':',
                   linewidth=2, label=f'1e⁻ peak ≈ {peak_mass:.0f} (median)')
        ax3.axvline(peak_mass * 2, color='orange', linestyle=':',
                   linewidth=2, label=f'2e⁻ location ≈ {peak_mass*2:.0f}')

        # Pile-up estimate
        window = peak_mass * 0.3
        pile_up_count = np.sum((test_features['mass'] > peak_mass * 2 - window) &
                               (test_features['mass'] < peak_mass * 2 + window))
        pile_up_fraction = 100 * pile_up_count / len(test_features)

        textstr = f'Events in 2e⁻ region: {pile_up_count} ({pile_up_fraction:.1f}%)'
        if pile_up_fraction > 5:
            textstr += '\n⚠️ HIGH PILE-UP!'
            ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        else:
            ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No flux data',
               ha='center', va='center', transform=ax3.transAxes)

    # Panel 4: Residual image (only for pile-up recovery mode)
    if ENABLE_PILE_UP_RECOVERY and n_panels == 4:
        ax4 = plt.subplot(1, n_panels, 4)
        try:
            resid = psfphot.make_residual_image(data_sub)
            ax4.imshow(resid, cmap='RdBu_r', origin='lower',
                      vmin=-3*std, vmax=3*std)
            ax4.set_title('PSF Fit Residual\n(should be noise-like)')
            ax4.set_xlabel('X (pixels)')
            ax4.set_ylabel('Y (pixels)')
        except Exception:
            ax4.text(0.5, 0.5, 'Residual not available',
                   ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_tuning_check_photutils.png", dpi=150)
    print(f"[Tuning] Saved: {OUTPUT_DIR}/01_tuning_check_photutils.png")
    plt.show()  # Show plot interactively

    # Exit early if tuning-only mode
    if TUNING_ONLY:
        print(f"\n{'=' * 80}")
        print("TUNING MODE - Batch processing skipped")
        print(f"{'=' * 80}")
        print(f"\nAdjust parameters and re-run until satisfied, then set TUNING_ONLY = False")
        print(f"\nCurrent parameters:")
        print(f"  FWHM: {FWHM} pixels")
        print(f"  DETECTION_THRESHOLD: {DETECTION_THRESHOLD}σ")
        if ENABLE_PILE_UP_RECOVERY:
            print(f"  GROUPING_SEPARATION: {GROUPING_SEPARATION:.1f} pixels")
            print(f"  FIT_SHAPE: {FIT_SHAPE}")
        print(f"\n{'=' * 80}\n")
        return  # Exit main() early

    # User confirmation (only in full processing mode)
    # Disabled for non-interactive batch processing
    # response = input("\n[Tuning] Parameters look good? (y/n): ")
    # if response.lower() != 'y':
    #     print("[Tuning] Stopping. Adjust parameters and try again.")
    #     sys.exit(0)
    print("\n[Tuning] Proceeding with batch processing...")

    # ========================================================================
    # 3. BATCH PROCESSING
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING")
    print(f"{'=' * 80}")

    all_events = []
    total_deblended = 0

    # Determine number of cores for parallel processing
    if ENABLE_PARALLEL:
        n_cores = NUM_CORES if NUM_CORES is not None else cpu_count()
        print(f"[Batch] Parallel processing ENABLED: using {n_cores} cores")
        print(f"[Batch] Processing {len(frames)} frames with IterativePSFPhotometry...")
        if ENABLE_PILE_UP_RECOVERY:
            print(f"[Batch] Pile-up recovery: ENABLED")
        else:
            print(f"[Batch] Pile-up recovery: DISABLED")

        # Prepare arguments for parallel processing
        # Pass file paths instead of loading all frames into memory
        print(f"[Batch] Preparing file paths for parallel processing...")
        frame_data = [(i, file_list[i], dark_frame, flat_field_normalized)
                      for i in range(len(file_list))]

        # Process frames in parallel
        print(f"[Batch] Starting parallel PSF photometry (loading frames on-demand)...")
        with Pool(processes=n_cores) as pool:
            # Use imap_unordered for progress bar with parallel processing
            results = list(tqdm(
                pool.imap(process_frame_worker, frame_data, chunksize=10),
                total=len(file_list),
                desc="Processing frames (parallel PSF)"
            ))

        # Filter out None results and collect events
        all_events = [result for result in results if result is not None]

        if not all_events:
            print("[ERROR] No events detected in any frame!")
            sys.exit(1)

        df = pd.concat(all_events, ignore_index=True)

        # Count deblended events
        if ENABLE_PILE_UP_RECOVERY and 'group_size' in df.columns:
            total_deblended = (df['group_size'] > 1).sum()

    else:
        # Sequential processing (original code)
        print(f"[Batch] Sequential processing (ENABLE_PARALLEL = False)")
        print(f"[Batch] Processing {len(file_list)} frames with IterativePSFPhotometry...")
        if ENABLE_PILE_UP_RECOVERY:
            print(f"[Batch] Pile-up recovery: ENABLED")

        for i in tqdm(range(len(file_list)), desc="Processing frames (sequential)"):
            result = process_frame_worker((i, file_list[i], dark_frame, flat_field_normalized))
            if result is not None:
                all_events.append(result)

                # Count deblended events
                if ENABLE_PILE_UP_RECOVERY and 'group_size' in result.columns:
                    total_deblended += (result['group_size'] > 1).sum()

        if not all_events:
            print("[ERROR] No events detected in any frame!")
            sys.exit(1)

        df = pd.concat(all_events, ignore_index=True)

    print(f"[Batch] Total detections: {len(df)} events")

    if ENABLE_PILE_UP_RECOVERY and total_deblended > 0:
        print(f"[Batch] Pile-up recovery: {total_deblended} events recovered from pile-up groups")

    # ========================================================================
    # 4. GMD EXTRACTION & NORMALIZATION
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("GMD NORMALIZATION")
    print(f"{'=' * 80}")
    print(f"[GMD] Reading metadata from TIFF files...")

    gmd_values = []
    for filepath in tqdm(file_list, desc="Extracting GMD"):
        gmd = get_tiff_metadata_gmd(filepath)
        gmd_values.append(gmd)

    valid_gmds = [g for g in gmd_values if g is not None]

    if not ENABLE_GMD_WEIGHTING:
        # GMD weighting disabled by user
        print(f"[GMD] Weighting DISABLED (ENABLE_GMD_WEIGHTING = False)")
        print(f"[GMD] All events will have weight = 1.0")
        df['gmd_energy'] = 1.0
        df['weight'] = 1.0
        gmd_enabled = False
    elif len(valid_gmds) == 0:
        print(f"[GMD] ⚠️ WARNING: No GMD data found!")
        df['gmd_energy'] = 1.0
        df['weight'] = 1.0
        gmd_enabled = False
    else:
        print(f"[GMD] ✓ Found GMD in {len(valid_gmds)}/{len(file_list)} files")

        # Force float dtype to handle None values properly
        gmd_series = pd.Series(gmd_values, dtype=float)
        mean_gmd = gmd_series.mean()  # Ignores NaN automatically
        gmd_series = gmd_series.fillna(mean_gmd)

        df['gmd_energy'] = df['frame'].apply(lambda f: gmd_series.iloc[int(f)])
        df['weight'] = mean_gmd / df['gmd_energy']

        print(f"[GMD] Mean: {mean_gmd:.3e}, Std: {gmd_series.std():.3e}")
        gmd_enabled = True

    # ========================================================================
    # 5. FILTERING
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("FILTERING")
    print(f"{'=' * 80}")

    initial_count = len(df)

    # Filter 1: Saturation
    if 'signal' in df.columns:
        n_before = len(df)
        df = df[df['signal'] < SATURATION_LEVEL]
        n_removed = n_before - len(df)
        if n_before > 0:
            print(f"[Filter] Saturated events: removed {n_removed} "
                  f"({100*n_removed/n_before:.1f}%)")
        else:
            print(f"[Filter] Saturated events: no events to filter")

    # Filter 2: Eccentricity (using converted metric)
    if 'ecc' in df.columns:
        n_before = len(df)
        df = df[df['ecc'] < ECC_CUTOFF]
        n_removed = n_before - len(df)
        if n_before > 0:
            print(f"[Filter] High eccentricity: removed {n_removed} "
                  f"({100*n_removed/n_before:.1f}%)")
        else:
            print(f"[Filter] High eccentricity: no events to filter")

    if initial_count > 0:
        print(f"[Filter] Final events: {len(df)} ({100*len(df)/initial_count:.1f}% retained)")
    else:
        print(f"[Filter] Final events: {len(df)} (no initial events)")

    # Check if all events were filtered out
    if len(df) == 0:
        print(f"[ERROR] All events removed by filters!")
        print(f"[ERROR] Check SATURATION_LEVEL and ECC_CUTOFF settings.")
        sys.exit(1)

    # ========================================================================
    # CIRCULAR MASK (MCP Active Area)
    # ========================================================================
    mask_radius_used = None
    mask_center_used = None

    if ENABLE_CIRCULAR_MASK:
        print(f"\n{'=' * 80}")
        print("CIRCULAR MASK")
        print(f"{'=' * 80}")

        # Determine mask center
        if MASK_CENTER_X is None or MASK_CENTER_Y is None:
            mask_center_x = img_shape[1] / 2.0
            mask_center_y = img_shape[0] / 2.0
            print(f"[Mask] Using geometric center: ({mask_center_x:.1f}, {mask_center_y:.1f})")
        else:
            mask_center_x = MASK_CENTER_X
            mask_center_y = MASK_CENTER_Y
            print(f"[Mask] Using user-specified center: ({mask_center_x:.1f}, {mask_center_y:.1f})")

        mask_center_used = (mask_center_x, mask_center_y)

        # Validate radius
        if MASK_RADIUS is None:
            print(f"[Mask] ERROR: MASK_RADIUS must be set when ENABLE_CIRCULAR_MASK = True!")
            print(f"[Mask] Skipping circular mask...")
        else:
            mask_radius_used = MASK_RADIUS
            print(f"[Mask] Using radius: {mask_radius_used:.1f} pixels")

            # Calculate radius from mask center
            df['radius_from_mask_center'] = np.sqrt(
                (df['x'] - mask_center_x)**2 +
                (df['y'] - mask_center_y)**2
            )

            # Apply mask
            n_before = len(df)
            df = df[df['radius_from_mask_center'] < mask_radius_used]
            n_removed = n_before - len(df)

            print(f"[Mask] Removed {n_removed} events outside MCP active area "
                  f"({100*n_removed/n_before:.2f}%)")

            if len(df) == 0:
                print(f"[Mask] ERROR: All events removed! Check MASK_RADIUS and MASK_CENTER.")
                sys.exit(1)


    # ========================================================================
    # 6. EVENTS PER FRAME DIAGNOSTIC
    # ========================================================================
    events_per_frame = df.groupby('frame').size()
    median_events = events_per_frame.median()

    print(f"\n[Diagnostic] Events/frame: mean={events_per_frame.mean():.1f}, "
          f"median={median_events:.1f}")

    # ========================================================================
    # 7. VMI RECONSTRUCTION
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("VMI RECONSTRUCTION")
    print(f"{'=' * 80}")

    bins_x = img_shape[1] * HISTOGRAM_RESOLUTION_FACTOR
    bins_y = img_shape[0] * HISTOGRAM_RESOLUTION_FACTOR

    print(f"[VMI] Creating 2D histogram ({bins_x}×{bins_y} bins)...")

    raw_vmi, xedges, yedges = np.histogram2d(
        df['x'].values, df['y'].values,
        bins=(bins_x, bins_y),
        range=[[0, img_shape[1]], [0, img_shape[0]]],
        weights=df['weight'].values
    )

    raw_vmi = raw_vmi.T

    print(f"[VMI] Total weighted counts: {raw_vmi.sum():.1f}")
    print(f"[VMI] Peak pixel: {raw_vmi.max():.1f}")

    # ========================================================================
    # 8. INVERSE ABEL TRANSFORM
    # ========================================================================
    if ENABLE_ABEL:
        print(f"\n[Abel] Performing inverse Abel transform...")

        try:
            import abel
            from abel.tools.center import center_image

            center = center_image(raw_vmi, method='com')
            print(f"[Abel] Center detected: {center}")

            print(f"[Abel] Running BASEX method...")
            recon_vmi = abel.Transform(
                raw_vmi,
                method='basex',
                direction='inverse',
                center=center,
                verbose=False
            ).transform

            print(f"[Abel] ✓ Transform complete")
            abel_available = True

        except ImportError:
            print(f"[Abel] ⚠️ PyAbel not installed")
            recon_vmi = None
            abel_available = False
        except Exception as e:
            print(f"[Abel] ⚠️ Error: {e}")
            recon_vmi = None
            abel_available = False
    else:
        recon_vmi = None
        abel_available = False

    # ========================================================================
    # 9. SAVE RESULTS
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    df.to_hdf(OUTPUT_H5, key='events', mode='w', format='table')
    print(f"[Save] HDF5: {OUTPUT_H5}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Save] CSV: {OUTPUT_CSV}")

    np.save(os.path.join(OUTPUT_DIR, "vmi_2d_raw.npy"), raw_vmi)
    print(f"[Save] 2D histogram: {OUTPUT_DIR}/vmi_2d_raw.npy")

    if abel_available:
        np.save(os.path.join(OUTPUT_DIR, "vmi_3d_abel.npy"), recon_vmi)
        print(f"[Save] Abel inverted: {OUTPUT_DIR}/vmi_3d_abel.npy")

    # ========================================================================
    # 10. FINAL VISUALIZATION
    # ========================================================================
    print(f"\n[Plot] Creating final visualization...")

    fig = plt.figure(figsize=(16, 10))

    if abel_available:
        ax1 = plt.subplot(2, 3, (1, 4))
        ax2 = plt.subplot(2, 3, (2, 5))
    else:
        ax1 = plt.subplot(2, 2, (1, 3))
        ax2 = None

    # Plot 1: Raw VMI
    # Guard against empty histogram (all zeros)
    positive_pixels = raw_vmi[raw_vmi > 0]
    if len(positive_pixels) > 0:
        vmin = max(1, np.percentile(positive_pixels, 1))
        vmax = raw_vmi.max()
    else:
        vmin = 1
        vmax = 1

    im1 = ax1.imshow(raw_vmi, cmap='hot',
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    origin='lower', aspect='equal')
    ax1.set_title(f'Raw VMI (2D Projection)\n'
                 f'{len(df)} events, photutils DAOStarFinder')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Counts')

    # Overlay circular mask if used
    if mask_radius_used is not None and mask_center_used is not None:
        circle = plt.Circle(mask_center_used, mask_radius_used,
                           color='cyan', fill=False, linewidth=2,
                           linestyle='--', label='MCP active area')
        ax1.add_patch(circle)
        ax1.legend(loc='upper right', fontsize=9)

    # Plot 2: Abel inverted
    if abel_available and ax2 is not None:
        # Guard against empty histogram (all zeros)
        positive_abel = recon_vmi[recon_vmi > 0]
        if len(positive_abel) > 0:
            vmin_abel = max(1, np.percentile(positive_abel, 1))
            vmax_abel = np.max(recon_vmi)
        else:
            vmin_abel = 1
            vmax_abel = 1

        im2 = ax2.imshow(recon_vmi, cmap='hot',
                        norm=LogNorm(vmin=vmin_abel, vmax=vmax_abel),
                        origin='lower', aspect='equal')
        ax2.set_title('Inverse Abel Transform\n(3D Momentum Slice)')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Counts')

        # Overlay circular mask if used
        if mask_radius_used is not None and mask_center_used is not None:
            circle = plt.Circle(mask_center_used, mask_radius_used,
                               color='cyan', fill=False, linewidth=2,
                               linestyle='--', label='MCP active area')
            ax2.add_patch(circle)
            ax2.legend(loc='upper right', fontsize=9)

        # Radial profiles
        ax3 = plt.subplot(2, 3, 3)
        try:
            radial_2d = abel.tools.vmi.angular_integration(raw_vmi, origin=center)
            radial_3d = abel.tools.vmi.angular_integration(recon_vmi, origin=center)
            r = np.arange(len(radial_2d))
            ax3.plot(r, radial_2d, label='2D Projection', alpha=0.7, linewidth=2)
            ax3.plot(r, radial_3d, label='Abel Inverted', alpha=0.7, linewidth=2)
            ax3.set_xlabel('Radius (pixels)')
            ax3.set_ylabel('Counts')
            ax3.set_title('Radial Profiles')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except:
            pass

        ax4 = plt.subplot(2, 3, 6)
    else:
        ax4 = plt.subplot(2, 2, 2)

    # Events per frame
    ax4.plot(events_per_frame.index, events_per_frame.values, alpha=0.7)
    ax4.axhline(median_events, color='r', linestyle='--')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Events')
    ax4.set_title('Events per Frame')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_final_summary_photutils.png", dpi=200)
    print(f"[Plot] Saved: {OUTPUT_DIR}/02_final_summary_photutils.png")
    # plt.show()  # Disabled for non-interactive batch processing

    # ========================================================================
    # 11. FINAL SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")

    if ENABLE_PILE_UP_RECOVERY:
        print(f"\nMethod: photutils.IterativePSFPhotometry + SourceGrouper")
        print(f"        (Simultaneous PSF fitting for pile-up recovery)")
    else:
        print(f"\nMethod: photutils.DAOStarFinder (astronomy DAOPHOT algorithm)")

    print(f"\nInput:")
    print(f"  Frames: {len(frames)}")
    print(f"  Resolution: {img_shape[1]}×{img_shape[0]}")

    print(f"\nDetection:")
    print(f"  Initial: {initial_count} events")
    print(f"  Final: {len(df)} events")
    print(f"  Events/frame: {len(df)/len(frames):.1f} avg")

    if ENABLE_PILE_UP_RECOVERY:
        print(f"\nPile-up Recovery:")
        print(f"  Status: ✓ ENABLED")
        print(f"  Grouping separation: {GROUPING_SEPARATION:.1f} pixels")
        print(f"  Fit shape: {FIT_SHAPE}")
        if total_deblended > 0:
            print(f"  Recovered events: {total_deblended} (from pile-up groups)")
        else:
            print(f"  Recovered events: 0 (no pile-up detected)")

    print(f"\nCentroiding Parameters:")
    print(f"  FWHM: {FWHM} pixels")
    print(f"  Threshold: {DETECTION_THRESHOLD}σ")
    if not ENABLE_PILE_UP_RECOVERY:
        print(f"  Sharpness: [{SHARPLO}, {SHARPHI}]")

    print(f"\nCorrections Applied:")
    if ENABLE_DARK_SUBTRACTION:
        print(f"  Dark frame: ✓ {DARK_FRAME_COUNT} frames median")
    else:
        print(f"  Dark frame: ✗ Disabled (per-frame background estimation)")
    if use_flat_field and flat_field_normalized is not None:
        print(f"  Flat field: ✓ Active")
    else:
        print(f"  Flat field: ✗ Disabled")
    if mask_radius_used is not None:
        print(f"  Circular mask: ✓ Radius={mask_radius_used:.1f} px, "
              f"Center=({mask_center_used[0]:.1f}, {mask_center_used[1]:.1f})")
    else:
        print(f"  Circular mask: ✗ Disabled")

    if gmd_enabled:
        print(f"\nGMD:")
        print(f"  Normalization: ✓ Active")

    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_H5}")
    print(f"  Images: {OUTPUT_DIR}/")

    print(f"\n{'=' * 80}")
    if ENABLE_PILE_UP_RECOVERY:
        print("✓ ANALYSIS COMPLETE (photutils + pile-up recovery)")
    elif abel_available:
        print("✓ ANALYSIS COMPLETE (photutils version)")
    else:
        print("✓ ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    main()
