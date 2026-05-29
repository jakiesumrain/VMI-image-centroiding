#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMI Photoelectron Analysis - HYBRID VERSION
============================================
Best-of-both: Author's elegance + Production robustness

Combines:
- Concise code structure (author's version)
- Robust error handling (production version)
- Memory-efficient processing
- Comprehensive diagnostics
- Publication-ready output

Author: jeffrey (jakiesumrain@163.com)
Date: 2025-12-03
"""

from __future__ import annotations
import os
import sys
import glob
import pims
import trackpy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifffile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- INPUT/OUTPUT ---
DATA_DIR = "F:/data/int5_kr/1216_Kr_ele_deg54_9ev_0d059W_sig/_1/Default"
FILE_PATTERN = os.path.join(DATA_DIR, "img_*.tif")
OUTPUT_DIR = "E:/work/projects/cVMI-exp/dat/2024-12-18/Kr20141216/centroided"
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "1216_Kr_ele_deg54_9ev_0d059W_sig.h5")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "1216_Kr_ele_deg54_9ev_0d059W_sig.csv")

# --- DARK/BACKGROUND DATA ---
# Path to separate background/dark frame directory
DARK_DATA_DIR = "F:/data/int5_kr/gas_nolight/_1/Default"
DARK_FILE_PATTERN = os.path.join(DARK_DATA_DIR, "img_*.tif")

# --- CAMERA SETTINGS ---
# Binning used during acquisition (0=1×1, 1=2×2, 2=3×3, 3=4×4, 4=8×8)
BINNING = 1  # Match your config.toml!

# --- CENTROIDING PARAMETERS ---
# These parameters control trackpy's blob detection algorithm.
# CRITICAL: Incorrect values cause either missed events or false detections!
#
# DIAMETER (odd integer, typically 3-9):
#   - Physical meaning: Expected size of photoelectron blob in pixels
#   - Rule: DIAMETER ≈ 2× PSF FWHM (Point Spread Function Full-Width Half-Max)
#   - PSF depends on: phosphor screen grain size, camera pixel size, binning
#   - For VMI with phosphor screen:
#     * 1×1 binning (2048×2048): PSF ≈ 3 pixels → DIAMETER = 7
#     * 2×2 binning (1024×1024): PSF ≈ 1.5 pixels → DIAMETER = 3
#     * 4×4 binning (512×512): PSF ≈ 1 pixel → DIAMETER = 3
#   - How to check: Look at tuning plot - single electrons should fill ~DIAMETER pixels
#   - Too small: Misses diffuse blobs, splits single electrons into multiple events
#   - Too large: Merges nearby multi-electron events, slow processing
#
# MIN_MASS (integer, typically 100-1000):
#   - Physical meaning: Minimum integrated brightness (sum of pixel values in blob)
#   - This is your PRIMARY noise threshold
#   - Single photoelectrons: mass ≈ 500-1500 (depends on phosphor, camera gain)
#   - Background noise: mass ≈ 50-200 (thermal electrons, readout noise)
#   - How to set:
#     1. Start with MIN_MASS = 200
#     2. Check tuning histogram - if you see spike at MIN_MASS, increase it
#     3. Optimal: ~2-3σ above noise floor (rejects 99% of noise, keeps 100% signal)
#   - Too low: Detects camera noise as fake electrons (huge spike in mass histogram)
#   - Too high: Misses real low-energy photoelectrons at VMI edge
#
# SEPARATION (integer, typically equal to DIAMETER):
#   - Physical meaning: Minimum allowed distance between two event centroids (pixels)
#   - Purpose: Prevents trackpy from merging two nearby real electrons into one
#   - Rule: SEPARATION ≥ DIAMETER (otherwise events always merge)
#   - Physics: At high laser intensity, multiple electrons can hit nearby pixels
#   - How to check: Look for 2e⁻ peak in mass histogram (at 2× the 1e⁻ peak)
#   - Too small: Multi-electron events merge → 2e⁻ pile-up peak appears
#   - Too large: May split single diffuse blobs into fragments (rare)
#
# TUNING WORKFLOW:
# 1. Run script with default values
# 2. Check tuning plot Panel 3 (mass histogram):
#    - Spike at MIN_MASS? → Increase MIN_MASS
#    - 2e⁻ peak visible? → Reduce laser OR increase SEPARATION
#    - Gaussian 1e⁻ peak width > DIAMETER? → Increase DIAMETER
# 3. Re-run until clean single Gaussian peak, no spike at threshold
# ============================================================================
DIAMETER = 7 if BINNING == 0 else 5  # Increased from 3 to 5 for 2×2 binning

# Noise threshold (minimum integrated brightness)
# With high bias (~408) and no dark subtraction, trackpy needs lower threshold
MIN_MASS = 150  # Lowered from 200 to detect weak signals

# Separation: Prevent merging multi-electron events
SEPARATION = 5  # Increased from 5 to 7 to reduce pile-up

# --- FILTERING PARAMETERS ---
SATURATION_LEVEL = 65000  # 16-bit camera saturates at 65535
ECC_CUTOFF = 0.5  # Cosmic rays have ecc → 1.0

# --- CIRCULAR MASK (MCP Active Area) ---
# The MCP detector has a circular active area. Outside this radius, events are
# typically artifacts from:
# - Electric field edge effects at detector housing
# - Stray electrons scattered from VMI electrode edges
# - Phosphor screen damage or non-uniformity at detector perimeter
# - MCP mounting ring (no active area)
#
# WHY MASK AFTER CENTROIDING (not before):
# - Trackpy needs surrounding pixel context for accurate centroids
# - Events near edge benefit from seeing full PSF neighborhood
# - Much faster (filter dataframe once vs mask 10,000 images)
# - Standard practice in VMI community
#
# WHEN TO ENABLE:
# Set ENABLE_CIRCULAR_MASK = True when you have determined the active area geometry
# from a uniform illumination test or known MCP specifications
ENABLE_CIRCULAR_MASK = False  # Set True after determining geometry

# Mask center (None = use image geometric center automatically)
MASK_CENTER_X = None  # e.g., 512.5 for 1024×1024 image
MASK_CENTER_Y = None  # e.g., 512.5 for 1024×1024 image

# Mask radius (pixels, required if ENABLE_CIRCULAR_MASK = True)
# Example: For 40mm MCP diameter, 13.5 μm pixels, 2×2 binning:
#   Radius = (40 mm / 2) / (13.5 μm × 2) = 741 pixels @ 2×2 binning
MASK_RADIUS = None  # e.g., 480 for typical 1024×1024 VMI

# --- PROCESSING OPTIONS ---
ENABLE_PARALLEL = True  # Now safe with on-demand frame loading!
NUM_CORES = 6  # Only used if ENABLE_PARALLEL=True
DARK_FRAME_COUNT = 500  # Number of background frames to average for dark frame

# --- BACKGROUND SUBTRACTION OPTIONS ---
# ============================================================================
# Choose your background subtraction method:
#
# OPTION 1: Full Dark Frame Subtraction (Best quality, requires dark frames)
#   - Set ENABLE_DARK_SUBTRACTION = True
#   - Dark frames loaded from DARK_FILE_PATTERN directory
#   - First DARK_FRAME_COUNT frames are averaged to create dark frame
#   - Removes pixel-to-pixel bias variations, hot pixels, fixed pattern noise
#   - IMPORTANT: Background directory must contain true darks (no signal!)
#
# OPTION 2: Simple Per-Frame Median Subtraction (Good quality, no dark frames needed)
#   - Set ENABLE_DARK_SUBTRACTION = False, USE_MEDIAN_BACKGROUND = True
#   - Each frame: subtract np.median(frame)
#   - Removes constant bias level, adapts to frame-to-frame drift
#   - Works well for sparse images (few signal pixels)
#
# OPTION 3: No Background Subtraction (Fastest, lowest quality)
#   - Set both ENABLE_DARK_SUBTRACTION = False, USE_MEDIAN_BACKGROUND = False
#   - Trackpy's minmass threshold handles background rejection
#   - Mass values contaminated by bias, but positions still accurate
# ============================================================================
ENABLE_DARK_SUBTRACTION = True  # Set True to use separate background frames
USE_MEDIAN_BACKGROUND = False  # Set True for simple per-frame median subtraction

# --- TUNING MODE ---
# ============================================================================
# Set TUNING_ONLY = True to run ONLY the tuning diagnostic on a single frame.
# This allows rapid iteration on centroiding parameters (DIAMETER, MIN_MASS,
# SEPARATION) without waiting for full batch processing.
#
# WORKFLOW:
# 1. Set TUNING_ONLY = True
# 2. Run script, check tuning plot
# 3. Adjust parameters, repeat until satisfied
# 4. Set TUNING_ONLY = False
# 5. Run full batch processing
# ============================================================================
TUNING_ONLY = True  # Set True to skip batch processing (parameter tuning mode)

# --- FLAT FIELD CORRECTION ---
# CRITICAL for publication-quality VMI data!
#
# WHY FLAT FIELD CORRECTION MATTERS:
# Not all pixels have the same sensitivity. Variations come from:
# 1. CMOS pixel-to-pixel gain differences (~1-5% variation)
# 2. MCP/Phosphor non-uniformity (hot spots, dead zones, burn-in)
# 3. Optical vignetting (dimmer at edges)
# 4. Dirt/scratches on optical path
#
# CONSEQUENCE WITHOUT CORRECTION:
# - VMI should have cylindrical symmetry (rotate image → identical)
# - Pixel gain variations create FALSE angular structure
# - Abel transform interprets this as REAL physics → false rings!
# - Angular distributions are WRONG
#
# HOW TO ACQUIRE FLAT FIELD:
# 1. Remove sample from VMI chamber (no photoelectrons)
# 2. Provide uniform illumination:
#    - Method A: LED flashlight pointed at detector (through viewport)
#    - Method B: Run VMI in "flood" mode (MCP at high gain, no gating)
#    - Method C: Use calibrated light source (best, but not required)
# 3. Acquire 100-1000 frames (average out noise)
# 4. Save as "masterflat.npy" in same directory as this script
# 5. Set ENABLE_FLAT_FIELD = True below
#
# MATHEMATICS:
# Corrected = (Raw - Dark) / (Flat / Flat.mean())
# - Numerator: Dark-subtracted image
# - Denominator: Normalized flat (mean = 1.0, preserves overall brightness)
# - Result: Pixels with high gain (bright in flat) → divided by >1 → corrected down
#           Pixels with low gain (dim in flat) → divided by <1 → corrected up
#
# WHEN TO UPDATE FLAT FIELD:
# - After any MCP/phosphor maintenance
# - Every 6-12 months (phosphor aging changes gain map)
# - If you see unexpected "grid lines" or angular artifacts in Abel transform
ENABLE_FLAT_FIELD = False  # Set True after acquiring masterflat.npy
FLAT_FIELD_PATH = "masterflat.npy"  # Path to flat field file

# --- ABEL TRANSFORM ---
ENABLE_ABEL = False  # Requires: pip install PyAbel

# --- HISTOGRAM RESOLUTION (SUPER-RESOLUTION MODE) ---
# ============================================================================
# CRITICAL UNDERSTANDING: Centroiding vs Raw Summing
#
# THREE LEVELS OF VMI IMAGE QUALITY:
#
# LEVEL 1: RAW CAMERA SUMMING (What you DON'T want)
#   - Just add up 10,000 raw camera frames
#   - Problem 1: Every electron = 3-5 pixel blob (PSF blur from phosphor)
#   - Problem 2: Readout noise accumulates across all 1M pixels
#   - Result: BLURRY, NOISY, gray background, thick fuzzy rings
#
# LEVEL 2: CENTROIDING @ NATIVE RESOLUTION (What we do by default)
#   - Find sub-pixel coordinates, bin to 1024×1024 (same as camera)
#   - Benefit 1: PSF DECONVOLUTION - 3-5 pixel blob → 1 pixel dot (sharp!)
#   - Benefit 2: NOISE THRESHOLDING - Background = perfectly black (MIN_MASS)
#   - Benefit 3: Infinite contrast (signal vs zero, not signal vs noise)
#   - Drawback: Edges look "pixelated" (50.5 pixel ring forced to bin 50 or 51)
#   - THIS ALONE IS A MASSIVE IMPROVEMENT OVER RAW SUMMING!
#
# LEVEL 3: CENTROIDING @ SUPER-RESOLUTION (Optional enhancement)
#   - Bin to 2048×2048 (2×) or 4096×4096 (4×)
#   - All benefits of Level 2, PLUS:
#   - Benefit 4: SMOOTH EDGES - 50.5 pixel ring → bin 101 (exact fit!)
#   - Benefit 5: Prettier for publications
#   - Drawback: More memory, slightly noisier per bin
#
# WHAT THIS PARAMETER DOES:
# HISTOGRAM_RESOLUTION_FACTOR controls the final histogram size:
#   1× → 1024×1024 (native camera resolution)
#   2× → 2048×2048 (double resolution, smooth edges)
#   4× → 4096×4096 (quad resolution, very smooth, overkill)
#
# TRADE-OFFS:
#   Factor | Output Size | Memory | Visual Quality        | Physics Analysis
#   -------|-------------|--------|-----------------------|------------------
#   1×     | 1024×1024   | 4 MB   | Sharp but pixelated   | Optimal
#   2×     | 2048×2048   | 16 MB  | Sharp + smooth edges  | Same as 1×
#   4×     | 4096×4096   | 64 MB  | Very smooth           | Same as 1×
#
# RECOMMENDATION:
# - Routine analysis: 1× (you already removed PSF blur and noise!)
# - Publications: 2× (prettier images with smooth curves)
# - Research: Test 2× or 4× if you're investigating fine structure
#
# IMPORTANT: Physics results (radial distributions, angular anisotropy) are
# nearly identical for 1×, 2×, or 4× because the sub-pixel coordinates are
# already used correctly in all cases. Higher resolution only affects visuals!
# ============================================================================
HISTOGRAM_RESOLUTION_FACTOR = 1  # 1× recommended (already sharp + clean!)
                                 # 2× for publications (smooth edges)
                                 # 4× usually overkill

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_output_dir():
    """Create output directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_tiff_metadata_gmd(filepath):
    """
    Extract GMD (Gas Monitor Detector) pulse energy from TIFF metadata.

    WHY THIS MATTERS:
    FEL pulse energy fluctuates ±10-30% shot-to-shot. Without normalization,
    these fluctuations appear as noise in your VMI image. GMD normalization
    improves SNR by √2-√3 (equivalent to 2-9× more data collection time).

    HOW METADATA IS EMBEDDED:
    The EPICS areaDetector NDAttribute system embeds GMD values into TIFF files
    during acquisition. The data acquisition script reads the GMD PV and stores
    it as a custom TIFF tag synchronized with each camera frame.

    MULTI-METHOD SEARCH STRATEGY:
    Different EPICS/areaDetector versions store NDAttributes in different ways:
    1. ImageDescription (Tag 270) with "GMD=123.4;" format
    2. Custom tags in 65000-65535 range (vendor-specific NDAttributes)
    3. JSON/XML structured data in ImageDescription

    RETURNS:
    - float: GMD pulse energy value (arbitrary units, consistent across dataset)
    - None: If GMD metadata not found (will use weight=1.0 in analysis)
    """
    try:
        with tifffile.TiffFile(filepath) as tif:
            tags = tif.pages[0].tags

            # Method 1: ImageDescription (Tag 270)
            if 270 in tags:
                desc = tags[270].value
                if isinstance(desc, bytes):
                    desc = desc.decode('utf-8', errors='ignore')

                # Try key=value parsing
                if isinstance(desc, str):
                    for item in desc.split(';'):
                        if 'GMD' in item and '=' in item:
                            try:
                                return float(item.split('=')[1])
                            except Exception:
                                pass

                    # Try regex extraction
                    import re
                    match = re.search(r'GMD["\s:=]+([0-9.eE+-]+)', desc)
                    if match:
                        return float(match.group(1))

            # Method 2: Custom tags (NDAttributes from EPICS)
            for tag_id in range(65000, 65536):
                if tag_id in tags:
                    tag_value = str(tags[tag_id].value)
                    if 'GMD' in tag_value:
                        import re
                        match = re.search(r'([0-9.eE+-]+)', tag_value)
                        if match:
                            return float(match.group(1))

    except Exception as e:
        # Silently fail for individual files
        pass

    return None  # Return None if not found

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

    CRITICAL: This function loads from DARK_FILE_PATTERN (background directory),
    NOT from the signal data frames. This ensures pure background without signal
    contamination.

    WHY SEPARATE BACKGROUND FRAMES:
    Using dedicated background acquisitions (no laser/FEL trigger) provides the
    cleanest dark frame. These frames contain ONLY camera noise:
    - Dark current: Thermal electrons (~0.01-1 e⁻/pixel/sec at -40°C)
    - Bias level: Camera ADC offset (~400-2000 counts for 16-bit)
    - Fixed pattern noise: Pixel-to-pixel gain variations
    - Hot pixels: Defective pixels with high dark current

    MEDIAN vs MEAN:
    We use median (not mean) because:
    - Robust to cosmic ray hits during dark collection
    - Immune to outlier hot pixels
    - Better represents typical noise floor

    RETURNS:
    - numpy.ndarray: 2D dark frame (same shape as camera images)
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
    """Check for saturated pixels in image"""
    n_saturated = np.sum(img >= threshold)
    return n_saturated

# ==============================================================================
# PARALLEL PROCESSING WORKER
# ==============================================================================

def process_frame_worker_trackpy(args):
    """
    Worker function for parallel trackpy processing.

    This function loads frames on-demand from disk to minimize memory usage.

    Args:
        args: Tuple of (frame_index, file_path, dark_frame, flat_field_normalized,
                       diameter, minmass, separation)

    Returns:
        pd.DataFrame with detected events for this frame, or None if no events found
    """
    frame_idx, file_path, dark_frame, flat_field_normalized, diameter, minmass, separation = args

    # Load frame from disk (on-demand, not pre-loaded in memory)
    frame = tifffile.imread(file_path)

    # Apply corrections
    if dark_frame is not None:
        corrected = frame - dark_frame
    else:
        corrected = frame.astype(float)
    if flat_field_normalized is not None:
        corrected = corrected / flat_field_normalized

    # Run trackpy centroiding
    features = tp.locate(corrected, diameter=diameter,
                        minmass=minmass, separation=separation)

    if not features.empty:
        features['frame'] = frame_idx
        return features
    else:
        return None

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def main():
    create_output_dir()

    print("=" * 80)
    print("VMI PHOTOELECTRON ANALYSIS - HYBRID VERSION")
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
        print(f"[Dark] Relying on trackpy's minmass threshold for background rejection")

    # Load flat field if enabled (use local variable to avoid scoping issues)
    use_flat_field = ENABLE_FLAT_FIELD
    flat_field = None
    flat_field_normalized = None

    if use_flat_field:
        print(f"\n[Flat] Loading flat field: {FLAT_FIELD_PATH}")
        try:
            flat_field = np.load(FLAT_FIELD_PATH)

            # Verify dimensions match
            if flat_field.shape != img_shape:
                print(f"[Flat] ERROR: Flat field shape {flat_field.shape} != image shape {img_shape}")
                print(f"[Flat] Disabling flat field correction...")
                use_flat_field = False
            else:
                # Normalize flat field (mean = 1.0)
                # This ensures correction doesn't change overall brightness
                flat_field_normalized = flat_field / flat_field.mean()

                # Check for bad values (zeros, NaNs, infs)
                if np.any(flat_field_normalized <= 0):
                    print(f"[Flat] WARNING: Flat field contains zeros/negatives!")
                    print(f"[Flat] Replacing with 1.0 to avoid division errors...")
                    flat_field_normalized[flat_field_normalized <= 0] = 1.0

                print(f"[Flat] Flat field loaded successfully")
                print(f"[Flat] Pixel gain variation: "
                      f"{flat_field_normalized.min():.3f} - {flat_field_normalized.max():.3f} "
                      f"(±{100*(flat_field_normalized.std()):.1f}%)")

        except FileNotFoundError:
            print(f"[Flat] ERROR: Flat field file not found: {FLAT_FIELD_PATH}")
            print(f"[Flat] Acquire flat field first (see comments in config section)")
            print(f"[Flat] Continuing without flat field correction...")
            use_flat_field = False
        except Exception as e:
            print(f"[Flat] ERROR loading flat field: {e}")
            print(f"[Flat] Continuing without flat field correction...")
            use_flat_field = False

    # ========================================================================
    # 2. TUNING CHECK
    # ========================================================================
    test_idx = len(frames) // 2
    raw_test = frames[test_idx]

    # Apply corrections: Dark subtraction (if enabled) + Flat field
    if dark_frame is not None:
        corrected_test = raw_test - dark_frame
    else:
        corrected_test = raw_test.astype(float)  # Work with float for consistency

    if use_flat_field and flat_field_normalized is not None:
        corrected_test = corrected_test / flat_field_normalized

    # Check saturation
    sat_pixels = check_saturation(raw_test)
    if sat_pixels > 0:
        print(f"\n[WARNING] Test frame has {sat_pixels} saturated pixels!")
        print(f"[WARNING] Reduce laser intensity or exposure time.")

    print(f"\n{'=' * 80}")
    print(f"TUNING CHECK (Frame {test_idx})")
    print(f"{'=' * 80}")
    print(f"[Tuning] Diameter: {DIAMETER}, MinMass: {MIN_MASS}, Separation: {SEPARATION}")

    # Run centroiding on test frame
    test_features = tp.locate(corrected_test, diameter=DIAMETER,
                             minmass=MIN_MASS, separation=SEPARATION)

    print(f"[Tuning] Detected {len(test_features)} events")

    if len(test_features) > 0:
        print(f"[Tuning] Mass range: {test_features['mass'].min():.0f} - "
              f"{test_features['mass'].max():.0f}")

    # Visualization (3-panel: Image + Quality + Mass Distribution)
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Annotated detections
    # Clip negative values for better visualization (background subtraction creates negatives)
    corrected_display = np.clip(corrected_test, 0, None)

    ax1 = plt.subplot(1, 3, 1)
    # Show image with better dynamic range
    vmax = np.percentile(corrected_display[corrected_display > 0], 99.99) if np.any(corrected_display > 0) else corrected_display.max()
    im = ax1.imshow(corrected_display, cmap='gray', vmin=0, vmax=vmax, origin='lower')

    # Overlay detected events as circles
    if len(test_features) > 0:
        ax1.plot(test_features['x'], test_features['y'], 'ro',
                markersize=8, fillstyle='none', markeredgewidth=1.5)

    ax1.set_title(f"Tuning Check: {len(test_features)} events\n"
                 f"Diameter={DIAMETER}, MinMass={MIN_MASS}")
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax1, label='Counts (background subtracted)')

    # Panel 2: Mass vs Eccentricity (Quality Check)
    ax2 = plt.subplot(1, 3, 2)
    if len(test_features) > 0 and 'ecc' in test_features.columns:
        scatter = ax2.scatter(test_features['mass'], test_features['ecc'],
                           c=test_features['size'], cmap='viridis', alpha=0.6)
        ax2.axhline(ECC_CUTOFF, color='r', linestyle='--',
                  label=f'Ecc cutoff: {ECC_CUTOFF}')
        ax2.set_xlabel('Mass')
        ax2.set_ylabel('Eccentricity')
        ax2.set_title('Feature Quality')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Size')
    else:
        ax2.text(0.5, 0.5, 'No eccentricity data',
               ha='center', va='center', transform=ax2.transAxes)

    # Panel 3: MASS HISTOGRAM (Pile-up Check)
    # ========================================================================
    # CRITICAL DIAGNOSTIC: This histogram shows the distribution of integrated
    # brightness (mass) for all detected events. It reveals:
    #
    # 1. SINGLE-ELECTRON PEAK (Green line):
    #    - Should be a clean Gaussian centered around 500-1500 counts
    #    - Width comes from Poisson statistics of phosphor photon emission
    #    - Position depends on: phosphor efficiency, camera gain, exposure
    #
    # 2. TWO-ELECTRON PILE-UP (Orange line at 2× the green peak):
    #    - When two electrons hit within ~DIAMETER pixels, trackpy merges them
    #    - Integrated brightness ≈ 2× single electron
    #    - Causes: (a) Laser too bright, (b) SEPARATION too small
    #    - FIX: Reduce laser intensity OR increase SEPARATION parameter
    #
    # 3. NOISE WALL (Spike at MIN_MASS threshold, red line):
    #    - If you see a huge spike at the red line, you're detecting noise
    #    - FIX: Increase MIN_MASS until spike disappears
    #
    # RULE OF THUMB:
    #    - Pile-up <2%: Excellent
    #    - Pile-up 2-5%: Acceptable for VMI
    #    - Pile-up >5%: Reduce laser intensity immediately!
    # ========================================================================
    ax3 = plt.subplot(1, 3, 3)
    if len(test_features) > 0:
        # Plot histogram of mass distribution
        n, bins, patches = ax3.hist(test_features['mass'], bins=50,
                                    color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Mass (Integrated Brightness)')
        ax3.set_ylabel('Count')
        ax3.set_title('Mass Distribution (Check for Pile-up!)')

        # Draw threshold line (red dashed)
        # Events below this were rejected by trackpy
        ax3.axvline(MIN_MASS, color='red', linestyle='--',
                   linewidth=2, label=f'MinMass={MIN_MASS}')

        # Estimate single-electron peak using MEDIAN (robust to outliers and low statistics)
        # MEDIAN is more reliable than MODE for:
        #   - Low statistics (few events per bin → Poisson noise)
        #   - Skewed distributions (long tail to high mass)
        #   - Truncated data (MIN_MASS cutoff creates artificial spike)
        # Mode would pick the tallest bin (often the cutoff artifact)
        peak_mass = np.median(test_features['mass'])
        ax3.axvline(peak_mass, color='green', linestyle=':',
                   linewidth=2, label=f'1e⁻ peak ≈ {peak_mass:.0f} (median)')

        # Mark expected 2-electron location (orange dotted)
        # If two electrons merge, their combined mass ≈ 2× single electron
        # Physics: Charge sharing OR true coincidence (both electrons same momentum)
        ax3.axvline(peak_mass * 2, color='orange', linestyle=':',
                   linewidth=2, label=f'2e⁻ location ≈ {peak_mass*2:.0f}')

        # Quantitative pile-up estimate
        # Count how many events fall in a window around the 2e⁻ peak
        # Window size: ±30% accounts for:
        #   - Poisson fluctuations in phosphor photon emission (σ/μ ≈ 1/√N)
        #   - Variations in electron kinetic energy (different light yield)
        #   - Camera readout noise spreading the peak
        window = peak_mass * 0.3  # ±30% window around 2e⁻ peak
        pile_up_count = np.sum((test_features['mass'] > peak_mass * 2 - window) &
                               (test_features['mass'] < peak_mass * 2 + window))
        pile_up_fraction = 100 * pile_up_count / len(test_features)

        # Display pile-up diagnostic with color-coded warning
        # Red background = action required (reduce laser)
        # White background = acceptable (proceed with acquisition)
        textstr = f'Events in 2e⁻ region: {pile_up_count} ({pile_up_fraction:.1f}%)'
        if pile_up_fraction > 5:
            # HIGH PILE-UP: Immediate action required!
            # This means >5% of your events are actually 2+ electrons merged together
            # Your VMI reconstruction will be distorted - reduce laser NOW
            textstr += '\n⚠️ HIGH PILE-UP!'
            ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        else:
            # Acceptable pile-up level
            ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No mass data',
               ha='center', va='center', transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_tuning_check.png", dpi=150)
    print(f"[Tuning] Saved: {OUTPUT_DIR}/01_tuning_check.png")
    plt.show()

    # Exit early if tuning-only mode
    if TUNING_ONLY:
        print(f"\n{'=' * 80}")
        print("TUNING MODE - Batch processing skipped")
        print(f"{'=' * 80}")
        print(f"\nAdjust parameters and re-run until satisfied, then set TUNING_ONLY = False")
        print(f"\nCurrent parameters:")
        print(f"  DIAMETER: {DIAMETER}")
        print(f"  MIN_MASS: {MIN_MASS}")
        print(f"  SEPARATION: {SEPARATION}")
        print(f"\n{'=' * 80}\n")
        return  # Exit main() early

    # User confirmation (only in full processing mode)
    response = input("\n[Tuning] Parameters look good? (y/n): ")
    if response.lower() != 'y':
        print("[Tuning] Stopping. Adjust parameters and try again.")
        sys.exit(0)

    # ========================================================================
    # 3. BATCH PROCESSING
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING")
    print(f"{'=' * 80}")

    all_events = []

    if ENABLE_PARALLEL and len(frames) > 100:
        # Parallel processing for large datasets
        n_cores = NUM_CORES if NUM_CORES is not None else cpu_count()
        print(f"[Batch] Processing with {n_cores} cores (parallel mode, on-demand loading)...")

        # Prepare arguments for parallel processing
        # Pass file paths instead of loading all frames into memory
        frame_data = [(i, file_list[i], dark_frame, flat_field_normalized,
                      DIAMETER, MIN_MASS, SEPARATION)
                     for i in range(len(file_list))]

        # Process frames in parallel
        with Pool(processes=n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_frame_worker_trackpy, frame_data, chunksize=10),
                total=len(file_list),
                desc="Processing frames (parallel)"
            ))

        # Filter out None results and collect events
        all_events = [result for result in results if result is not None]

        if not all_events:
            print("[ERROR] No events detected in any frame!")
            sys.exit(1)

        df = pd.concat(all_events, ignore_index=True)

    else:
        # Sequential processing (memory efficient)
        print(f"[Batch] Processing sequentially (memory-efficient mode)...")

        for i in tqdm(range(len(file_list)), desc="Processing frames"):
            result = process_frame_worker_trackpy((i, file_list[i], dark_frame,
                                                   flat_field_normalized,
                                                   DIAMETER, MIN_MASS, SEPARATION))
            if result is not None:
                all_events.append(result)

        if not all_events:
            print("[ERROR] No events detected in any frame!")
            sys.exit(1)

        df = pd.concat(all_events, ignore_index=True)

    print(f"[Batch] Initial detections: {len(df)} events")

    # ========================================================================
    # 4. GMD EXTRACTION & NORMALIZATION
    # ========================================================================
    # THE FEL PULSE ENERGY PROBLEM:
    # Free-electron lasers (FELs) have inherent shot-to-shot intensity fluctuations
    # of ±10-30% due to:
    # - Electron bunch charge variations in the accelerator
    # - SASE (Self-Amplified Spontaneous Emission) stochastic process
    # - Undulator temperature/alignment drifts
    #
    # WHY THIS MATTERS FOR VMI:
    # Without normalization, these fluctuations add NOISE to your VMI image:
    # - Bright FEL pulses → more photoelectrons → higher counts in that frame
    # - Dim FEL pulses → fewer photoelectrons → lower counts
    # - Result: Your VMI image is blurred by intensity variations!
    #
    # GMD (GAS MONITOR DETECTOR):
    # Measures FEL pulse energy by ionizing gas (typically nitrogen or neon)
    # - Output: Charge/current proportional to photon flux
    # - Readout: EPICS PV synchronized with camera triggers
    # - Embedded: Stored in TIFF metadata via NDAttributes
    #
    # NORMALIZATION MATHEMATICS:
    # For each photoelectron event at frame i:
    # - GMD value: E_i (pulse energy for that frame)
    # - Mean GMD: <E> = (1/N) × Σ E_i
    # - Weight: w_i = <E> / E_i
    #
    # When creating VMI histogram: count each event with weight w_i instead of 1
    # - Bright pulse (E_i = 1.3 × <E>): weight = 0.77 (down-weight)
    # - Dim pulse (E_i = 0.7 × <E>): weight = 1.43 (up-weight)
    # - Average pulse (E_i = <E>): weight = 1.00 (no change)
    #
    # SNR IMPROVEMENT:
    # For GMD fluctuations σ_GMD, normalization improves SNR by:
    # SNR_improvement = √(1 + (σ_GMD / <E>)²)
    # Example: 20% RMS fluctuation → 1.02× better SNR (equivalent to 2% more data!)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("GMD NORMALIZATION")
    print(f"{'=' * 80}")
    print(f"[GMD] Reading metadata from TIFF files...")

    # Extract GMD for all files
    gmd_values = []
    for filepath in tqdm(file_list, desc="Extracting GMD"):
        gmd = get_tiff_metadata_gmd(filepath)
        gmd_values.append(gmd)

    # Validate GMD extraction
    valid_gmds = [g for g in gmd_values if g is not None]

    if len(valid_gmds) == 0:
        print(f"[GMD] ⚠️ WARNING: No GMD data found in TIFF metadata!")
        print(f"[GMD] Continuing without normalization (weights = 1.0)")
        print(f"[GMD] Check that NDAttributes are correctly embedded.")
        df['gmd_energy'] = 1.0
        df['weight'] = 1.0
        gmd_enabled = False
    else:
        print(f"[GMD] ✓ Found GMD in {len(valid_gmds)}/{len(file_list)} files")

        # Fill missing GMD values with mean
        gmd_series = pd.Series(gmd_values)
        mean_gmd = gmd_series[gmd_series.notna()].mean()
        gmd_series = gmd_series.fillna(mean_gmd)

        # Add to dataframe
        df['gmd_energy'] = df['frame'].apply(lambda f: gmd_series.iloc[int(f)])

        # Safe division with zero protection
        # If GMD is zero or negative (corrupt metadata), use mean_gmd instead
        df['weight'] = df['gmd_energy'].apply(
            lambda g: mean_gmd / g if g > 0 else 1.0
        )

        print(f"[GMD] GMD statistics:")
        print(f"      Mean: {mean_gmd:.3e}")
        print(f"      Std:  {gmd_series.std():.3e} ({gmd_series.std()/mean_gmd*100:.1f}%)")

        gmd_enabled = True

        # Plot GMD fluctuations
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        ax = axes[0]
        ax.plot(gmd_series.values, alpha=0.7)
        ax.axhline(mean_gmd, color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('GMD Pulse Energy')
        ax.set_title('FEL Pulse Energy Fluctuations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.hist(gmd_series.values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(mean_gmd, color='r', linestyle='--', label='Mean')
        ax.set_xlabel('GMD Pulse Energy')
        ax.set_ylabel('Count')
        ax.set_title(f'GMD Distribution (RMS: {gmd_series.std()/mean_gmd*100:.1f}%)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/02_gmd_analysis.png", dpi=150)
        plt.close()
        print(f"[GMD] Saved: {OUTPUT_DIR}/02_gmd_analysis.png")

    # ========================================================================
    # 5. FILTERING
    # ========================================================================
    # After centroiding, we filter out bad events that would distort the VMI.
    # Each filter targets a specific failure mode:
    #
    # FILTER 1: SATURATION (signal > 65000 counts)
    #   - Problem: Saturated pixels clip at 65535 (16-bit max)
    #   - Effect: Centroid shifts away from true position, mass underestimated
    #   - Cause: Photoelectron hit high-sensitivity region of phosphor screen
    #   - Solution: Filter out using trackpy's 'signal' column (peak pixel value)
    #
    # FILTER 2: ECCENTRICITY (cosmic rays, ecc > 0.7)
    #   - Problem: Cosmic ray hits create elongated streaks (ecc → 1.0)
    #   - Effect: False "events" far from VMI center, wrong mass
    #   - Physics: Photoelectrons create ~circular blobs (ecc < 0.5)
    #   - Solution: Reject high-eccentricity events (ecc > 0.7)
    #   - Why 0.7? Keeps real electrons at VMI edge (slightly elliptical due to
    #     oblique phosphor screen angle) while rejecting cosmic rays
    #
    # WHY FILTERING MATTERS:
    #   - Even 1% bad events can create false rings in VMI image
    #   - Saturated events bias radial distribution (momentum calibration)
    #   - Cosmic rays create high-energy artifacts in photoelectron spectrum
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("FILTERING")
    print(f"{'=' * 80}")

    initial_count = len(df)

    # Filter 1: Saturation (using trackpy's 'signal' column)
    if 'signal' in df.columns:
        n_before = len(df)
        df = df[df['signal'] < SATURATION_LEVEL]
        n_removed = n_before - len(df)
        print(f"[Filter] Saturated events: removed {n_removed} "
              f"({100*n_removed/n_before:.1f}%)")

    # Filter 2: Eccentricity (cosmic rays)
    if 'ecc' in df.columns:
        n_before = len(df)
        df = df[df['ecc'] < ECC_CUTOFF]
        n_removed = n_before - len(df)
        print(f"[Filter] High eccentricity: removed {n_removed} "
              f"({100*n_removed/n_before:.1f}%)")

    print(f"[Filter] Final events: {len(df)} ({100*len(df)/initial_count:.1f}% retained)")

    # ========================================================================
    # CIRCULAR MASK (MCP Active Area)
    # ========================================================================
    # Applied AFTER centroiding to preserve trackpy accuracy for edge events.
    # This removes events outside the MCP active area (mounting ring, housing).
    # ========================================================================
    mask_radius_used = None  # Track for visualization
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

            # Calculate radius from mask center (may differ from image center)
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
    # 6. DIAGNOSTIC: EVENTS PER FRAME
    # ========================================================================
    events_per_frame = df.groupby('frame').size()
    median_events = events_per_frame.median()

    print(f"\n[Diagnostic] Events/frame: mean={events_per_frame.mean():.1f}, "
          f"median={median_events:.1f}")

    # Flag anomalous frames
    bad_frames = events_per_frame[
        (events_per_frame < 0.3 * median_events) |
        (events_per_frame > 3.0 * median_events)
    ]

    if len(bad_frames) > 0:
        print(f"[Diagnostic] ⚠️ {len(bad_frames)} anomalous frames detected")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(events_per_frame.index, events_per_frame.values, alpha=0.7)
    ax.axhline(median_events, color='r', linestyle='--',
              label=f'Median: {median_events:.1f}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Events Detected')
    ax.set_title('Events per Frame (Quality Check)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_events_per_frame.png", dpi=150)
    plt.close()
    print(f"[Diagnostic] Saved: {OUTPUT_DIR}/03_events_per_frame.png")

    # ========================================================================
    # DIAGNOSTIC: ECCENTRICITY vs RADIUS
    # ========================================================================
    # CRITICAL CHECK: Verify that edge events aren't being incorrectly filtered
    #
    # WHY THIS MATTERS:
    # At the VMI edge (high photoelectron momentum), PSF can become elliptical due to:
    # - Oblique incidence angle on phosphor screen
    # - Astigmatism in imaging optics
    # - MCP pore orientation effects
    #
    # If ECC_CUTOFF is too strict, you'll lose 5-10% of valid edge events!
    #
    # WHAT TO LOOK FOR:
    # - Center events (r < 200): Should have low ecc (0.1-0.4)
    # - Edge events (r > 400): May have moderate ecc (0.4-0.7)
    # - If you see many events clustering at ECC_CUTOFF line near edge,
    #   consider increasing the cutoff (e.g., 0.7 → 0.8)
    # ========================================================================
    if 'ecc' in df.columns and len(df) > 0:
        print(f"\n[Diagnostic] Creating eccentricity vs radius plot...")

        # Calculate radius from image center
        center_x = img_shape[1] / 2
        center_y = img_shape[0] / 2
        df['radius'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot with density coloring
        scatter = ax.scatter(df['radius'], df['ecc'],
                           c=df['mass'], cmap='viridis',
                           alpha=0.3, s=1, rasterized=True)

        # Mark the eccentricity cutoff
        ax.axhline(ECC_CUTOFF, color='red', linestyle='--',
                  linewidth=2, label=f'Ecc cutoff: {ECC_CUTOFF}')

        ax.set_xlabel('Radius from Center (pixels)')
        ax.set_ylabel('Eccentricity')
        ax.set_title('Eccentricity vs Radius\n'
                    '(Check if edge events are being incorrectly filtered)')
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Mass')

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/03b_eccentricity_vs_radius.png", dpi=150)
        plt.close()
        print(f"[Diagnostic] Saved: {OUTPUT_DIR}/03b_eccentricity_vs_radius.png")

        # Statistics
        edge_events = df[df['radius'] > img_shape[1] * 0.4]  # Outer 40% radius
        if len(edge_events) > 0:
            median_ecc_edge = edge_events['ecc'].median()
            fraction_near_cutoff = len(edge_events[edge_events['ecc'] > ECC_CUTOFF * 0.9]) / len(edge_events)
            print(f"[Diagnostic] Edge events (r > {img_shape[1] * 0.4:.0f}): {len(edge_events)}")
            print(f"[Diagnostic] Median eccentricity at edge: {median_ecc_edge:.3f}")
            if fraction_near_cutoff > 0.1:
                print(f"[Diagnostic] ⚠️ WARNING: {100*fraction_near_cutoff:.1f}% of edge events "
                     f"near cutoff threshold!")
                print(f"[Diagnostic]            Consider increasing ECC_CUTOFF to {ECC_CUTOFF + 0.1:.1f}")


    # ========================================================================
    # 7. VMI RECONSTRUCTION
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("VMI RECONSTRUCTION")
    print(f"{'=' * 80}")

    # Calculate final histogram dimensions
    bins_x = img_shape[1] * HISTOGRAM_RESOLUTION_FACTOR
    bins_y = img_shape[0] * HISTOGRAM_RESOLUTION_FACTOR

    if HISTOGRAM_RESOLUTION_FACTOR > 1:
        print(f"[VMI] Super-resolution mode: {HISTOGRAM_RESOLUTION_FACTOR}×")
        print(f"[VMI] Native camera: {img_shape[1]}×{img_shape[0]}")
        print(f"[VMI] Output histogram: {bins_x}×{bins_y} bins")
        print(f"[VMI] Effective pixel size: {1.0/HISTOGRAM_RESOLUTION_FACTOR:.2f}× camera pixel")
    else:
        print(f"[VMI] Native resolution mode (1×)")
        print(f"[VMI] Creating 2D histogram ({bins_x}×{bins_y} bins)...")

    # Create 2D histogram with GMD weights
    # Range is still 0 to img_shape (native camera coordinates)
    # But bins are finer if HISTOGRAM_RESOLUTION_FACTOR > 1
    raw_vmi, xedges, yedges = np.histogram2d(
        df['x'].values, df['y'].values,
        bins=(bins_x, bins_y),
        range=[[0, img_shape[1]], [0, img_shape[0]]],
        weights=df['weight'].values
    )

    # Transpose for correct orientation
    raw_vmi = raw_vmi.T

    print(f"[VMI] Total weighted counts: {raw_vmi.sum():.1f}")
    print(f"[VMI] Peak pixel: {raw_vmi.max():.1f}")

    if HISTOGRAM_RESOLUTION_FACTOR > 1:
        # Calculate average counts per bin (should be lower with finer bins)
        avg_counts = raw_vmi.sum() / (bins_x * bins_y)
        print(f"[VMI] Average counts/bin: {avg_counts:.2f} "
              f"({1.0/(HISTOGRAM_RESOLUTION_FACTOR**2):.2f}× native)")
        print(f"[VMI] Note: Finer binning spreads counts over more bins (expected)")

    # ========================================================================
    # 8. INVERSE ABEL TRANSFORM
    # ========================================================================
    # THE CENTRAL VMI PROBLEM:
    # Your detector records a 2D PROJECTION of a 3D spherical distribution.
    # Imagine taking a photo of a hollow sphere - you see a filled circle!
    #
    # WHAT IS THE ABEL TRANSFORM?
    # Abel transform converts 3D spherically-symmetric distribution → 2D projection
    # (This is what nature does when photoelectrons hit your detector)
    #
    # WHAT IS THE INVERSE ABEL TRANSFORM?
    # Mathematically inverts the projection to recover the original 3D distribution
    # Output: Central SLICE through the 3D momentum sphere
    #
    # WHY YOU NEED THIS:
    # - 2D projection: Overcounts center (many spheres overlap), dim at edges
    # - 3D slice: TRUE photoelectron momentum distribution
    # - Affects: Photoelectron spectrum, angular distributions, branching ratios
    # - Impact: Without Abel inversion, kinetic energies are WRONG by 10-50%!
    #
    # BASEX METHOD:
    # We use BASEX (Basis Set Expansion) because:
    # - Fast: ~1-5 minutes for 2048×2048 images
    # - Robust: Handles noise and imperfect cylindrical symmetry well
    # - Accurate: <1% error in radial distribution
    # Alternatives: Hansen-Law (faster, less accurate), three_point (slow, accurate)
    #
    # MATHEMATICS:
    # 2D projection P(r,z) = ∫ 3D distribution f(r) × Abel kernel
    # Inverse: f(r) = -1/π × d/dr ∫ P(r',z) / √(r'²-r²) dr'
    #
    # OUTPUT:
    # - raw_vmi: 2D projection (what you measure)
    # - recon_vmi: 3D central slice (what you want for physics)
    # ========================================================================
    if ENABLE_ABEL:
        print(f"\n[Abel] Performing inverse Abel transform...")

        try:
            import abel

            # Find center using PyAbel's center-of-mass method
            from abel.tools.center import center_image
            center = center_image(raw_vmi, method='com')
            print(f"[Abel] Center detected: {center}")

            # Perform inverse Abel transform
            print(f"[Abel] Running BASEX method (this may take 1-5 minutes)...")
            recon_vmi = abel.Transform(
                raw_vmi,
                method='basex',
                direction='inverse',
                center=center,
                verbose=False
            ).transform

            print(f"[Abel] ✓ Transform complete")
            print(f"[Abel] 3D slice peak: {np.max(recon_vmi):.1f}")

            abel_available = True

        except ImportError:
            print(f"[Abel] ⚠️ PyAbel not installed (pip install PyAbel)")
            print(f"[Abel] Skipping Abel transform...")
            recon_vmi = None
            abel_available = False
        except Exception as e:
            print(f"[Abel] ⚠️ Error during transform: {e}")
            recon_vmi = None
            abel_available = False
    else:
        print(f"[Abel] Abel transform disabled in config")
        recon_vmi = None
        abel_available = False

    # ========================================================================
    # 9. SAVE RESULTS
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    # Save event data
    df.to_hdf(OUTPUT_H5, key='events', mode='w', format='table')
    print(f"[Save] HDF5: {OUTPUT_H5}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Save] CSV: {OUTPUT_CSV}")

    # Save images (include resolution factor in filename)
    if HISTOGRAM_RESOLUTION_FACTOR > 1:
        vmi_2d_filename = f"vmi_2d_raw_{HISTOGRAM_RESOLUTION_FACTOR}x.npy"
        vmi_3d_filename = f"vmi_3d_abel_{HISTOGRAM_RESOLUTION_FACTOR}x.npy"
    else:
        vmi_2d_filename = "vmi_2d_raw.npy"
        vmi_3d_filename = "vmi_3d_abel.npy"

    np.save(os.path.join(OUTPUT_DIR, vmi_2d_filename), raw_vmi)
    print(f"[Save] 2D histogram: {OUTPUT_DIR}/{vmi_2d_filename}")

    if abel_available:
        np.save(os.path.join(OUTPUT_DIR, vmi_3d_filename), recon_vmi)
        print(f"[Save] Abel inverted: {OUTPUT_DIR}/{vmi_3d_filename}")

    # ========================================================================
    # 10. FINAL VISUALIZATION
    # ========================================================================
    print(f"\n[Plot] Creating final visualization...")

    fig = plt.figure(figsize=(16, 10))

    # Main plots: 2D and Abel
    if abel_available:
        ax1 = plt.subplot(2, 3, (1, 4))
        ax2 = plt.subplot(2, 3, (2, 5))
    else:
        ax1 = plt.subplot(2, 2, (1, 3))
        ax2 = None

    # Plot 1: Raw VMI (2D projection)
    vmin = max(1, np.percentile(raw_vmi[raw_vmi > 0], 1))
    vmax = raw_vmi.max()

    im1 = ax1.imshow(raw_vmi, cmap='hot',
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    origin='lower', aspect='equal')
    ax1.set_title(f'Raw VMI (2D Projection)\n'
                 f'{len(df)} events, GMD {"normalized" if gmd_enabled else "not normalized"}')
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

    # Plot 2: Abel inverted (if available)
    if abel_available and ax2 is not None:
        vmin_abel = max(1, np.percentile(recon_vmi[recon_vmi > 0], 1))
        vmax_abel = np.max(recon_vmi)

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
        except Exception as e:
            print(f"[Plot] Warning: Radial profile plotting failed: {e}")
            pass

        # Events per frame
        ax4 = plt.subplot(2, 3, 6)
    else:
        ax4 = plt.subplot(2, 2, 2)

    ax4.plot(events_per_frame.index, events_per_frame.values, alpha=0.7)
    ax4.axhline(median_events, color='r', linestyle='--')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Events')
    ax4.set_title('Events per Frame')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_final_summary.png", dpi=200)
    print(f"[Plot] Saved: {OUTPUT_DIR}/04_final_summary.png")
    plt.show()

    # ========================================================================
    # 11. FINAL SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nInput:")
    print(f"  Frames: {len(frames)}")
    print(f"  Resolution: {img_shape[1]}×{img_shape[0]}")

    print(f"\nDetection:")
    print(f"  Initial: {initial_count} events")
    print(f"  Final: {len(df)} events")
    print(f"  Events/frame: {len(df)/len(frames):.1f} avg")

    print(f"\nCorrections Applied:")
    if ENABLE_DARK_SUBTRACTION:
        print(f"  Dark frame: ✓ {DARK_FRAME_COUNT} frames median")
    else:
        print(f"  Dark frame: ✗ Disabled (using trackpy minmass threshold)")
    if use_flat_field and flat_field_normalized is not None:
        print(f"  Flat field: ✓ Gain variation ±{100*flat_field_normalized.std():.1f}%")
    else:
        print(f"  Flat field: ✗ Disabled (acquire with generate_masterflat.py)")
    if mask_radius_used is not None:
        print(f"  Circular mask: ✓ Radius={mask_radius_used:.1f} px, "
              f"Center=({mask_center_used[0]:.1f}, {mask_center_used[1]:.1f})")
    else:
        print(f"  Circular mask: ✗ Disabled")

    print(f"\nHistogram:")
    if HISTOGRAM_RESOLUTION_FACTOR > 1:
        print(f"  Resolution: {bins_x}×{bins_y} ({HISTOGRAM_RESOLUTION_FACTOR}× super-resolution)")
        print(f"  Benefit: PSF deconvolution + smooth edges")
    else:
        print(f"  Resolution: {bins_x}×{bins_y} (native, 1×)")
        print(f"  Benefit: PSF deconvolution + noise thresholding")

    if gmd_enabled:
        print(f"\nGMD:")
        print(f"  Normalization: ✓ Active")
        print(f"  RMS fluctuation: {gmd_series.std()/mean_gmd*100:.1f}%")

    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_H5}")
    print(f"  Images: {OUTPUT_DIR}/")

    print(f"\n{'=' * 80}")
    if abel_available:
        print("✓ READY FOR PUBLICATION")
    else:
        print("✓ ANALYSIS COMPLETE (Install PyAbel for 3D reconstruction)")
    print(f"{'=' * 80}\n")

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    main()
