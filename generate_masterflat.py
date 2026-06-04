#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Master Flat Field for VMI Detector Calibration
========================================================
This script acquires and processes flat field calibration data for VMI experiments.

WHAT IS A FLAT FIELD?
A flat field is a reference image showing the pixel-to-pixel gain variations
of your detector system (camera + MCP + phosphor screen + optics).

WHY YOU NEED THIS:
Without flat field correction, your VMI image will have false angular structure
from detector non-uniformity. This shows up as "grid lines" or false "rings" in
your Abel-transformed momentum distribution.

BEFORE RUNNING THIS SCRIPT:
1. Remove any sample from the VMI chamber (no photoelectrons!)
2. Provide uniform illumination - choose ONE method:

   METHOD A - LED Flashlight (Simplest):
   - Point LED flashlight at detector viewport
   - Should see uniform glow on camera preview
   - Adjust distance/angle to minimize vignetting

   METHOD B - VMI Flood Mode (Best):
   - Set MCP voltage to operating level
   - Remove gate/trigger (continuous mode)
   - Add weak UV/IR LED inside chamber for uniform illumination
   - Should see ~5000-15000 counts per pixel

   METHOD C - Calibrated Light Source (Most accurate):
   - Use integrating sphere or calibrated LED array
   - Most expensive but gives best results

3. Adjust camera settings:
   - Exposure time: Long enough to get ~10000-20000 counts/pixel
   - Binning: Match the binning you'll use for VMI experiments!
   - Gain: Same as VMI acquisition

PROCEDURE:
This script will:
1. Acquire NUM_FRAMES images (100-1000 frames)
2. Calculate median (robust to outliers)
3. Save as "masterflat.npy" in output directory
4. Display diagnostic plots showing gain map

After running, set ENABLE_FLAT_FIELD=True in process_andor_vmi_HYBRID.py

Author: VMI Analysis Pipeline
Date: 2025-12-03
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- INPUT DATA ---
# Option 1: Use TIFF files from Andor acquisition
FLAT_FIELD_DIR = "/data/calibration/flatfield/"
FILE_PATTERN = os.path.join(FLAT_FIELD_DIR, "flat_*.tiff")

# Option 2: Use live acquisition (requires EPICS setup)
USE_LIVE_ACQUISITION = False  # Set True to acquire new data
NUM_FRAMES = 100  # Number of frames to acquire (100-1000)

# --- OUTPUT ---
OUTPUT_FILE = "masterflat.npy"
OUTPUT_DIR = "calibration"

# --- QUALITY CHECKS ---
MIN_COUNTS_PER_PIXEL = 1000  # Minimum to avoid Poisson noise dominating
MAX_COUNTS_PER_PIXEL = 60000  # Maximum to avoid saturation

# ==============================================================================
# LOAD OR ACQUIRE DATA
# ==============================================================================

print("=" * 80)
print("VMI FLAT FIELD GENERATION")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_LIVE_ACQUISITION:
    print("\n[ERROR] Live acquisition not yet implemented!")
    print("[ERROR] Please acquire flat field images manually and use FILE_PATTERN mode")
    sys.exit(1)
else:
    # Load from TIFF files
    print(f"\n[Load] Searching: {FILE_PATTERN}")
    file_list = sorted(glob.glob(FILE_PATTERN))

    if len(file_list) == 0:
        print(f"[ERROR] No files found matching pattern!")
        print(f"[ERROR] Expected location: {FILE_PATTERN}")
        print(f"\n[HELP] Acquisition procedure:")
        print(f"  1. Remove sample from VMI chamber")
        print(f"  2. Add uniform light source (LED flashlight or flood mode)")
        print(f"  3. Acquire 100-1000 frames using Andor acquisition script")
        print(f"  4. Save to {FLAT_FIELD_DIR}")
        print(f"  5. Run this script again")
        sys.exit(1)

    print(f"[Load] Found {len(file_list)} frames")

    # Load first frame to get dimensions
    import tifffile
    test_img = tifffile.imread(file_list[0])
    img_shape = test_img.shape
    print(f"[Load] Image size: {img_shape[1]}×{img_shape[0]} pixels")

    # Load all frames
    print(f"\n[Load] Loading frames for median calculation...")
    stack = np.zeros((len(file_list), img_shape[0], img_shape[1]), dtype=np.float32)

    for i, filepath in tqdm(enumerate(file_list), total=len(file_list),
                            desc="Loading frames"):
        stack[i] = tifffile.imread(filepath).astype(np.float32)

# ==============================================================================
# QUALITY CHECKS
# ==============================================================================

print(f"\n[Quality] Checking flat field data quality...")

# Check brightness levels
mean_counts = stack.mean()
min_counts = stack.min()
max_counts = stack.max()

print(f"[Quality] Brightness: min={min_counts:.0f}, mean={mean_counts:.0f}, max={max_counts:.0f}")

if mean_counts < MIN_COUNTS_PER_PIXEL:
    print(f"[WARNING] Mean counts ({mean_counts:.0f}) < {MIN_COUNTS_PER_PIXEL}")
    print(f"[WARNING] Low signal → Poisson noise will dominate")
    print(f"[WARNING] Recommendation: Increase exposure time or light intensity")

if max_counts > MAX_COUNTS_PER_PIXEL:
    print(f"[WARNING] Max counts ({max_counts:.0f}) > {MAX_COUNTS_PER_PIXEL}")
    print(f"[WARNING] Saturation detected! Flat field will be incorrect")
    print(f"[WARNING] Recommendation: Reduce exposure time or light intensity")
    print(f"[ERROR] Cannot proceed with saturated data. Please re-acquire.")
    sys.exit(1)

# Check uniformity (should be < 20% variation for good flat field)
uniformity = 100 * stack.std(axis=0).mean() / mean_counts
print(f"[Quality] Frame-to-frame variation: {uniformity:.1f}% (should be <5%)")

if uniformity > 10:
    print(f"[WARNING] High frame-to-frame variation!")
    print(f"[WARNING] Light source may be unstable (flickering?)")
    print(f"[WARNING] Recommendation: Use more frames or stabilize light source")

# ==============================================================================
# CALCULATE MASTER FLAT
# ==============================================================================

print(f"\n[Flat] Calculating master flat (median of {len(stack)} frames)...")

# Use median (robust to outliers like cosmic rays)
masterflat = np.median(stack, axis=0)

print(f"[Flat] Master flat calculated")
print(f"[Flat] Mean: {masterflat.mean():.0f}")
print(f"[Flat] Std: {masterflat.std():.0f} ({100*masterflat.std()/masterflat.mean():.1f}%)")
print(f"[Flat] Range: {masterflat.min():.0f} - {masterflat.max():.0f}")

# Calculate normalized flat (what will be used for correction)
normalized_flat = masterflat / masterflat.mean()
print(f"\n[Flat] Normalized flat statistics:")
print(f"       Min gain: {normalized_flat.min():.3f} (dimmest pixels)")
print(f"       Max gain: {normalized_flat.max():.3f} (brightest pixels)")
print(f"       Variation: ±{100*normalized_flat.std():.1f}%")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
np.save(output_path, masterflat)
print(f"\n[Save] Master flat saved: {output_path}")

# Also save as TIFF for inspection
tiff_path = os.path.join(OUTPUT_DIR, "masterflat.tiff")
tifffile.imwrite(tiff_path, masterflat.astype(np.uint16))
print(f"[Save] TIFF version saved: {tiff_path}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print(f"\n[Plot] Creating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Master Flat Field
ax = axes[0, 0]
im = ax.imshow(masterflat, cmap='gray', origin='lower')
ax.set_title(f'Master Flat Field\n({len(stack)} frames, median)')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im, ax=ax, label='Counts')

# Plot 2: Normalized Gain Map
ax = axes[0, 1]
im = ax.imshow(normalized_flat, cmap='RdBu_r', vmin=0.9, vmax=1.1, origin='lower')
ax.set_title('Normalized Gain Map\n(Deviations from unity)')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
cbar = plt.colorbar(im, ax=ax, label='Relative Gain')
cbar.ax.axhline(1.0, color='black', linewidth=2, linestyle='--')

# Plot 3: Gain Distribution Histogram
ax = axes[1, 0]
ax.hist(normalized_flat.ravel(), bins=100, alpha=0.7, edgecolor='black')
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unity gain')
ax.set_xlabel('Relative Gain')
ax.set_ylabel('Number of Pixels')
ax.set_title('Pixel Gain Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Radial Profile (check for vignetting)
ax = axes[1, 1]
center_y, center_x = np.array(img_shape) // 2
y, x = np.indices(img_shape)
r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

# Bin radial profile
max_r = int(np.sqrt(2) * min(center_x, center_y))
r_bins = np.linspace(0, max_r, 50)
radial_profile = []
for i in range(len(r_bins) - 1):
    mask = (r >= r_bins[i]) & (r < r_bins[i+1])
    if mask.any():
        radial_profile.append(normalized_flat[mask].mean())
    else:
        radial_profile.append(np.nan)

r_centers = (r_bins[:-1] + r_bins[1:]) / 2
ax.plot(r_centers, radial_profile, 'o-', markersize=3)
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Unity gain')
ax.set_xlabel('Radius from center (pixels)')
ax.set_ylabel('Average Gain')
ax.set_title('Radial Gain Profile (Vignetting Check)')
ax.legend()
ax.grid(True, alpha=0.3)

# Add text note about vignetting
vignetting = 100 * (1 - radial_profile[-1] / radial_profile[0])
ax.text(0.95, 0.05, f'Edge/Center ratio: {radial_profile[-1]/radial_profile[0]:.3f}\n'
                    f'Vignetting: {vignetting:.1f}%',
        transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "masterflat_diagnostics.png")
plt.savefig(plot_path, dpi=150)
print(f"[Plot] Diagnostic plot saved: {plot_path}")
plt.show()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print(f"\n{'=' * 80}")
print("FLAT FIELD GENERATION COMPLETE")
print(f"{'=' * 80}")

print(f"\nInput:")
print(f"  Frames: {len(stack)}")
print(f"  Resolution: {img_shape[1]}×{img_shape[0]}")
print(f"  Mean brightness: {mean_counts:.0f} counts")

print(f"\nFlat Field Quality:")
print(f"  Pixel gain variation: ±{100*normalized_flat.std():.1f}%")
print(f"  Vignetting (edge/center): {vignetting:.1f}%")

if normalized_flat.std() < 0.05:
    print(f"  Quality: ✓ EXCELLENT (< 5% variation)")
elif normalized_flat.std() < 0.10:
    print(f"  Quality: ✓ GOOD (< 10% variation)")
else:
    print(f"  Quality: ⚠️ FAIR (> 10% variation)")
    print(f"  Recommendation: Check light source uniformity")

print(f"\nOutput:")
print(f"  Master flat: {output_path}")
print(f"  Diagnostics: {plot_path}")

print(f"\n{'=' * 80}")
print("NEXT STEPS:")
print(f"{'=' * 80}")
print(f"1. Review diagnostic plots (check for hot spots, dead zones)")
print(f"2. Copy {OUTPUT_FILE} to your analysis directory")
print(f"3. In process_andor_vmi_HYBRID.py, set:")
print(f"   ENABLE_FLAT_FIELD = True")
print(f"   FLAT_FIELD_PATH = \"{OUTPUT_FILE}\"")
print(f"4. Run VMI analysis script - flat field will be applied automatically")
print(f"{'=' * 80}\n")
