# VMI Centroiding Analysis — Tutorial

## Which script should I use?

| Script | Algorithm | Best for | Pile-up handling |
|--------|-----------|----------|-----------------|
| `process_andor_vmi_HYBRID.py` | trackpy center-of-mass | Everyday VMI analysis | Detection + warning only |
| `process_andor_vmi_PICASSO_v2.py` | Picasso Net Gradient + MLE fit | VMI with overlapping events | Better detection of close neighbors |
| `process_andor_vmi_PHOTUTILS.py` | photutils PSF fitting + SourceGrouper | VMI with severe pile-up | Full joint deblending |

**Start with HYBRID.** It's the simplest and fastest. Switch to PICASSO or PHOTUTILS only if you need better pile-up recovery.

---

## Quick Start

### 1. Edit `config.toml`

Set these three paths under `[vmi_picasso]` (they're shared by all scripts):

```toml
data_dir     = "F:/data/my_experiment/_1/Default"   # your signal data
dark_data_dir = "F:/data/my_darks/_1/Default"        # dark frames (leave "" if none)
output_dir   = "vmi_analysis_output"                  # where results go
```

### 2. Tune parameters on one frame

Set `tuning_only = true` under `[vmi_picasso.processing]`, then run the script. A 4-panel diagnostic plot opens — inspect it and adjust parameters. Repeat until satisfied.

### 3. Run the batch

Set `tuning_only = false` and run the script again. All frames are processed and the final VMI image is saved.

---

## HYBRID Script Parameters

Configured at the top of `process_andor_vmi_HYBRID.py` (not in config.toml):

| Parameter | Meaning | How to tune |
|-----------|---------|-------------|
| `DIAMETER` | Expected blob size in pixels (odd integer). ≈ 2× PSF FWHM. | Check Panel 1: circles should encompass each blob without excessive whitespace. For 2×2 binning: 5. For 1×1 binning: 7. |
| `MIN_MASS` | Minimum integrated brightness to accept an event. | Panel 3 (mass histogram): no spike at the threshold (red line). Increase if noise creates a spike at the cutoff. |
| `SEPARATION` | Minimum pixel distance between two events. | Prevents merging adjacent electrons. Set ≥ DIAMETER. Panel 3: a peak at 2× single-electron mass means SEPARATION is too small. |

**Tuning check panels (HYBRID):**

- **Panel 1** — Raw frame + red circles at detection positions. All visible blobs should have a circle; no circles on dark background.
- **Panel 2** — Mass vs eccentricity scatter. Real electrons cluster at low eccentricity; cosmic rays scatter high.
- **Panel 3** — Mass histogram. Single Gaussian peak = clean data. Peak at 2× means pile-up.

---

## PICASSO v2 Script Parameters

Configured in `config.toml` under `[vmi_picasso]`:

### Core detection parameters

| Parameter | SMLM meaning | VMI meaning | How to tune |
|-----------|-------------|-------------|-------------|
| `box_size` | Fitting ROI size (odd). Rule: 6σ + 1. | Same — the fitting window around each electron blob. | Panel 1: circles should fully enclose blobs. For 1×1 binning (FWHM≈4 px, σ≈1.7): use **11**. For 2×2 binning (FWHM≈2.5 px, σ≈1.06): use **7**. |
| `min_net_gradient` | Detection threshold. Net Gradient ≈ spot brightness / noise. | Primary sensitivity knob. Lower = more detections (including noise). Higher = fewer, cleaner detections. | Panel 3 (Net Gradient histogram): the red threshold line should sit in the **valley** between the noise peak (near zero) and the signal peak. |

### Camera parameters (leave at defaults unless you know your camera specs)

| Parameter | Effect on VMI analysis | When to change |
|-----------|----------------------|----------------|
| `camera_baseline` | Set to **0** when dark subtraction is enabled. | Only change if you disable dark subtraction — then set to mean(dark_frame). |
| `camera_sensitivity` | e⁻ per ADU conversion. Affects photon count values, not positions. | Only if you need absolute photon numbers. Default 1.0 is fine for VMI. |
| `camera_gain` | EM gain. Set to **1** for non-EMCCD cameras. | Leave at 1. |
| `camera_qe` | Quantum efficiency. Not actively used by Picasso. | Leave at 0.9. |

Does **not** affect the final VMI histogram (each detection = 1 count regardless of brightness).

### Convergence parameters

| Parameter | Meaning | Guideline |
|-----------|---------|-----------|
| `mle_convergence` | Fit precision. Smaller = more precise, slower. | 0.01 is a good balance. Lower to 0.001 if you see poor localization precision. |
| `mle_max_iterations` | Max MLE fitting iterations before giving up. | 1000 is generous. Increase only if many fits fail to converge. |

### Processing options (`[vmi_picasso.processing]`)

| Parameter | Meaning |
|-----------|---------|
| `enable_parallel` | Use multiple CPU cores for batch processing. Recommended: true. |
| `num_cores` | Number of cores. `null` = all available. |
| `enable_dark_subtraction` | Subtract median dark frame before detection. Recommended: true if you have dark frames. |
| `dark_frame_count` | How many dark frames to median-average. More = cleaner dark. |
| `tuning_only` | true = process one frame and show diagnostics, skip batch. false = full batch run. |

### Tuning visualization (`[vmi_picasso]`)

| Parameter | Controls |
|-----------|----------|
| `tuning_frame_index` | Which frame (0-based) to use for tuning. |
| `tuning_bins_photons` | Bins in Panel 2 (photon histogram). |
| `tuning_bins_net_gradient` | Bins in Panel 3 (Net Gradient histogram). |
| `tuning_bins_precision` | Bins in Panel 4 (precision histogram). |

### Tuning check panels (PICASSO)

- **Panel 1** — Raw frame + red circles at fitted positions. All visible blobs should have a circle. No circles on noise. **If circles are offset from blobs, increase `box_size`.**
- **Panel 2** — Photon (integrated brightness) distribution. Clean single peak = good. Peak at 2× median = pile-up of fully overlapping electrons. No spike at low values.
- **Panel 3** — Net Gradient (detection metric) distribution. The red threshold line should be in the valley between the noise cluster (near zero) and the signal cluster. If the threshold cuts through the signal peak, lower `min_net_gradient`.
- **Panel 4** — Localization precision from MLE uncertainty. Lower is better. Median < 0.3 px = good. Long tail or many NaN values = fitting problems (try larger `box_size`).

---

## Tuning workflow

```
1. Set tuning_only = true, tuning_frame_index = 1000
2. Run the script
3. Check Panel 1: all blobs circled? No noise circles?
   - Missing blobs → decrease min_net_gradient (PICASSO) or MIN_MASS (HYBRID)
   - Noise circles → increase threshold
4. Check Panel 2/3: clean signal distribution?
   - Spike at 2× single-electron → pile-up (reduce laser or increase separation)
5. Check Panel 4 (PICASSO only): precision OK?
   - Median > 0.5 px → increase box_size
6. Repeat until satisfied
7. Set tuning_only = false, run full batch
```

## What "mass", "flux", "photons" mean

All three scripts report the **total integrated brightness** of a blob, just under different names:

| Script | Column name | How it's computed |
|--------|------------|-------------------|
| HYBRID | `mass` | Raw sum of pixel values in the blob (background subtracted) |
| PICASSO | `mass` (renamed from `photons`) | MLE-fitted integrated intensity of the Gaussian model |
| PHOTUTILS | `mass` (renamed from `flux_fit`) | Fitted integrated flux of the Gaussian PSF model |

They represent the same physical quantity (total signal per electron) but computed differently. Their values may differ by a factor of 2-3 between scripts due to different background estimation and fitting methods.

**Each detection always counts as exactly 1 event in the final VMI histogram**, regardless of its mass/ flux/ photons value.

## Output files

```
vmi_analysis_output/
├── vmi_results.h5              # All events (x, y, mass, frame, weight, ...)
├── vmi_results.csv             # Same, CSV format
├── vmi_2d_raw.npy              # 2D VMI histogram
├── vmi_3d_abel.npy             # 3D Abel-inverted (if enabled)
├── 01_tuning_check_picasso.png # Tuning diagnostic
├── 04_final_summary.png        # Final VMI image + statistics
```

## Notes

- **Binning must match** your camera acquisition setting. The HYBRID script's `BINNING` variable, the `box_size` in config.toml, and the camera's actual binning mode must be consistent.
- **Dark frames** should contain no signal — acquire them with the laser/FEL off.
- **GMD normalization** corrects for FEL pulse energy shot-to-shot fluctuations. Enable if your GMD values are embedded in TIFF metadata.
- **Abel transform** (PyAbel) converts the 2D VMI projection to a 3D momentum slice. Required for quantitative momentum/energy analysis.
