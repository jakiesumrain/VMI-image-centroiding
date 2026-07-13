# VMI Centroiding Analysis Toolkit

VMI (Velocity Map Imaging) photoelectron centroiding analysis using Picasso localization microscopy algorithms with HDF5 and TIFF input support.

## Which script should I use?

### HDF5-based (new pipeline)

| Script | Algorithm | Best for |
|--------|-----------|----------|
| `process_andor_vmi_PICASSO_v3_h5.py` | Picasso MLE + Net Gradient, per-frame | Single HDF5 run, tuning, or small datasets |
| `process_andor_vmi_PICASSO_v4_h5_batch.py` | Picasso MLE + batch movie mode (all 126 frames at once) | Large datasets, ~3-5× faster than v3 |
| `process_andor_vmi_PICASSO_v5_h5_batch.py` | Picasso LQ *or* MLE + batch movie mode | Large datasets with configurable fitting; avoids MLE divergence issues |

### TIFF-based (legacy)

| Script | Algorithm | Best for | Pile-up handling |
|--------|-----------|----------|-----------------|
| `process_andor_vmi_HYBRID.py` | trackpy center-of-mass | Everyday VMI analysis | Detection + warning only |
| `process_andor_vmi_PICASSO_v2.py` | Picasso Net Gradient + MLE fit | VMI with overlapping events | Better detection of close neighbors |
| `process_andor_vmi_PHOTUTILS.py` | photutils PSF fitting + SourceGrouper | VMI with severe pile-up | Full joint deblending |

**Start with v5 batch (HDF5).** It's the fastest and supports both LQ (fast, robust) and MLE (statistically optimal) fitting. Use v3 for tuning on single files. Fall back to TIFF scripts only for legacy data.

---

## Quick Start (HDF5 pipeline)

### 1. Pre-compute dark frame (optional, but recommended)

```bash
# Edit config.toml to set dark_data_dir first
uv run python compute_dark.py
# Produces dark_mean.npy (or dark_median.npy) — loads instantly in subsequent runs
```

### 2. Build GMD map (optional, for pulse-energy correction)

```bash
# Edit paths in src/build_gmd_map.py
uv run python src/build_gmd_map.py
# Produces gmd_map_all.txt — maps each frame to its FEL pulse energy
```

### 3. Edit `config.toml`

Set the key paths under `[vmi_picasso]`:

```toml
h5_dir    = "H:/Xe_sig"          # HDF5 signal data directory
h5_pattern = "RAW-*.h5"           # HDF5 file glob pattern
h5_dataset = "raw/data"           # HDF5 dataset with frame data

dark_data_dir = "H:/Xe_bkg"       # Dark frame HDF5 directory
dark_file     = "dark_mean.npy"   # Pre-computed dark (from step 1)

output_dir = "vmi_analysis_output"
```

### 4. Tune parameters on one file

```toml
tuning_only = true           # in [vmi_picasso.processing]
tuning_frame_index = 300     # which frame to inspect
```

Run the script, inspect the 4-panel diagnostic, adjust `box_size` and `min_net_gradient` in config.toml, and repeat.

### 5. Run batch processing

```toml
tuning_only = false          # process all files
```

```bash
uv run python process_andor_vmi_PICASSO_v5_h5_batch.py
```

---

## Configuration Reference (`config.toml`)

### Input/Output (HDF5)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h5_dir` | — | Directory containing HDF5 signal files |
| `h5_pattern` | `"RAW-*.h5"` | Glob pattern for HDF5 files |
| `h5_dataset` | `"raw/data"` | HDF5 internal path to frame data (N_frames × H × W) |
| `h5_meta` | `"raw/meta"` | HDF5 internal path to metadata (GMD column) |
| `meta_column_gmd` | 1 | Column index in `raw/meta` holding GMD (pulse energy) |
| `gmd_map_path` | — | Pre-computed frame-to-GMD map (overrides `meta_column_gmd`) |
| `dark_data_dir` | — | Directory containing dark HDF5 files |
| `dark_h5_pattern` | `"RAW-*.h5"` | Glob pattern for dark HDF5 files |
| `dark_file` | — | Pre-computed dark `.npy` file (instant load, overrides `dark_data_dir`) |
| `output_dir` | `"vmi_analysis_output"` | Where results go |
| `checkpoint_file` | — | HDF5 checkpoint table for batch resume/partial results |

### Localization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `box_size` | 5 | ROI size for MLE fitting (odd). Rule: 6σ + 1. |
| `min_net_gradient` | 500 | Detection threshold. Lower = more detections (incl. noise). |
| `camera_baseline` | 0 | Dark count level. Set to 0 if dark subtraction is enabled. |
| `camera_sensitivity` | 1.0 | e⁻/ADU conversion factor |
| `camera_gain` | 1 | EM gain (1 for non-EMCCD) |
| `camera_qe` | 0.9 | Quantum efficiency (not actively used by Picasso) |

### Fitting Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mle_convergence` | 0.1 | Fit precision threshold. Smaller = more precise, slower. |
| `mle_max_iterations` | 50 | Max MLE iterations before giving up |
| `mle_method` | `"sigma"` | `"sigma"` (symmetric, stable) or `"sigmaxy"` (elliptical spots) |
| `fitting_method` | `"mle"` | `"mle"` (Poisson-optimal, slower) or `"lq"` (least-squares, fast, robust) |

### Processing Options (`[vmi_picasso.processing]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_parallel` | true | Use multiple CPU cores |
| `num_cores` | 6 | Number of cores. `null` = all available. |
| `enable_dark_subtraction` | true | Subtract dark frame before detection |
| `dark_method` | `"mean"` | `"mean"` (streaming, all files) or `"median"` (subsample, up to `dark_frame_count`) |
| `dark_frame_count` | 250000 | Max frames used for median dark (only for `dark_method = "median"`) |
| `enable_flat_field` | false | Detector response normalization |
| `enable_circular_mask` | false | Exclude events outside MCP active area |
| `tuning_only` | true | Process one frame and show diagnostics |

### Histogram & GMD (`[vmi_picasso.histogram]`, `[vmi_picasso.gmd]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution_factor` | 1 | 2D histogram resolution multiplier (1 = native) |
| `enable_abel` | false | Abel inversion (requires `pyabel`) |
| `enable_gmd_weighting` | false | Weight events by mean_gmd / pulse_gmd |
| `gmd_weight_min` | 0.0 | Only keep frames with weight ≥ this |
| `gmd_weight_max` | 4.0 | Only keep frames with weight ≤ this |

---

## Script Reference

### HDF5 Processing Scripts

| Script | Key Feature | Usage |
|--------|-------------|-------|
| `compute_dark.py` | Pre-compute dark frame (mean or median) from HDF5 files | `uv run python compute_dark.py` |
| `process_andor_vmi_PICASSO_v3_h5.py` | Per-frame Picasso processing of HDF5 files | Tuning + single-file analysis |
| `process_andor_vmi_PICASSO_v4_h5_batch.py` | Batch movie-mode (126 frames/call), MLE only | ~3-5× faster than v3 |
| `process_andor_vmi_PICASSO_v5_h5_batch.py` | Batch movie-mode, configurable LQ or MLE | Recommended for production |

### Utility Scripts (`src/`)

| Script | Purpose |
|--------|---------|
| `src/build_gmd_map.py` | Build frame-to-GMD correspondence map by matching timestamps against `current-time-both.txt` |
| `src/plot_gmd_weights.py` | Histogram of GMD weight distribution — helps choose weight cap values |
| `src/inspect_h5.py` | Inspect HDF5 file structure, datasets, shapes, dtypes, and attributes |
| `src/plot_frame_histogram.py` | Plot pixel-value histograms of raw vs dark-subtracted frames |
| `src/h5_accumulate_fast_gmd.py` | Fast frame accumulation with thresholding, morphological denoising, and optional GMD scaling |

---

## Tuning Workflow

```
1. Set tuning_only = true, tuning_frame_index = 300
2. Run the script (v3 for single file, or v5 in tuning mode)
3. Check Panel 1: all blobs circled? No noise circles?
   - Missing blobs → decrease min_net_gradient
   - Noise circles → increase min_net_gradient
4. Check Panel 2/3: clean signal distribution?
   - Spike at 2× single-electron → pile-up (reduce laser or increase separation)
5. Check Panel 4: precision OK?
   - Median > 0.5 px → increase box_size
6. Repeat until satisfied
7. Set tuning_only = false, run full batch
```

## What "mass", "flux", "photons" mean

All scripts report the **total integrated brightness** of a blob, just under different names:

| Script | Column name | How it's computed |
|--------|------------|-------------------|
| HYBRID | `mass` | Raw sum of pixel values in the blob (background subtracted) |
| PICASSO | `mass` (renamed from `photons`) | MLE/LQ-fitted integrated intensity of the Gaussian model |
| PHOTUTILS | `mass` (renamed from `flux_fit`) | Fitted integrated flux of the Gaussian PSF model |

They represent the same physical quantity (total signal per electron) but computed differently. Their values may differ by a factor of 2-3 between scripts due to different background estimation and fitting methods.

**Each detection always counts as exactly 1 event in the final VMI histogram**, regardless of its mass/ flux/ photons value.

## Output files

```
vmi_analysis_output/
├── vmi_results.h5                # All events (x, y, mass, frame, weight, ...)
├── vmi_results.csv               # Same, CSV format
├── vmi_results_raw_v5batch.h5    # Batch checkpoint table (v5)
├── vmi_2d_raw.npy                # 2D VMI histogram
├── vmi_3d_abel.npy               # 3D Abel-inverted (if enabled)
├── 01_tuning_check_picasso.png   # Tuning diagnostic
├── 04_final_summary.png          # Final VMI image + statistics
```

## Requirements

- Python 3.11+
- [Picasso](https://github.com/jungmannlab/picasso) (localization microscopy)
- `h5py`, `numpy`, `matplotlib`, `pandas`, `tqdm`, `tomli` (or Python 3.11+ built-in `tomllib`)
- Optional: `pyabel` (Abel inversion), `opencv-python` (morphological denoising)

Install with:

```bash
uv pip install h5py numpy matplotlib pandas tqdm tomli pyabel opencv-python
```

## Notes

- **Binning must match** your camera acquisition setting. The `box_size` in config.toml and the camera's actual binning mode must be consistent.
- **Dark frames** should contain no signal — acquire them with the laser/FEL off.
- **GMD normalization** corrects for FEL pulse energy shot-to-shot fluctuations. Use `src/plot_gmd_weights.py` to check the weight distribution and set sensible `gmd_weight_min/max` caps.
- **Abel transform** (PyAbel) converts the 2D VMI projection to a 3D momentum slice. Required for quantitative momentum/energy analysis.
