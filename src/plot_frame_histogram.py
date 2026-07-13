import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random

h5_dir = r"H:\Xe_sig"
N = 1   # number of random h5 files to accumulate in the second plot
count_max = 50
thresh = 5

# Pre-computed dark frame (leave empty to skip)
dark_path = r"E:\work\projects\VMI-FEL\cpvmi\data_20260703\centroiding\dark_mean.npy"

files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]

# Load dark frame
# Note: dark and signal data are from different directories. The camera
# baseline may differ slightly between runs, so after dark subtraction
# the noise peak sits at a small positive value (not exactly zero).
# This is normal — the threshold and mask handle it correctly.
dark_frame = np.load(dark_path) if dark_path and os.path.exists(dark_path) else None
if dark_frame is not None:
    print(f"Dark frame loaded: mean={dark_frame.mean():.1f}, std={dark_frame.std():.1f}")
else:
    print("Dark subtraction disabled")

# --- Plot 1: single random frame ---
chosen_file = random.choice(files)
fpath = os.path.join(h5_dir, chosen_file)

with h5py.File(fpath, 'r') as hf:
    dset = hf['raw/data']
    n_frames = dset.shape[0]
    chosen_frame = random.randint(0, n_frames - 1)
    frame = dset[chosen_frame][:]

if dark_frame is not None:
    frame = frame.astype(np.float64) - dark_frame
    # Not clamping negatives: preserving zero-centered background
    # avoids statistical bias from truncating the noise distribution.

print(f"[Single frame] File: {chosen_file}  Frame: {chosen_frame}/{n_frames-1}")
print(f"  Pixel range: {frame.min():.0f} – {frame.max():.0f}")

# --- Plot 2: accumulated frames from N random h5 files ---
chosen_files = random.sample(files, min(N, len(files)))
acc = None
total_frames = 0

for fname in chosen_files:
    with h5py.File(os.path.join(h5_dir, fname), 'r') as hf:
        data = hf['raw/data'][:]
    for f in data:
        corrected = f.astype(np.float64)
        if dark_frame is not None:
            corrected = corrected - dark_frame
            # Not clamping negatives: preserves correct noise statistics.
        acc = corrected if acc is None else acc + corrected
        total_frames += 1
    print(f"  Loaded {fname}  ({data.shape[0]} frames)")

print(f"[Accumulated] {total_frames} frames from {len(chosen_files)} files")
print(f"  Accumulated pixel range: {acc.min():.0f} – {acc.max():.0f}")

# --- plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Row 0: single frame
axes[0, 0].hist(frame.flatten(), bins=800, range = (-40.0, count_max), color = 'k')
axes[0, 0].set_xlabel('Pixel value')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_yscale('log')
axes[0, 0].axvline(x = thresh, ls = ':', label = f'thresh = {thresh}')
axes[0, 0].set_title(f'Histogram — single frame\n{chosen_file}  frame {chosen_frame}')
axes[0, 0].set_xlim([-40, count_max])
axes[0, 0].legend()

im0 = axes[0, 1].imshow(frame, cmap='gray', aspect='equal', origin='lower')
axes[0, 1].set_title(f'Image — single frame\n{chosen_file}  frame {chosen_frame}')
fig.colorbar(im0, ax=axes[0, 1])
axes[0, 1].text(0.02, 0.02, f'min={frame.min()}  max={frame.max()}',
                transform=axes[0, 1].transAxes, fontsize=8,
                color='white', ha='left', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

# Row 1: accumulated (averaged per frame)
avg = acc / total_frames
axes[1, 0].hist(avg.flatten(), bins=800, range = (-40.0, count_max), color = 'k')
axes[1, 0].set_xlabel('Average pixel value per frame')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_yscale('log')
axes[1, 0].axvline(x = thresh, ls = ':')
axes[1, 0].set_title(f'Histogram — averaged\n({total_frames} frames, {len(chosen_files)} files)')
axes[1, 0].set_xlim([-40, count_max])

im1 = axes[1, 1].imshow(avg, cmap='gray', aspect='equal', origin='lower')
axes[1, 1].set_title(f'Image — averaged\n({total_frames} frames, {len(chosen_files)} files)')
fig.colorbar(im1, ax=axes[1, 1])
axes[1, 1].text(0.02, 0.02, f'min={avg.min():.0f}  max={avg.max():.0f}',
                transform=axes[1, 1].transAxes, fontsize=8,
                color='white', ha='left', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

plt.tight_layout()
plt.show()
