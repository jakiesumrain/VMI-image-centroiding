#!/usr/bin/env python3
"""
Plot histogram of GMD weights (mean_gmd / gmd_value).

Shows the distribution of frame weights to help choose a
sensible cap for GMD normalization.
"""

import numpy as np
import matplotlib.pyplot as plt

gmd_map_path = r"E:\work\projects\VMI-FEL\cpvmi\data_20260703\gmd_map_all.txt"

# Load GMD values (column 4 = GMD method 1)
map_data = np.loadtxt(gmd_map_path, comments='#', usecols=(0, 4))
gmd_vals = map_data[:, 1]

mean_gmd = gmd_vals.mean()
gmd_safe = np.maximum(gmd_vals, 1e-12)
weights = mean_gmd / gmd_safe

print(f"Total frames: {len(weights)}")
print(f"GMD mean:     {mean_gmd:.3e}")
print(f"Weight stats:")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  {p:3d}% = {np.percentile(weights, p):.3f}")
print(f"  >5:   {(weights > 5).sum():>6d} frames ({(weights > 5).mean()*100:.2f}%)")
print(f"  >10:  {(weights > 10).sum():>6d} frames ({(weights > 10).mean()*100:.2f}%)")
print(f"  >50:  {(weights > 50).sum():>6d} frames")
print(f"  >100: {(weights > 100).sum():>6d} frames")
print(f"  max = {weights.max():.1f}")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# --- Panel 1: GMD values histogram ---
ax = axes[0]
valid = gmd_vals[gmd_vals > 0]  # skip zero/near-zero for log scale
ax.hist(valid, bins=200, color='k')
ax.set_xlabel('GMD value (method 1)')
ax.set_ylabel('Count')
ax.set_title('GMD Value Distribution')
ax.text(0.95, 0.95,
        f"mean = {gmd_vals.mean():.3e}\n"
        f"median = {np.median(gmd_vals):.3e}\n"
        f"min = {gmd_vals.min():.3e}\n"
        f"max = {gmd_vals.max():.3e}",
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Panel 2: Weight distribution [0-10] ---
ax = axes[1]
ax.hist(weights, bins=200, range=(0, 10), color='k')
ax.axvline(5, color='r', ls='--', label='Cap at 5')
ax.set_xlabel('Weight = mean_gmd / gmd')
ax.set_ylabel('Count')
ax.set_title('GMD Weight Distribution [0-10]')
ax.legend()
ax.text(0.95, 0.95, f">5: {(weights > 5).sum()} frames\n{'>10: ' + str((weights > 10).sum()) + ' frames'}",
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Panel 3: Weight distribution [0-200] (log y) ---
ax = axes[2]
ax.hist(weights, bins=200, range=(0, 200), color='k')
ax.axvline(5, color='r', ls='--', label='Cap at 5')
ax.set_xlabel('Weight')
ax.set_ylabel('Count')
ax.set_title('GMD Weight Distribution [0-200]')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('gmd_weight_histogram.png', dpi=150)
print("\nSaved: gmd_weight_histogram.png")
plt.show()
