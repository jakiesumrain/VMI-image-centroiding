import h5py
import numpy as np
import os
import random

h5_dir = r"H:\Xe_sig"
files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
#fpath = os.path.join(h5_dir, random.choice(files))
fpath = os.path.join(h5_dir, "RAW-R0090-07_01_Xe_Ele_Extrig-S03969.h5")


def inspect(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}  shape={obj.shape}  dtype={obj.dtype}")
        if obj.ndim > 0:
            # read up to ~10M elements without loading the entire dataset
            n = min(10_000_000, obj.size)
            rows = max(1, n // (obj.size // obj.shape[0]))
            arr = obj[:rows].ravel()[:n]
            print(f"      min={arr.min():.4g}  max={arr.max():.4g}  "
                  f"mean={arr.mean():.4g}  (sampled {len(arr):,} of {obj.size:,} values)")
        else:
            print(f"      value={obj[()]}")
        for key, val in obj.attrs.items():
            print(f"      attr [{key}] = {val}")
    elif isinstance(obj, h5py.Group):
        print(f"  {name}/")
        for key, val in obj.attrs.items():
            print(f"    attr [{key}] = {val}")

with h5py.File(fpath, 'r') as hf:
    print(f"File: {os.path.basename(fpath)}")
    hf.visititems(inspect)
