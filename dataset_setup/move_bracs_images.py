#!/usr/bin/env python3

import os
import shutil

# Source and destination directories
src_root = "/data/fftmil/BRACS/histoimage.na.icar.cnr.it/BRACS_WSI/"
delete_root = "/data/fftmil/BRACS/histoimage.na.icar.cnr.it/"
dst_dir = "/data/fftmil/BRACS/images/"

# Ensure destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Move all .svs files recursively
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith(".svs"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, file)

            # Skip if already exists
            if os.path.exists(dst_path):
                print(f"Skipped (already exists): {file}")
                continue

            try:
                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"Error moving {src_path}: {e}")

# Delete the entire source root directory
try:
    shutil.rmtree(delete_root)
    print(f"Deleted directory: {delete_root}")
except Exception as e:
    print(f"Error deleting {delete_root}: {e}")
