#!/usr/bin/env python3

import os
import shutil

# Define root and target directories
root_dir = "/data/fftmil/LUAD"
target_dir = os.path.join(root_dir, "images")

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Move all .svs files to the target directory
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.lower().endswith(".svs"):
            src_path = os.path.join(dirpath, file)
            dst_path = os.path.join(target_dir, file)

            # Skip if the file is already in the target directory
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue

            # Move only if it doesn't already exist in target
            if not os.path.exists(dst_path):
                print(f"Moving: {src_path} -> {dst_path}")
                shutil.move(src_path, dst_path)
            else:
                print(f"Skipped (already exists): {dst_path}")

# Delete all subdirectories under root_dir except "images"
for entry in os.listdir(root_dir):
    entry_path = os.path.join(root_dir, entry)
    if os.path.isdir(entry_path) and entry != "images":
        print(f"Deleting directory: {entry_path}")
        shutil.rmtree(entry_path)
