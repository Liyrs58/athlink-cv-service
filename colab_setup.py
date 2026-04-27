#!/usr/bin/env python3
"""
Colab setup - downloads all models automatically
Run this FIRST in Colab before tracking
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("COLAB SETUP - DOWNLOADING MODELS")
print("=" * 70)

# Create models directory
models_dir = Path('/content/athlink-cv-service/models')
models_dir.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Models directory: {models_dir}")

# Model URLs (from Hugging Face / Roboflow)
models = {
    'yolov8n-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt',
    'osnet_x1_0_msmt17.pt': 'https://drive.google.com/uc?id=1vduhm68Xn7Gy7-la91AxBAa97MTKMw3O&export=download'
}

print("\nDownloading models:")
print("-" * 70)

import subprocess
for filename, url in models.items():
    filepath = models_dir / filename

    if filepath.exists():
        print(f"✓ {filename} (already exists, skipping)")
        continue

    print(f"⏳ Downloading {filename}...")
    try:
        if 'drive.google.com' in url:
            # Google Drive
            cmd = f'wget --no-check-certificate -q -O {filepath} "{url}"'
        else:
            # Direct URL
            cmd = f'wget -q -O {filepath} "{url}"'

        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0 and filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.1f}MB)")
        else:
            print(f"✗ Failed to download {filename}")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

print("\n" + "=" * 70)
print("Verifying models:")
print("-" * 70)

for filename in models.keys():
    filepath = models_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} ({size_mb:.1f}MB)")
    else:
        print(f"✗ {filename} NOT FOUND")

print("\n" + "=" * 70)
print("✓ SETUP COMPLETE - Ready to run tracking")
print("=" * 70)
