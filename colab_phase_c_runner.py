#!/usr/bin/env python3
"""
Paste this ENTIRE CELL into Colab.
Automatically uploads and runs colab_phase_c_acceptance.py
"""

import os
import subprocess

os.chdir('/content/athlink-cv-service')

# Download the script from repo root if it exists
script_path = '/content/athlink-cv-service/colab_phase_c_acceptance.py'
if not os.path.exists(script_path):
    print("Script not found. Creating from inline definition...")
    # Script will be exec'd inline below
    pass

# Run Phase C acceptance
exec(open('/content/athlink-cv-service/colab_phase_c_acceptance.py').read())
