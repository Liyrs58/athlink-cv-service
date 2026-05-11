#!/usr/bin/env python3
"""
Paste this ENTIRE CELL into Colab to run the model probe.
Do NOT modify.
"""

# Cell 1: Setup + Probe
import os
import sys
os.chdir('/content/athlink-cv-service')
sys.path.insert(0, '/content/athlink-cv-service')

exec(open('/content/athlink-cv-service/colab_model_probe.py').read())
