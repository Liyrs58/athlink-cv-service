# model_downloader.py — DEPRECATED
# Model loading is now handled by services/model_cache.py
# This file is kept empty to avoid import errors in existing code.

import logging
logger = logging.getLogger(__name__)

FOOTBALL_MODEL_PATH = "weights/football_yolov8m.pt"

def download_football_model():
    """No-op — models are pre-loaded via model_cache at server startup."""
    logger.info("download_football_model() is deprecated — models loaded via model_cache")
    return FOOTBALL_MODEL_PATH
