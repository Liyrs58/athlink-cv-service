"""
Download football-specific YOLO model at build time or first use.
Model: yolov8m trained on broadcast football footage (player, goalkeeper, referee, ball).
Source: https://huggingface.co/keremberke/yolov8m-football-player-detection
"""

import os
import logging
import urllib.request

logger = logging.getLogger(__name__)

FOOTBALL_MODEL_URL = "https://huggingface.co/keremberke/yolov8m-football-player-detection/resolve/main/best.pt"
FOOTBALL_MODEL_PATH = "weights/football_yolov8m.pt"


def download_football_model():
    """Download football YOLO model if not already present. Safe to call multiple times."""
    if os.path.exists(FOOTBALL_MODEL_PATH):
        return FOOTBALL_MODEL_PATH

    os.makedirs("weights", exist_ok=True)
    logger.info("Downloading football-specific YOLO model from HuggingFace...")
    try:
        token = os.getenv('HUGGINGFACE_TOKEN', '')
        headers = {'Authorization': f'Bearer {token}'}
        req = urllib.request.Request(FOOTBALL_MODEL_URL, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(FOOTBALL_MODEL_PATH, 'wb') as f:
                f.write(response.read())
        logger.info("Football model downloaded to %s", FOOTBALL_MODEL_PATH)
    except Exception as e:
        logger.error("Failed to download football model: %s", e)
        if os.path.exists(FOOTBALL_MODEL_PATH):
            os.remove(FOOTBALL_MODEL_PATH)
        raise
    return FOOTBALL_MODEL_PATH


if __name__ == "__main__":
    download_football_model()
    print(f"Football model ready at {FOOTBALL_MODEL_PATH}")
