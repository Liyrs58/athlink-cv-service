"""Lazy-load and cache YOLO and ball detection models at startup.
"""

import os
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_tracking_model = None
_ball_model = None

def get_tracking_model():
    global _tracking_model
    if _tracking_model is None:
        model_path = os.getenv('YOLO_MODEL_PATH', 'models/roboflow_players.pt')
        logger.info(f"Loading tracking model: {model_path}")
        _tracking_model = YOLO(model_path)
        logger.info("Tracking model loaded and cached")
    return _tracking_model

def get_ball_model():
    global _ball_model
    if _ball_model is None:
        ball_path = os.environ.get("BALL_MODEL_PATH", "models/roboflow_ball.pt")
        if os.path.exists(ball_path):
            logger.info(f"Loading ball model: {ball_path}")
            _ball_model = YOLO(ball_path)
        else:
            logger.info("Loading ball model: yolov8s.pt (fallback)")
            _ball_model = YOLO('yolov8s.pt')
        logger.info("Ball model loaded and cached")
    return _ball_model

def preload_all_models():
    logger.info("Pre-loading all models at startup...")
    get_tracking_model()
    get_ball_model()
    logger.info("All models pre-loaded and cached")
