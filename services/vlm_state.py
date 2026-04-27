import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
import torch

class GameState(Enum):
    PLAY = "play"
    PAUSED = "paused"
    BENCH_SHOT = "bench_shot"
    INJURY = "injury"
    SET_PIECE = "set_piece"

class VLMStateMachine:
    def __init__(self, device="cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.frozen_tracks = {}
        self.frame_counter = 0

        # Load Moondream VLM
        try:
            from moondream import Moondream
            self.md = Moondream.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self.md = self.md.to(device)
            self.vlm_available = True
            print("[VLM] Moondream loaded successfully")
        except Exception as e:
            print(f"[VLM] Failed to load Moondream: {e}. Using fallback heuristics.")
            self.vlm_available = False

    def analyze(self, frame, video_frame: int) -> GameState:
        """
        Analyze game state every 50 frames.
        Returns: GameState enum
        """
        self.frame_counter += 1

        # Only analyze every 50 frames
        if self.frame_counter % 50 != 0:
            return self.state

        self.prev_state = self.state

        if self.vlm_available:
            self.state = self._vlm_classify(frame, video_frame)
        else:
            self.state = self._heuristic_classify(frame, video_frame)

        if self.state != self.prev_state:
            print(f"[VLM] Frame {video_frame}: {self.prev_state.value} → {self.state.value}")

        return self.state

    def _vlm_classify(self, frame, video_frame: int) -> GameState:
        """Use Moondream to classify game state."""
        try:
            # Encode frame as image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ask VLM about game state
            prompt = (
                "Is this a football match in play, a bench/dugout shot, an injury pause, "
                "a set piece (free kick/corner), or a broadcast pause? "
                "Reply with exactly one word: PLAY, BENCH_SHOT, INJURY, SET_PIECE, or PAUSED."
            )

            answer = self.md.query(frame_rgb, prompt)["answer"].strip().upper()

            # Parse response
            state_map = {
                "PLAY": GameState.PLAY,
                "BENCH_SHOT": GameState.BENCH_SHOT,
                "INJURY": GameState.INJURY,
                "SET_PIECE": GameState.SET_PIECE,
                "PAUSED": GameState.PAUSED,
            }
            return state_map.get(answer, self.state)
        except Exception as e:
            print(f"[VLM] Error during classification: {e}. Falling back to heuristics.")
            return self._heuristic_classify(frame, video_frame)

    def _heuristic_classify(self, frame, video_frame: int) -> GameState:
        """Fallback: use heuristics based on green field presence."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green field detection (H: 35-85, S: 40-255, V: 40-255)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = mask.sum() / (frame.shape[0] * frame.shape[1] * 255)

        # If <30% green, likely bench shot or cutaway
        if green_ratio < 0.3:
            return GameState.BENCH_SHOT

        return GameState.PLAY

    def freeze_tracks(self, tracks: List, video_frame: int):
        """
        Save active tracks before a pause/bench shot.
        Tracks format: [(x1, y1, x2, y2, tid, conf, cls, ...)]
        """
        self.frozen_tracks = {}
        for t in tracks:
            if len(t) >= 5:
                tid = int(t[4])
                bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                conf = float(t[5]) if len(t) > 5 else 0.0
                self.frozen_tracks[tid] = {
                    "bbox": bbox,
                    "confidence": conf,
                    "frozen_at": video_frame,
                }
        print(f"[VLM] Froze {len(self.frozen_tracks)} tracks at frame {video_frame}")

    def resume_tracks(self, current_tracks: List, video_frame: int) -> Dict[int, int]:
        """
        Match current visible players to frozen roster by:
        1. Position proximity (IoU > 0.5)
        2. Confidence score
        Returns: Dict[new_tid -> old_tid] mapping for ID override
        """
        if not self.frozen_tracks:
            return {}

        id_mapping = {}
        matched_old = set()

        for ct in current_tracks:
            if len(ct) < 5:
                continue

            new_tid = int(ct[4])
            curr_bbox = [float(ct[0]), float(ct[1]), float(ct[2]), float(ct[3])]

            # Find best match in frozen tracks by IoU
            best_iou = 0
            best_old_tid = None

            for old_tid, frozen in self.frozen_tracks.items():
                if old_tid in matched_old:
                    continue

                frozen_bbox = frozen["bbox"]
                iou = self._bbox_iou(curr_bbox, frozen_bbox)

                if iou > best_iou and iou > 0.3:  # Must have >30% overlap
                    best_iou = iou
                    best_old_tid = old_tid

            if best_old_tid is not None:
                id_mapping[new_tid] = best_old_tid
                matched_old.add(best_old_tid)

        print(f"[VLM] Resumed {len(id_mapping)} tracks at frame {video_frame}")
        return id_mapping

    def _bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
