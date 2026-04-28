import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional


class GameState(Enum):
    PLAY = "play"
    BENCH_SHOT = "bench_shot"


class PlayerRegistry:
    """
    Permanent player roster built from first 30 frames.
    Each slot = one real player, identified by kit color signature.

    After registration, every new BoT-SORT track ID gets matched
    to a permanent slot ID. This means:
    - P1 is always the Villa goalkeeper
    - P2 is always the Villa #9
    - etc.
    Even after camera pans, bench shots, or ID switches.
    """

    def __init__(self, max_players=30):
        self.max_players = max_players
        self.slots: Dict[int, dict] = {}       # slot_id -> appearance data
        self.botsort_to_slot: Dict[int, int] = {}  # botsort_tid -> slot_id
        self.next_slot = 1
        self.is_frozen = False
        self._frames_seen = 0
        self._registration_frames = 30

    def update(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        """
        Called every PLAY frame. Returns {botsort_tid: permanent_slot_id}.
        """
        if len(tracks) == 0:
            return dict(self.botsort_to_slot)

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(frame, tracks, video_frame)
            if self._frames_seen >= self._registration_frames:
                self.is_frozen = True
                print(f"[Registry] Frozen with {len(self.slots)} permanent player slots")
        else:
            self._match(frame, tracks, video_frame)

        return dict(self.botsort_to_slot)

    def flush_stale_mappings(self, active_tids: set):
        """
        Remove botsort_to_slot entries for BoT-SORT IDs that no longer exist.
        Called after bench→play transition so new IDs get re-matched to slots.
        """
        stale = [tid for tid in self.botsort_to_slot if tid not in active_tids]
        for tid in stale:
            del self.botsort_to_slot[tid]
        if stale:
            print(f"[Registry] Flushed {len(stale)} stale mappings, {len(self.botsort_to_slot)} remain")

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int):
        """Build roster: every new BoT-SORT ID = new permanent slot."""
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                crop = self._crop(frame, t)
                if crop is not None:
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"],
                        self._color_hist(crop),
                        alpha=0.7
                    )
            else:
                crop = self._crop(frame, t)
                if crop is None:
                    continue
                hist = self._color_hist(crop)
                if self.next_slot <= self.max_players:
                    slot_id = self.next_slot
                    self.next_slot += 1
                    self.slots[slot_id] = {
                        "hist": hist,
                        "first_seen": video_frame,
                        "last_bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                    }
                    self.botsort_to_slot[tid] = slot_id

    def _match(self, frame: np.ndarray, tracks: List, video_frame: int):
        """
        After freeze: match every BoT-SORT track to a permanent slot.
        Known mappings kept. New BoT-SORT IDs matched by kit color.
        """
        # Collect which slots are actively used by current tracks
        used_slots = set()
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                used_slots.add(self.botsort_to_slot[tid])

        # Match all tracks
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                # Known track — update appearance
                slot_id = self.botsort_to_slot[tid]
                crop = self._crop(frame, t)
                if crop is not None:
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"],
                        self._color_hist(crop),
                        alpha=0.85
                    )
                    self.slots[slot_id]["last_bbox"] = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
            else:
                # New BoT-SORT ID — find best slot by kit color
                crop = self._crop(frame, t)
                if crop is None:
                    continue
                hist = self._color_hist(crop)
                # Lower threshold (0.4) to aggressively re-match after cuts/pans
                matched_slot = self._find_slot(hist, exclude_slots=used_slots, threshold=0.4)
                if matched_slot is not None:
                    self.botsort_to_slot[tid] = matched_slot
                    used_slots.add(matched_slot)
                    print(f"[Registry] Frame {video_frame}: BoT-SORT {tid} → slot P{matched_slot} (re-matched)")

    def _find_slot(self, hist: np.ndarray, exclude_slots: set = None,
                   threshold: float = 0.55) -> Optional[int]:
        """Find best matching slot by cosine similarity."""
        exclude_slots = exclude_slots or set()
        best_score = threshold
        best_slot = None
        for slot_id, data in self.slots.items():
            if slot_id in exclude_slots:
                continue
            score = float(np.dot(hist, data["hist"]))
            if score > best_score:
                best_score = score
                best_slot = slot_id
        return best_slot

    def _crop(self, frame: np.ndarray, t) -> Optional[np.ndarray]:
        """Crop torso region from player bbox."""
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return None
        mid_y = (y2 - y1) // 4
        crop = frame[y1 + mid_y: y2 - mid_y, x1:x2]
        return crop if crop.size > 0 else None

    def _color_hist(self, crop: np.ndarray) -> np.ndarray:
        """Normalized HSV color histogram (hue + saturation)."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist = np.concatenate([h, s])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _blend_hist(self, old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average of histogram."""
        blended = alpha * old + (1 - alpha) * new
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 0 else blended


class VLMStateMachine:
    """
    Game state detection + permanent player registry.

    Key behaviors:
    - PLAY: tracker runs normally, registry maps BoT-SORT IDs → permanent slots
    - BENCH_SHOT: tracker is SKIPPED entirely (no dets fed), tracks frozen
    - bench→play transition: stale mappings flushed, new IDs re-matched to slots
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.registry = PlayerRegistry(max_players=30)
        self.frame_counter = 0
        self._bench_to_play = False  # flag: just transitioned back to play
        print("[VLM] State machine + permanent player registry ready")

    def analyze(self, frame: np.ndarray, video_frame: int) -> GameState:
        """Classify frame state every 10 frames."""
        self.frame_counter += 1
        if self.frame_counter % 10 != 0:
            return self.state

        self.prev_state = self.state
        self.state = self._classify(frame)

        if self.state != self.prev_state:
            print(f"[VLM] Frame {video_frame}: {self.prev_state.value} → {self.state.value}")
            if self.prev_state == GameState.BENCH_SHOT and self.state == GameState.PLAY:
                self._bench_to_play = True

        return self.state

    def check_bench_to_play(self) -> bool:
        """Check and consume the bench→play transition flag."""
        if self._bench_to_play:
            self._bench_to_play = False
            return True
        return False

    def _classify(self, frame: np.ndarray) -> GameState:
        """Green field ratio < 25% = bench/cutaway."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = mask.sum() / (frame.shape[0] * frame.shape[1] * 255)
        return GameState.BENCH_SHOT if green_ratio < 0.25 else GameState.PLAY

    def get_id_remap(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        """Returns {botsort_tid: permanent_slot_id} for this frame."""
        # After bench→play: flush stale mappings so new IDs get re-matched
        if self.check_bench_to_play() and len(tracks) > 0:
            active_tids = set(int(t[4]) for t in tracks if len(t) >= 5)
            self.registry.flush_stale_mappings(active_tids)

        return self.registry.update(frame, tracks, video_frame)
