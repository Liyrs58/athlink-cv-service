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
    - T1 is always the Villa goalkeeper
    - T2 is always the Villa #9
    - etc.
    Even after camera pans, bench shots, or ID switches.
    """

    def __init__(self, max_players=30):
        self.max_players = max_players
        self.slots: Dict[int, dict] = {}       # slot_id -> appearance data
        self.botsort_to_slot: Dict[int, int] = {}  # botsort_tid -> slot_id
        self.next_slot = 1
        self.is_frozen = False                  # True after registration phase
        self._frames_seen = 0
        self._registration_frames = 30         # build roster over first 30 frames

    def update(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        """
        Main entry point. Called every frame.
        Returns: {botsort_tid: permanent_slot_id}

        During registration (first 30 frames): builds roster.
        After registration: matches every track to its slot.
        """
        if len(tracks) == 0:
            return {}

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(frame, tracks, video_frame)
            if self._frames_seen >= self._registration_frames:
                self.is_frozen = True
                print(f"[Registry] Frozen with {len(self.slots)} permanent player slots")
        else:
            self._match(frame, tracks, video_frame)

        return dict(self.botsort_to_slot)

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int):
        """Build roster: assign new slots to unseen tracks."""
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                # Already registered — update appearance
                slot_id = self.botsort_to_slot[tid]
                crop = self._crop(frame, t)
                if crop is not None:
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"],
                        self._color_hist(crop),
                        alpha=0.7  # keep 70% old, 30% new
                    )
            else:
                # New track — try to match to existing slot first
                crop = self._crop(frame, t)
                if crop is None:
                    continue
                hist = self._color_hist(crop)
                matched_slot = self._find_slot(hist, exclude_tids=set(self.botsort_to_slot.values()))
                if matched_slot is not None:
                    self.botsort_to_slot[tid] = matched_slot
                elif self.next_slot <= self.max_players:
                    # New player — create new slot
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
        Tracks that were already matched keep their slot.
        New track IDs (after ID switch/cut) get re-matched by appearance.
        """
        used_slots = set()
        # First pass: keep existing matches that are still valid
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                used_slots.add(self.botsort_to_slot[tid])

        # Second pass: re-match unmatched tracks
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                # Update appearance
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
                # New BoT-SORT ID — find best slot match
                crop = self._crop(frame, t)
                if crop is None:
                    continue
                hist = self._color_hist(crop)
                matched_slot = self._find_slot(hist, exclude_slots=used_slots, threshold=0.5)
                if matched_slot is not None:
                    self.botsort_to_slot[tid] = matched_slot
                    used_slots.add(matched_slot)
                elif self.next_slot <= self.max_players:
                    # Genuinely new player (e.g. sub)
                    slot_id = self.next_slot
                    self.next_slot += 1
                    self.slots[slot_id] = {
                        "hist": hist,
                        "first_seen": video_frame,
                        "last_bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                    }
                    self.botsort_to_slot[tid] = slot_id
                    used_slots.add(slot_id)

    def _find_slot(self, hist: np.ndarray, exclude_slots: set = None,
                   exclude_tids: set = None, threshold: float = 0.55) -> Optional[int]:
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
        """Crop player bbox from frame. Returns None if invalid."""
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return None
        # Use torso region only (middle 50%) for kit color — avoids legs/head
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

    The registry assigns permanent slot IDs (1..N) to players.
    BoT-SORT track IDs are remapped to slot IDs every frame,
    so boxes never jump IDs after pans or cuts.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.registry = PlayerRegistry(max_players=30)
        self.frame_counter = 0
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

        return self.state

    def _classify(self, frame: np.ndarray) -> GameState:
        """Green field ratio < 25% = bench/cutaway."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = mask.sum() / (frame.shape[0] * frame.shape[1] * 255)
        return GameState.BENCH_SHOT if green_ratio < 0.25 else GameState.PLAY

    def get_id_remap(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        """
        Returns {botsort_tid: permanent_slot_id} for this frame.
        Only called during PLAY state.
        """
        return self.registry.update(frame, tracks, video_frame)
