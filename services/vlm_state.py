import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GameState(Enum):
    PLAY = "play"
    BENCH_SHOT = "bench_shot"


def _detect_team(hist: np.ndarray) -> str:
    """
    Classify kit into team A, team B, or referee based on dominant hue bin.

    HSV hue bins (36-bin, each bin = 5 degrees):
      - Bin 0-2   (~0-10°):  red/maroon  → team_A  (Aston Villa claret)
      - Bin 3-6   (~15-30°): orange      → other
      - Bin 7-10  (~35-50°): yellow      → referee
      - Bin 14-20 (~70-100°):green       → pitch (should not appear in torso crop)
      - Bin 24-30 (~120-150°):blue/navy  → team_B  (PSG dark blue)
      - Bin 0-1 + low sat   : white/grey → referee or GK

    Returns: "A", "B", "REF", or "UNK"
    """
    hue = hist[:36]        # first 36 values = hue bins
    sat = hist[36:]        # last 16 values = saturation bins

    low_sat = float(sat[:4].sum())   # low saturation = white/grey/black
    total = float(hue.sum()) + 1e-6

    # Dominant hue bin
    dominant_bin = int(np.argmax(hue))
    dominant_strength = float(hue[dominant_bin]) / total

    # Black/dark kit: very low saturation overall
    if low_sat / (total + low_sat) > 0.6:
        return "REF"  # black kit = referee

    # Yellow: bins 7-10 (35-50°)
    yellow = float(hue[7:11].sum()) / total
    if yellow > 0.35:
        return "REF"

    # Red/claret: bins 0-3 and 33-36 (wraps around)
    red = float(hue[0:4].sum() + hue[33:36].sum()) / total
    if red > 0.25:
        return "A"

    # Blue/navy/dark blue: bins 22-30 (110-150°)
    blue = float(hue[22:31].sum()) / total
    if blue > 0.20:
        return "B"

    # White/light grey (high brightness, low sat) — could be GK or away kit
    if low_sat / (total + low_sat) > 0.35:
        return "UNK"

    return "UNK"


class PlayerRegistry:
    """
    Permanent player roster.

    Registration phase (first 30 PLAY frames): every new BoT-SORT ID gets
    its own slot. Referees are excluded.

    After freeze: new BoT-SORT IDs are matched to existing slots by:
      1. Kit color histogram (cosine similarity ≥ 0.45)
      2. Team consistency (A slot never matches B detection)
      3. One slot per active track (strict exclusion)

    This gives stable P1–P22 labels across pans and cuts.
    """

    def __init__(self, max_players=30):
        self.max_players = max_players
        self.slots: Dict[int, dict] = {}              # slot_id → data
        self.botsort_to_slot: Dict[int, int] = {}     # botsort_tid → slot_id
        self.next_slot = 1
        self.is_frozen = False
        self._frames_seen = 0
        self._registration_frames = 30

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        """Called every PLAY frame. Returns {botsort_tid: permanent_slot_id}."""
        if len(tracks) == 0:
            return dict(self.botsort_to_slot)

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(frame, tracks, video_frame)
            if self._frames_seen >= self._registration_frames:
                self.is_frozen = True
                print(f"[Registry] Frozen with {len(self.slots)} slots "
                      f"(teams: A={sum(1 for s in self.slots.values() if s['team']=='A')}, "
                      f"B={sum(1 for s in self.slots.values() if s['team']=='B')}, "
                      f"UNK={sum(1 for s in self.slots.values() if s['team'] not in ('A','B'))})")
        else:
            self._match(frame, tracks, video_frame)

        return dict(self.botsort_to_slot)

    def flush_stale_mappings(self, active_tids: set):
        """Remove mappings for dead BoT-SORT IDs. Called after bench→play."""
        stale = [tid for tid in self.botsort_to_slot if tid not in active_tids]
        for tid in stale:
            del self.botsort_to_slot[tid]
        if stale:
            print(f"[Registry] Flushed {len(stale)} stale mappings, "
                  f"{len(self.botsort_to_slot)} live mappings remain")

    # ------------------------------------------------------------------
    # Registration phase
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int):
        """
        Build roster. Each new BoT-SORT ID → new slot.
        Referees (black/yellow kit) are skipped entirely.
        """
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])

            if tid in self.botsort_to_slot:
                # Update existing slot appearance
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
                team = _detect_team(hist)

                # Skip referees — they must not consume a player slot
                if team == "REF":
                    continue

                if self.next_slot <= self.max_players:
                    slot_id = self.next_slot
                    self.next_slot += 1
                    self.slots[slot_id] = {
                        "hist": hist,
                        "team": team,
                        "first_seen": video_frame,
                        "last_bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                        "last_seen": video_frame,
                    }
                    self.botsort_to_slot[tid] = slot_id

    # ------------------------------------------------------------------
    # Matching phase (post-freeze)
    # ------------------------------------------------------------------

    def _match(self, frame: np.ndarray, tracks: List, video_frame: int):
        """
        After freeze: map every active BoT-SORT track to a permanent slot.

        Rules:
        - Known tid → keep its slot, update appearance
        - Unknown tid → find best slot by color + team, exclude already-used slots
        - Referees → skip (no slot assigned)
        - One slot per active track (strict)
        """
        # Slots already claimed by known tracks this frame
        used_slots: set = set()
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                used_slots.add(self.botsort_to_slot[tid])

        # Process all tracks
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])

            crop = self._crop(frame, t)
            if crop is None:
                continue
            hist = self._color_hist(crop)
            team = _detect_team(hist)

            # Skip referees
            if team == "REF":
                continue

            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                slot = self.slots[slot_id]

                # Team consistency check: if team flipped, force re-match
                if slot["team"] in ("A", "B") and team in ("A", "B") and slot["team"] != team:
                    # Team mismatch — this BoT-SORT ID got swapped to wrong player
                    # Remove stale mapping and re-match below
                    del self.botsort_to_slot[tid]
                    used_slots.discard(slot_id)
                else:
                    # Good — update appearance
                    slot["hist"] = self._blend_hist(slot["hist"], hist, alpha=0.85)
                    slot["last_bbox"] = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                    slot["last_seen"] = video_frame
                    continue

            # Unknown or just de-mapped tid — find best slot
            matched_slot = self._find_slot(hist, team, exclude_slots=used_slots, threshold=0.45)
            if matched_slot is not None:
                self.botsort_to_slot[tid] = matched_slot
                used_slots.add(matched_slot)
                self.slots[matched_slot]["last_seen"] = video_frame

    # ------------------------------------------------------------------
    # Slot search
    # ------------------------------------------------------------------

    def _find_slot(self, hist: np.ndarray, team: str,
                   exclude_slots: set = None, threshold: float = 0.45) -> Optional[int]:
        """
        Best matching slot by cosine similarity + team filter.
        A slot of team A never matches a detection of team B, and vice versa.
        """
        exclude_slots = exclude_slots or set()
        best_score = threshold
        best_slot = None

        for slot_id, data in self.slots.items():
            if slot_id in exclude_slots:
                continue

            # Team gate: reject cross-team matches
            slot_team = data.get("team", "UNK")
            if slot_team in ("A", "B") and team in ("A", "B") and slot_team != team:
                continue

            score = float(np.dot(hist, data["hist"]))
            if score > best_score:
                best_score = score
                best_slot = slot_id

        return best_slot

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------

    def _crop(self, frame: np.ndarray, t) -> Optional[np.ndarray]:
        """Crop torso region (middle 50% vertically) from player bbox."""
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return None
        h = y2 - y1
        # Take middle 50%: skip top 25% (head) and bottom 25% (legs/feet)
        y_start = y1 + h // 4
        y_end = y2 - h // 4
        if y_end <= y_start + 3:
            return None
        crop = frame[y_start:y_end, x1:x2]
        return crop if crop.size > 0 else None

    def _color_hist(self, crop: np.ndarray) -> np.ndarray:
        """Normalized HSV color histogram: 36 hue + 16 saturation bins."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist = np.concatenate([h, s])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _blend_hist(self, old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average histogram blend."""
        blended = alpha * old + (1 - alpha) * new
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 0 else blended


class VLMStateMachine:
    """
    Game state detection + permanent player registry.

    - PLAY: tracker runs, registry maps BoT-SORT IDs → permanent slots
    - BENCH_SHOT: tracker skipped (tracks not fed empty dets)
    - bench→play: stale mappings flushed, new IDs re-matched
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.registry = PlayerRegistry(max_players=30)
        self.frame_counter = 0
        self._bench_to_play = False
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
        """Consume and return the bench→play transition flag."""
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
        if self.check_bench_to_play() and len(tracks) > 0:
            active_tids = set(int(t[4]) for t in tracks if len(t) >= 5)
            self.registry.flush_stale_mappings(active_tids)

        return self.registry.update(frame, tracks, video_frame)
