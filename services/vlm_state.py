import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Optional


class GameState(Enum):
    PLAY = "play"
    BENCH_SHOT = "bench_shot"


def _hungarian(cost: np.ndarray):
    """Greedy min-cost assignment for small matrices. Returns (rows, cols)."""
    n_rows, n_cols = cost.shape
    rows, cols = [], []
    used_r, used_c = set(), set()
    pairs = sorted(((cost[i, j], i, j) for i in range(n_rows) for j in range(n_cols)))
    for c, r, col in pairs:
        if r not in used_r and col not in used_c:
            rows.append(r)
            cols.append(col)
            used_r.add(r)
            used_c.add(col)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


class PlayerRegistry:
    """
    Permanent player roster.

    Identity logic:
    - COLOR → determines TEAM (A vs B), not individual player
    - POSITION → determines individual player within team
    - YOLO CLASS → filters referees (class 3)

    Registration (first 30 frames):
      Every BoT-SORT track gets a slot. Tracks with YOLO class=3 (referee)
      are excluded. After freeze, K-means clusters slots into 2 teams.

    Matching (post-freeze, normal play):
      Known BoT-SORT IDs keep their slot. New IDs are matched using
      Hungarian on SPATIAL distance within the SAME TEAM. Color is only
      used to determine team membership.

    Post-bench recovery:
      Hungarian on color similarity (positions have reset).
    """

    REFEREE_CLASS = 3

    def __init__(self, max_players=30):
        self.max_players = max_players
        self.slots: Dict[int, dict] = {}
        self.botsort_to_slot: Dict[int, int] = {}
        self.next_slot = 1
        self.is_frozen = False
        self._frames_seen = 0
        self._registration_frames = 30
        self._post_bench = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        if len(tracks) == 0:
            return dict(self.botsort_to_slot)

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(frame, tracks, video_frame)
            if self._frames_seen >= self._registration_frames:
                self._freeze()
        else:
            if self._post_bench:
                self._match_post_bench(frame, tracks, video_frame)
                self._post_bench = False
            else:
                self._match_spatial(frame, tracks, video_frame)

        return dict(self.botsort_to_slot)

    def flush_stale_mappings(self, active_tids: set):
        stale = [tid for tid in self.botsort_to_slot if tid not in active_tids]
        for tid in stale:
            del self.botsort_to_slot[tid]
        if stale:
            print(f"[Registry] Flushed {len(stale)} stale, {len(self.botsort_to_slot)} remain")
        self._post_bench = True

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int):
        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])

            # Skip referees by YOLO class
            if cls == self.REFEREE_CLASS:
                continue

            bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            crop = self._crop(frame, t)
            if crop is None:
                continue
            hist = self._color_hist(crop)

            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                s = self.slots[slot_id]
                s["hist"] = self._blend_hist(s["hist"], hist, alpha=0.7)
                s["last_cx"] = cx
                s["last_cy"] = cy
                s["last_bbox"] = bbox
                s["n_updates"] += 1
            else:
                if self.next_slot <= self.max_players:
                    slot_id = self.next_slot
                    self.next_slot += 1
                    self.slots[slot_id] = {
                        "hist": hist,
                        "team": "UNK",
                        "first_seen": video_frame,
                        "last_seen": video_frame,
                        "last_bbox": bbox,
                        "last_cx": cx,
                        "last_cy": cy,
                        "n_updates": 1,
                    }
                    self.botsort_to_slot[tid] = slot_id

    # ------------------------------------------------------------------
    # Freeze + team clustering
    # ------------------------------------------------------------------

    def _freeze(self):
        self.is_frozen = True
        self._assign_teams()
        ta = sum(1 for s in self.slots.values() if s["team"] == "A")
        tb = sum(1 for s in self.slots.values() if s["team"] == "B")
        print(f"[Registry] Frozen: {len(self.slots)} slots | A={ta} B={tb}")

    def _assign_teams(self):
        """K-means (k=2) on accumulated color histograms."""
        sids = list(self.slots.keys())
        if len(sids) < 4:
            return

        hists = np.array([self.slots[s]["hist"] for s in sids])
        n = len(hists)

        # Pairwise cosine distance
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - float(np.dot(hists[i], hists[j]))
                dist[i, j] = d
                dist[j, i] = d

        # Init: most dissimilar pair
        idx = np.unravel_index(np.argmax(dist), dist.shape)
        c0, c1 = hists[idx[0]].copy(), hists[idx[1]].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(5):
            for i in range(n):
                d0 = 1.0 - float(np.dot(hists[i], c0))
                d1 = 1.0 - float(np.dot(hists[i], c1))
                labels[i] = 0 if d0 < d1 else 1
            m0, m1 = labels == 0, labels == 1
            if m0.sum() > 0:
                c0 = hists[m0].mean(axis=0)
                norm = np.linalg.norm(c0)
                if norm > 0: c0 /= norm
            if m1.sum() > 0:
                c1 = hists[m1].mean(axis=0)
                norm = np.linalg.norm(c1)
                if norm > 0: c1 /= norm

        for i, sid in enumerate(sids):
            self.slots[sid]["team"] = "A" if labels[i] == 0 else "B"

        # Store centroids for runtime team detection
        self._centroid_a = c0
        self._centroid_b = c1

    # ------------------------------------------------------------------
    # Matching — spatial (normal play)
    # ------------------------------------------------------------------

    def _match_spatial(self, frame: np.ndarray, tracks: List, video_frame: int):
        """
        Known tracks: update position + appearance.
        Unknown tracks: Hungarian on spatial distance, constrained by team.

        This is the key insight: within the same team, players are distinguished
        by POSITION, not color (all same-team players look alike in torso crops).
        """
        used_slots: set = set()
        unknown = []

        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])

            if cls == self.REFEREE_CLASS:
                continue

            bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                s = self.slots[slot_id]
                # Update position + appearance
                crop = self._crop(frame, t)
                if crop is not None:
                    hist = self._color_hist(crop)
                    s["hist"] = self._blend_hist(s["hist"], hist, alpha=0.85)
                s["last_cx"] = cx
                s["last_cy"] = cy
                s["last_bbox"] = bbox
                s["last_seen"] = video_frame
                used_slots.add(slot_id)
            else:
                crop = self._crop(frame, t)
                if crop is not None:
                    hist = self._color_hist(crop)
                    team = self._detect_team(hist)
                    unknown.append((t, tid, cx, cy, hist, team))

        if not unknown:
            return

        # Available slots (not currently used by active tracks)
        available = [(sid, self.slots[sid]) for sid in self.slots
                     if sid not in used_slots]
        if not available:
            return

        # Build cost matrix: spatial distance, blocked by team mismatch
        n_t = len(unknown)
        n_s = len(available)
        cost = np.full((n_t, n_s), 1e6)

        for i, (t, tid, cx, cy, hist, team) in enumerate(unknown):
            for j, (sid, slot) in enumerate(available):
                # Team gate
                st = slot["team"]
                if st in ("A", "B") and team in ("A", "B") and st != team:
                    continue

                # Spatial distance (pixels)
                dx = cx - slot["last_cx"]
                dy = cy - slot["last_cy"]
                dist = np.sqrt(dx * dx + dy * dy)

                # Also check color similarity as tiebreaker
                color_sim = float(np.dot(hist, slot["hist"]))

                # Combined: spatial distance penalized if color is bad
                # Max reasonable movement between frames: ~200px
                if dist < 300 and color_sim > 0.3:
                    cost[i, j] = dist * (2.0 - color_sim)  # lower color_sim = higher cost

        rows, cols = _hungarian(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < 500:  # reasonable spatial distance
                _, tid, cx, cy, hist, team = unknown[r]
                sid = available[c][0]
                self.botsort_to_slot[tid] = sid
                used_slots.add(sid)
                self.slots[sid]["last_cx"] = cx
                self.slots[sid]["last_cy"] = cy
                self.slots[sid]["last_seen"] = video_frame

    # ------------------------------------------------------------------
    # Matching — post-bench (color-based, positions have reset)
    # ------------------------------------------------------------------

    def _match_post_bench(self, frame: np.ndarray, tracks: List, video_frame: int):
        """After bench cut, positions are meaningless. Use color + Hungarian."""
        unknown = []
        used_slots: set = set()

        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            if cls == self.REFEREE_CLASS:
                continue

            bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if tid in self.botsort_to_slot:
                used_slots.add(self.botsort_to_slot[tid])
                crop = self._crop(frame, t)
                if crop is not None:
                    slot_id = self.botsort_to_slot[tid]
                    hist = self._color_hist(crop)
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"], hist, alpha=0.85
                    )
                    self.slots[slot_id]["last_cx"] = cx
                    self.slots[slot_id]["last_cy"] = cy
                    self.slots[slot_id]["last_seen"] = video_frame
            else:
                crop = self._crop(frame, t)
                if crop is not None:
                    hist = self._color_hist(crop)
                    team = self._detect_team(hist)
                    unknown.append((t, tid, cx, cy, hist, team))

        if not unknown:
            return

        available = [(sid, self.slots[sid]) for sid in self.slots
                     if sid not in used_slots]
        if not available:
            return

        # Color-based cost matrix with team gating
        n_t = len(unknown)
        n_s = len(available)
        cost = np.ones((n_t, n_s))

        for i, (t, tid, cx, cy, hist, team) in enumerate(unknown):
            for j, (sid, slot) in enumerate(available):
                st = slot["team"]
                if st in ("A", "B") and team in ("A", "B") and st != team:
                    cost[i, j] = 1.0
                    continue
                sim = float(np.dot(hist, slot["hist"]))
                cost[i, j] = 1.0 - sim

        rows, cols = _hungarian(cost)
        matched = 0
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.55:
                _, tid, cx, cy, hist, team = unknown[r]
                sid = available[c][0]
                self.botsort_to_slot[tid] = sid
                used_slots.add(sid)
                self.slots[sid]["last_cx"] = cx
                self.slots[sid]["last_cy"] = cy
                self.slots[sid]["last_seen"] = video_frame
                matched += 1

        print(f"[Registry] Post-bench: matched {matched}/{n_t} tracks to {n_s} slots")

    # ------------------------------------------------------------------
    # Team detection at runtime
    # ------------------------------------------------------------------

    def _detect_team(self, hist: np.ndarray) -> str:
        if not hasattr(self, '_centroid_a'):
            return "UNK"
        sim_a = float(np.dot(hist, self._centroid_a))
        sim_b = float(np.dot(hist, self._centroid_b))
        if max(sim_a, sim_b) < 0.4:
            return "UNK"
        return "A" if sim_a > sim_b else "B"

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------

    def _crop(self, frame: np.ndarray, t) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return None
        h = y2 - y1
        y_start = y1 + h // 4
        y_end = y2 - h // 4
        if y_end <= y_start + 3:
            return None
        crop = frame[y_start:y_end, x1:x2]
        return crop if crop.size > 0 else None

    def _color_hist(self, crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist = np.concatenate([h, s])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _blend_hist(self, old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        blended = alpha * old + (1 - alpha) * new
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 0 else blended


class VLMStateMachine:
    def __init__(self, device="cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.registry = PlayerRegistry(max_players=30)
        self.frame_counter = 0
        self._bench_to_play = False
        print("[VLM] State machine + permanent player registry ready")

    def analyze(self, frame: np.ndarray, video_frame: int) -> GameState:
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
        if self._bench_to_play:
            self._bench_to_play = False
            return True
        return False

    def _classify(self, frame: np.ndarray) -> GameState:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = mask.sum() / (frame.shape[0] * frame.shape[1] * 255)
        return GameState.BENCH_SHOT if green_ratio < 0.25 else GameState.PLAY

    def get_id_remap(self, frame: np.ndarray, tracks: List, video_frame: int) -> Dict[int, int]:
        if self.check_bench_to_play() and len(tracks) > 0:
            active_tids = set(int(t[4]) for t in tracks if len(t) >= 5)
            self.registry.flush_stale_mappings(active_tids)
        return self.registry.update(frame, tracks, video_frame)
