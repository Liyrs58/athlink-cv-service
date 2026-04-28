import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Optional


def _hungarian(cost: np.ndarray):
    """
    Greedy approximation of Hungarian algorithm for small matrices.
    Returns (row_indices, col_indices) of optimal assignment.
    """
    n_rows, n_cols = cost.shape
    row_idx, col_idx = [], []
    used_rows, used_cols = set(), set()

    # Flatten and sort by cost ascending
    indices = []
    for i in range(n_rows):
        for j in range(n_cols):
            indices.append((cost[i, j], i, j))
    indices.sort()

    for c, r, col in indices:
        if r in used_rows or col in used_cols:
            continue
        row_idx.append(r)
        col_idx.append(col)
        used_rows.add(r)
        used_cols.add(col)

    return np.array(row_idx), np.array(col_idx)


class GameState(Enum):
    PLAY = "play"
    BENCH_SHOT = "bench_shot"


class PlayerRegistry:
    """
    Permanent player roster for broadcast football tracking.

    Phase 1 — Registration (first 30 PLAY frames):
      Every BoT-SORT track gets its own slot. ALL tracks registered (no filtering).
      Appearance histograms are accumulated over multiple frames for stability.

    Phase 2 — Freeze:
      Team labels are assigned by clustering all slot histograms into 2 groups
      (using cosine distance). This is far more reliable than single-frame hue
      thresholds.

    Phase 3 — Matching (ongoing):
      New BoT-SORT IDs are matched to slots using:
        1. Cosine similarity of HSV histograms
        2. Team constraint (A slot ≠ B detection)
        3. Strict one-slot-per-track exclusion
      After bench→play transitions, Hungarian algorithm finds optimal assignment
      across all new tracks and available slots.
    """

    def __init__(self, max_players=30):
        self.max_players = max_players
        self.slots: Dict[int, dict] = {}
        self.botsort_to_slot: Dict[int, int] = {}
        self.next_slot = 1
        self.is_frozen = False
        self._frames_seen = 0
        self._registration_frames = 30
        self._post_bench = False  # use Hungarian for first match after bench

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
                self._match_hungarian(frame, tracks, video_frame)
                self._post_bench = False
            else:
                self._match(frame, tracks, video_frame)

        return dict(self.botsort_to_slot)

    def flush_stale_mappings(self, active_tids: set):
        stale = [tid for tid in self.botsort_to_slot if tid not in active_tids]
        for tid in stale:
            del self.botsort_to_slot[tid]
        if stale:
            print(f"[Registry] Flushed {len(stale)} stale, {len(self.botsort_to_slot)} remain")
        self._post_bench = True  # next match uses Hungarian

    # ------------------------------------------------------------------
    # Registration — no filtering, register everything
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int):
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])

            crop = self._crop(frame, t)
            if crop is None:
                continue
            hist = self._color_hist(crop)

            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                self.slots[slot_id]["hist"] = self._blend_hist(
                    self.slots[slot_id]["hist"], hist, alpha=0.7
                )
                self.slots[slot_id]["n_updates"] += 1
            else:
                if self.next_slot <= self.max_players:
                    slot_id = self.next_slot
                    self.next_slot += 1
                    self.slots[slot_id] = {
                        "hist": hist,
                        "team": "UNK",
                        "first_seen": video_frame,
                        "last_seen": video_frame,
                        "last_bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                        "n_updates": 1,
                    }
                    self.botsort_to_slot[tid] = slot_id

    # ------------------------------------------------------------------
    # Freeze — cluster slots into 2 teams
    # ------------------------------------------------------------------

    def _freeze(self):
        self.is_frozen = True
        self._assign_teams()
        team_a = sum(1 for s in self.slots.values() if s["team"] == "A")
        team_b = sum(1 for s in self.slots.values() if s["team"] == "B")
        ref = sum(1 for s in self.slots.values() if s["team"] == "REF")
        print(f"[Registry] Frozen: {len(self.slots)} slots | "
              f"A={team_a} B={team_b} REF={ref}")

    def _assign_teams(self):
        """
        Cluster slots into 2 teams using K-means on HSV histograms.
        Slots with very few updates (≤2) or very different from both
        clusters are marked REF.
        """
        slot_ids = list(self.slots.keys())
        if len(slot_ids) < 4:
            return

        hists = np.array([self.slots[sid]["hist"] for sid in slot_ids])

        # K-means with 2 clusters (team A and team B)
        # Initialize with the two most dissimilar histograms
        n = len(hists)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - float(np.dot(hists[i], hists[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Find most dissimilar pair
        idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        c0, c1 = hists[idx[0]].copy(), hists[idx[1]].copy()

        # 5 iterations of K-means
        labels = np.zeros(n, dtype=int)
        for _ in range(5):
            for i in range(n):
                d0 = 1.0 - float(np.dot(hists[i], c0))
                d1 = 1.0 - float(np.dot(hists[i], c1))
                labels[i] = 0 if d0 < d1 else 1

            # Update centroids
            mask0 = labels == 0
            mask1 = labels == 1
            if mask0.sum() > 0:
                c0 = hists[mask0].mean(axis=0)
                norm = np.linalg.norm(c0)
                if norm > 0:
                    c0 /= norm
            if mask1.sum() > 0:
                c1 = hists[mask1].mean(axis=0)
                norm = np.linalg.norm(c1)
                if norm > 0:
                    c1 /= norm

        # Assign teams; outliers (low similarity to both centroids) = REF
        for i, sid in enumerate(slot_ids):
            sim0 = float(np.dot(hists[i], c0))
            sim1 = float(np.dot(hists[i], c1))
            best_sim = max(sim0, sim1)

            # Low update count or poor fit to either cluster = likely referee
            if self.slots[sid]["n_updates"] <= 2 or best_sim < 0.5:
                self.slots[sid]["team"] = "REF"
            elif labels[i] == 0:
                self.slots[sid]["team"] = "A"
            else:
                self.slots[sid]["team"] = "B"

    # ------------------------------------------------------------------
    # Matching — greedy (normal play)
    # ------------------------------------------------------------------

    def _match(self, frame: np.ndarray, tracks: List, video_frame: int):
        # Build reverse map: slot → tid (for active tracks only)
        slot_to_tid: Dict[int, int] = {}
        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                slot_to_tid[slot_id] = tid

        used_slots = set(slot_to_tid.keys())

        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])

            crop = self._crop(frame, t)
            if crop is None:
                continue
            hist = self._color_hist(crop)

            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]
                slot = self.slots[slot_id]
                # Update appearance
                slot["hist"] = self._blend_hist(slot["hist"], hist, alpha=0.85)
                slot["last_bbox"] = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                slot["last_seen"] = video_frame
            else:
                # New BoT-SORT ID — find best available slot
                team = self._detect_team_from_hist(hist)
                matched = self._find_slot(hist, team, exclude_slots=used_slots, threshold=0.45)
                if matched is not None:
                    self.botsort_to_slot[tid] = matched
                    used_slots.add(matched)
                    self.slots[matched]["last_seen"] = video_frame

    # ------------------------------------------------------------------
    # Matching — Hungarian (after bench→play transition)
    # ------------------------------------------------------------------

    def _match_hungarian(self, frame: np.ndarray, tracks: List, video_frame: int):
        """
        Optimal assignment of all new tracks to all available slots.
        Uses Hungarian algorithm on cosine distance matrix.
        """
        # Separate known and unknown tracks
        unknown_tracks = []
        used_slots = set()

        for t in tracks:
            if len(t) < 5:
                continue
            tid = int(t[4])
            if tid in self.botsort_to_slot:
                used_slots.add(self.botsort_to_slot[tid])
                # Update appearance
                crop = self._crop(frame, t)
                if crop is not None:
                    slot_id = self.botsort_to_slot[tid]
                    hist = self._color_hist(crop)
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"], hist, alpha=0.85
                    )
                    self.slots[slot_id]["last_seen"] = video_frame
            else:
                crop = self._crop(frame, t)
                if crop is not None:
                    hist = self._color_hist(crop)
                    unknown_tracks.append((t, tid, hist))

        if not unknown_tracks:
            return

        # Available slots (not currently used)
        available = [(sid, data) for sid, data in self.slots.items()
                     if sid not in used_slots and data["team"] != "REF"]

        if not available:
            return

        # Build cost matrix: rows = unknown tracks, cols = available slots
        n_tracks = len(unknown_tracks)
        n_slots = len(available)
        cost = np.ones((n_tracks, n_slots), dtype=float)  # 1.0 = max distance

        for i, (t, tid, hist) in enumerate(unknown_tracks):
            det_team = self._detect_team_from_hist(hist)
            for j, (sid, data) in enumerate(available):
                # Team gate
                slot_team = data["team"]
                if slot_team in ("A", "B") and det_team in ("A", "B") and slot_team != det_team:
                    cost[i, j] = 1.0  # blocked
                    continue
                sim = float(np.dot(hist, data["hist"]))
                cost[i, j] = 1.0 - sim

        # Hungarian assignment
        row_idx, col_idx = _hungarian(cost)

        matched_count = 0
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 0.55:  # similarity > 0.45
                t, tid, hist = unknown_tracks[r]
                sid = available[c][0]
                self.botsort_to_slot[tid] = sid
                used_slots.add(sid)
                self.slots[sid]["last_seen"] = video_frame
                matched_count += 1

        print(f"[Registry] Hungarian: matched {matched_count}/{n_tracks} "
              f"tracks to {n_slots} available slots")

    # ------------------------------------------------------------------
    # Team detection from accumulated histogram
    # ------------------------------------------------------------------

    def _detect_team_from_hist(self, hist: np.ndarray) -> str:
        """
        Quick team detection by comparing to cluster centroids.
        Falls back to UNK if no clear match.
        """
        if not self.is_frozen:
            return "UNK"

        # Compare to all slots, find most similar, inherit its team
        best_sim = 0.0
        best_team = "UNK"
        for data in self.slots.values():
            if data["team"] == "REF":
                continue
            sim = float(np.dot(hist, data["hist"]))
            if sim > best_sim:
                best_sim = sim
                best_team = data["team"]

        return best_team if best_sim > 0.5 else "UNK"

    # ------------------------------------------------------------------
    # Slot search (greedy)
    # ------------------------------------------------------------------

    def _find_slot(self, hist: np.ndarray, team: str,
                   exclude_slots: set = None, threshold: float = 0.45) -> Optional[int]:
        exclude_slots = exclude_slots or set()
        best_score = threshold
        best_slot = None

        for slot_id, data in self.slots.items():
            if slot_id in exclude_slots:
                continue
            if data["team"] == "REF":
                continue

            slot_team = data["team"]
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
    """
    Game state detection + permanent player registry.
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
