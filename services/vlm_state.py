import cv2
import numpy as np
from collections import deque
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


REFEREE_CLASS = 3


def _is_referee_by_color(crop: np.ndarray) -> bool:
    """Orange/yellow kit = referee. Returns True if referee-colored."""
    if crop is None or crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([25, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([25, 80, 80]), np.array([40, 255, 255]))
    orange_r = orange.sum() / (crop.shape[0] * crop.shape[1] * 255 + 1e-6)
    yellow_r = yellow.sum() / (crop.shape[0] * crop.shape[1] * 255 + 1e-6)
    return orange_r > 0.12 or yellow_r > 0.12


class PlayerRegistry:
    """
    Permanent player roster with ReID embedding-based identity matching.

    Identity signals:
    - ReID embeddings (from BoT-SORT's OSNet) → individual player identity
    - Color histograms → team classification (A vs B) only
    - YOLO class → referee exclusion (class 3)

    Registration (first 30 frames):
      Every non-referee BoT-SORT track gets a permanent slot.
      Embeddings and color histograms are accumulated.

    Freeze:
      K-means on color histograms → team A/B labels.
      Embedding banks are ready for matching.

    Normal play (post-freeze):
      Known tracks keep their slot. New tracks matched via Hungarian
      on ReID embedding cosine distance, team-gated, strict 1:1.

    Post-bench:
      All mappings flushed. Hungarian on ReID embeddings to re-match
      new BoT-SORT IDs to the same permanent player slots.
    """

    def __init__(self, max_players=30):
        self.max_players = max_players
        self.slots: Dict[int, dict] = {}              # slot_id → metadata
        self.slot_embeddings: Dict[int, deque] = {}    # slot_id → deque of ReID vectors
        self.slot_mean_embed: Dict[int, np.ndarray] = {}  # slot_id → L2-norm mean
        self.botsort_to_slot: Dict[int, int] = {}      # botsort_tid → slot_id
        self.next_slot = 1
        self.is_frozen = False
        self._frames_seen = 0
        self._registration_frames = 30
        self._post_bench = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, tracks: List, video_frame: int,
               embed_map: Dict[int, np.ndarray] = None) -> Dict[int, int]:
        if embed_map is None:
            embed_map = {}

        if len(tracks) == 0:
            return dict(self.botsort_to_slot)

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(frame, tracks, video_frame, embed_map)
            if self._frames_seen >= self._registration_frames:
                self._freeze()
        else:
            if self._post_bench:
                self._match_post_bench(frame, tracks, video_frame, embed_map)
                self._post_bench = False
            else:
                self._match_reid(frame, tracks, video_frame, embed_map)

        return dict(self.botsort_to_slot)

    def flush_stale_mappings(self, active_tids: set):
        stale = [tid for tid in self.botsort_to_slot if tid not in active_tids]
        for tid in stale:
            del self.botsort_to_slot[tid]
        if stale:
            print(f"[Registry] Flushed {len(stale)} stale, {len(self.botsort_to_slot)} remain")
        self._post_bench = True

    # ------------------------------------------------------------------
    # Registration (first 30 frames)
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int,
                  embed_map: Dict[int, np.ndarray]):
        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])

            # Skip referees (YOLO class + color)
            if cls == REFEREE_CLASS:
                continue
            crop = self._crop(frame, t)
            if _is_referee_by_color(crop):
                continue

            if tid in self.botsort_to_slot:
                # Update existing slot
                slot_id = self.botsort_to_slot[tid]
                if crop is not None:
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"],
                        self._color_hist(crop), alpha=0.7
                    )
                    self.slots[slot_id]["n_updates"] += 1
                # Update embedding bank
                if tid in embed_map:
                    self._update_slot_embedding(slot_id, embed_map[tid])
            else:
                # New track → new slot (crop already computed above)
                if crop is None:
                    continue
                hist = self._color_hist(crop)

                if self.next_slot <= self.max_players:
                    slot_id = self.next_slot
                    self.next_slot += 1
                    bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                    self.slots[slot_id] = {
                        "hist": hist,
                        "team": "UNK",
                        "first_seen": video_frame,
                        "last_seen": video_frame,
                        "last_bbox": bbox,
                        "n_updates": 1,
                    }
                    self.botsort_to_slot[tid] = slot_id
                    # Store initial embedding
                    if tid in embed_map:
                        self._update_slot_embedding(slot_id, embed_map[tid])

    # ------------------------------------------------------------------
    # Freeze + team clustering
    # ------------------------------------------------------------------

    def _freeze(self):
        self.is_frozen = True
        self._assign_teams()
        ta = sum(1 for s in self.slots.values() if s["team"] == "A")
        tb = sum(1 for s in self.slots.values() if s["team"] == "B")
        n_with_embed = sum(1 for sid in self.slots if sid in self.slot_mean_embed)
        print(f"[Registry] Frozen: {len(self.slots)} slots | A={ta} B={tb} | "
              f"{n_with_embed} with ReID embeddings")

    def _assign_teams(self):
        """K-means (k=2) on color histograms for team classification."""
        sids = list(self.slots.keys())
        if len(sids) < 4:
            return

        hists = np.array([self.slots[s]["hist"] for s in sids])
        n = len(hists)

        # Pairwise cosine distance → find most dissimilar pair as seeds
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - float(np.dot(hists[i], hists[j]))
                dist[i, j] = d
                dist[j, i] = d

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

        self._centroid_a = c0
        self._centroid_b = c1

    # ------------------------------------------------------------------
    # Normal play matching (ReID embedding-based)
    # ------------------------------------------------------------------

    def _match_reid(self, frame: np.ndarray, tracks: List, video_frame: int,
                    embed_map: Dict[int, np.ndarray]):
        """
        Known tracks → keep slot, update embedding bank.
        New tracks → Hungarian on ReID cosine distance, team-gated, strict 1:1.
        """
        used_slots: set = set()
        unknown = []

        # Pass 1: known tracks keep their slot
        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            if cls == REFEREE_CLASS:
                continue

            if tid in self.botsort_to_slot:
                slot_id = self.botsort_to_slot[tid]

                # Check for slot collision (two tids claiming same slot)
                if slot_id in used_slots:
                    # Collision: evict this tid, let it re-match as unknown
                    del self.botsort_to_slot[tid]
                    if tid in embed_map:
                        team = self._detect_team_from_hist(
                            self._color_hist(self._crop(frame, t)) if self._crop(frame, t) is not None
                            else np.zeros(52))
                        unknown.append((t, tid, embed_map.get(tid), team))
                    continue

                used_slots.add(slot_id)

                # Update embedding bank
                if tid in embed_map:
                    self._update_slot_embedding(slot_id, embed_map[tid])

                # Update color histogram + position
                crop = self._crop(frame, t)
                if crop is not None:
                    hist = self._color_hist(crop)
                    self.slots[slot_id]["hist"] = self._blend_hist(
                        self.slots[slot_id]["hist"], hist, alpha=0.85)
                self.slots[slot_id]["last_bbox"] = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                self.slots[slot_id]["last_seen"] = video_frame
            else:
                # Unknown track — collect for Hungarian matching
                if cls == REFEREE_CLASS:
                    continue
                crop = self._crop(frame, t)
                if _is_referee_by_color(crop):
                    continue
                team = "UNK"
                if crop is not None:
                    hist = self._color_hist(crop)
                    team = self._detect_team_from_hist(hist)
                unknown.append((t, tid, embed_map.get(tid), team))

        if not unknown:
            return

        # Pass 2: Hungarian on ReID embeddings for unknown tracks
        available = [(sid, self.slots[sid]) for sid in self.slots
                     if sid not in used_slots]
        if not available:
            return

        n_t = len(unknown)
        n_s = len(available)
        cost = np.ones((n_t, n_s))

        for i, (t, tid, emb, team) in enumerate(unknown):
            for j, (sid, slot) in enumerate(available):
                # Team gate
                st = slot["team"]
                if st in ("A", "B") and team in ("A", "B") and st != team:
                    cost[i, j] = 1.0
                    continue

                # ReID embedding distance (primary signal)
                if emb is not None and sid in self.slot_mean_embed:
                    cost[i, j] = 1.0 - float(np.dot(emb, self.slot_mean_embed[sid]))
                else:
                    # Fallback: color histogram similarity
                    crop = self._crop(frame, t)
                    if crop is not None:
                        hist = self._color_hist(crop)
                        cost[i, j] = 1.0 - float(np.dot(hist, slot["hist"]))

        if len(cost) == 0:
            return

        rows, cols = _hungarian(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.5:  # cosine similarity > 0.5
                _, tid, emb, team = unknown[r]
                sid = available[c][0]
                self.botsort_to_slot[tid] = sid
                used_slots.add(sid)
                self.slots[sid]["last_seen"] = video_frame
                if emb is not None:
                    self._update_slot_embedding(sid, emb)

    # ------------------------------------------------------------------
    # Post-bench recovery (ReID embedding-based)
    # ------------------------------------------------------------------

    def _match_post_bench(self, frame: np.ndarray, tracks: List, video_frame: int,
                          embed_map: Dict[int, np.ndarray]):
        """All mappings flushed. Re-match all tracks to slots via ReID embeddings."""
        unknown = []
        used_slots: set = set()

        # Keep any surviving mappings
        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            if cls == REFEREE_CLASS:
                continue
            if _is_referee_by_color(self._crop(frame, t)):
                continue

            if tid in self.botsort_to_slot:
                used_slots.add(self.botsort_to_slot[tid])
                if tid in embed_map:
                    self._update_slot_embedding(self.botsort_to_slot[tid], embed_map[tid])
            else:
                crop = self._crop(frame, t)
                if _is_referee_by_color(crop):
                    continue
                team = "UNK"
                if crop is not None:
                    hist = self._color_hist(crop)
                    team = self._detect_team_from_hist(hist)
                unknown.append((t, tid, embed_map.get(tid), team))

        if not unknown:
            return

        available = [(sid, self.slots[sid]) for sid in self.slots
                     if sid not in used_slots]
        if not available:
            return

        n_t = len(unknown)
        n_s = len(available)
        cost = np.ones((n_t, n_s))

        for i, (t, tid, emb, team) in enumerate(unknown):
            for j, (sid, slot) in enumerate(available):
                st = slot["team"]
                if st in ("A", "B") and team in ("A", "B") and st != team:
                    cost[i, j] = 1.0
                    continue

                if emb is not None and sid in self.slot_mean_embed:
                    cost[i, j] = 1.0 - float(np.dot(emb, self.slot_mean_embed[sid]))
                else:
                    crop = self._crop(frame, t)
                    if crop is not None:
                        hist = self._color_hist(crop)
                        cost[i, j] = 1.0 - float(np.dot(hist, slot["hist"]))

        rows, cols = _hungarian(cost)
        matched = 0
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.6:  # relaxed threshold (sim > 0.4)
                _, tid, emb, team = unknown[r]
                sid = available[c][0]
                self.botsort_to_slot[tid] = sid
                used_slots.add(sid)
                self.slots[sid]["last_seen"] = video_frame
                if emb is not None:
                    self._update_slot_embedding(sid, emb)
                matched += 1

        print(f"[Registry] Post-bench: matched {matched}/{n_t} tracks to {n_s} slots")

    # ------------------------------------------------------------------
    # Team detection
    # ------------------------------------------------------------------

    def _detect_team_from_hist(self, hist: np.ndarray) -> str:
        if not hasattr(self, '_centroid_a'):
            return "UNK"
        sim_a = float(np.dot(hist, self._centroid_a))
        sim_b = float(np.dot(hist, self._centroid_b))
        # Always assign — no UNK after freeze, just pick best centroid
        return "A" if sim_a > sim_b else "B"

    # ------------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------------

    def _update_slot_embedding(self, slot_id: int, embedding: np.ndarray):
        """Append embedding to slot's bank and recompute mean."""
        if slot_id not in self.slot_embeddings:
            self.slot_embeddings[slot_id] = deque(maxlen=20)
        self.slot_embeddings[slot_id].append(embedding.copy())
        bank = np.array(self.slot_embeddings[slot_id])
        mean = bank.mean(axis=0)
        norm = np.linalg.norm(mean)
        self.slot_mean_embed[slot_id] = mean / norm if norm > 0 else mean

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

    def get_id_remap(self, frame: np.ndarray, tracks: List, video_frame: int,
                     embed_map: Dict[int, np.ndarray] = None) -> Dict[int, int]:
        if self.check_bench_to_play() and len(tracks) > 0:
            active_tids = set(int(t[4]) for t in tracks if len(t) >= 5)
            self.registry.flush_stale_mappings(active_tids)
        return self.registry.update(frame, tracks, video_frame, embed_map)
