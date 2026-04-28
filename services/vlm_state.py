import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState(Enum):
    PLAY        = "play"
    BENCH_SHOT  = "bench_shot"
    CELEBRATION = "celebration"
    REPLAY      = "replay"
    CROWD       = "crowd"
    CLOSEUP     = "closeup"

    def is_freeze(self) -> bool:
        """True for any scene where the field is not visible → freeze registry."""
        return self in (GameState.BENCH_SHOT, GameState.REPLAY,
                        GameState.CROWD, GameState.CLOSEUP)

    def is_play(self) -> bool:
        return self == GameState.PLAY


# ---------------------------------------------------------------------------
# Slot state
# ---------------------------------------------------------------------------

class SlotState(Enum):
    UNASSIGNED = "unassigned"
    ACTIVE     = "active"
    DORMANT    = "dormant"
    LOST       = "lost"       # dormant for too long — embeddings kept, won't match


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerSlot:
    slot_id: int
    team: str = "UNK"                                        # "A" | "B" | "UNK"
    embeddings: deque = field(default_factory=lambda: deque(maxlen=30))
    mean_embed: Optional[np.ndarray] = None
    hist: Optional[np.ndarray] = None
    last_bbox: Optional[List[float]] = None
    last_seen_frame: int = -1
    active_track_id: Optional[int] = None
    state: SlotState = SlotState.UNASSIGNED
    n_updates: int = 0


@dataclass
class RefereeSlot:
    ref_id: int                                              # 1-based (R1, R2, ...)
    embeddings: deque = field(default_factory=lambda: deque(maxlen=30))
    mean_embed: Optional[np.ndarray] = None
    last_seen_frame: int = -1
    active_track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REFEREE_CLASS = 3
_EMBED_DIM    = 512
_LOST_FRAMES  = 750   # ~30s @ 25fps — dormant slots older than this → LOST


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy min-cost assignment. Returns (row_indices, col_indices)."""
    rows, cols = [], []
    used_r, used_c = set(), set()
    pairs = sorted((cost[i, j], i, j)
                   for i in range(cost.shape[0])
                   for j in range(cost.shape[1]))
    for c, r, col in pairs:
        if r not in used_r and col not in used_c:
            rows.append(r); cols.append(col)
            used_r.add(r);  used_c.add(col)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def _iou(b1: List[float], b2: List[float]) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    return inter / (a + b - inter + 1e-6)


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _is_referee_by_color(crop: Optional[np.ndarray]) -> bool:
    if crop is None or crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([5,  80, 80]), np.array([25, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([25, 80, 80]), np.array([40, 255, 255]))
    px = crop.shape[0] * crop.shape[1] * 255 + 1e-6
    return (orange.sum() / px > 0.12) or (yellow.sum() / px > 0.12)


def _crop_torso(frame: np.ndarray, t) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None
    h = y2 - y1
    ys = y1 + h // 4
    ye = y2 - h // 4
    if ye <= ys + 3:
        return None
    crop = frame[ys:ye, x1:x2]
    return crop if crop.size > 0 else None


def _color_hist(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist = np.concatenate([h, s])
    return _l2norm(hist)


def _blend_hist(old: np.ndarray, new: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    return _l2norm(alpha * old + (1 - alpha) * new)


# ---------------------------------------------------------------------------
# PlayerRegistry — the identity truth
# ---------------------------------------------------------------------------

class PlayerRegistry:
    """
    Authoritative identity system. BotSort track IDs are temporary handles.
    Permanent player slots (P1–P22) are owned here and assigned by identity score.

    Lifecycle:
      REGISTRATION (frames 0–29): every non-referee track → new slot
      FREEZE (frame 30): K-means → team labels
      ACTIVE PLAY: identity engine matches BotSort tids → slots
      BENCH CUT: all slots → DORMANT, all BotSort mappings dropped
      RESUME: full re-identification via ReID + spatial + team gate
    """

    def __init__(self, max_players: int = 30, reg_frames: int = 30):
        self.max_players  = max_players
        self.reg_frames   = reg_frames
        self.slots: Dict[int, PlayerSlot] = {}
        self.botsort_to_slot: Dict[int, int] = {}   # temporary BotSort tid → slot_id
        self.next_slot    = 1
        self.is_frozen    = False
        self._frames_seen = 0
        self._post_bench  = False
        self._centroid_a: Optional[np.ndarray] = None
        self._centroid_b: Optional[np.ndarray] = None
        # Referee registry — separate from P1-P22
        self.ref_slots: Dict[int, RefereeSlot] = {}   # ref_id → slot
        self.botsort_to_ref: Dict[int, int] = {}       # BotSort tid → ref_id
        self.next_ref     = 1

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, tracks: List, video_frame: int,
               embed_map: Dict[int, np.ndarray]) -> Dict[int, int]:
        if len(tracks) == 0:
            return dict(self.botsort_to_slot)

        self._frames_seen += 1

        if not self.is_frozen:
            self._register(frame, tracks, video_frame, embed_map)
            if self._frames_seen >= self.reg_frames:
                self._freeze()
        else:
            if self._post_bench:
                self._match_post_bench(frame, tracks, video_frame, embed_map)
                self._post_bench = False
            else:
                self._match_active(frame, tracks, video_frame, embed_map)

        self._resolve_duplicates()
        return dict(self.botsort_to_slot)

    def flush_stale_mappings(self, active_tids: set, current_frame: int = 0):
        """Called on bench→play transition. Marks all slots dormant; ages old dormant → lost."""
        stale = [tid for tid in self.botsort_to_slot if tid not in active_tids]
        n_stale = len(stale)
        for tid in stale:
            sid = self.botsort_to_slot.pop(tid)
            slot = self.slots.get(sid)
            if slot:
                slot.state = SlotState.DORMANT
                slot.active_track_id = None
        # Dormant-ize active slots that lost their BotSort tid
        still_mapped = set(self.botsort_to_slot.values())
        for slot in self.slots.values():
            if slot.state == SlotState.ACTIVE and slot.slot_id not in still_mapped:
                slot.state = SlotState.DORMANT
                slot.active_track_id = None
        # Age dormant slots that haven't been seen in _LOST_FRAMES → lost
        if current_frame > 0:
            for slot in self.slots.values():
                if (slot.state == SlotState.DORMANT and
                        current_frame - slot.last_seen_frame > _LOST_FRAMES):
                    slot.state = SlotState.LOST
        # Clear stale referee mappings too
        stale_refs = [tid for tid in self.botsort_to_ref if tid not in active_tids]
        for tid in stale_refs:
            del self.botsort_to_ref[tid]
        print(f"[Registry] Flushed {n_stale} stale, {len(self.botsort_to_slot)} remain")
        self._post_bench = True

    # ------------------------------------------------------------------
    # Referee registry (R1, R2, ...)
    # ------------------------------------------------------------------

    def _register_referee(self, tid: int, emb: Optional[np.ndarray], video_frame: int):
        if tid in self.botsort_to_ref:
            rid = self.botsort_to_ref[tid]
            ref = self.ref_slots[rid]
        else:
            rid = self.next_ref
            ref = RefereeSlot(ref_id=rid)
            self.ref_slots[rid] = ref
            self.botsort_to_ref[tid] = rid
            self.next_ref += 1
        ref.last_seen_frame = video_frame
        ref.active_track_id = tid
        if emb is not None:
            ref.embeddings.append(emb.copy())
            bank = np.array(ref.embeddings)
            ref.mean_embed = _l2norm(bank.mean(0))

    # ------------------------------------------------------------------
    # Registration (pre-freeze)
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: List, video_frame: int,
                  embed_map: Dict[int, np.ndarray]):
        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            crop = _crop_torso(frame, t)

            # Route referees to ref registry, never into P1-P22
            if cls == REFEREE_CLASS or _is_referee_by_color(crop):
                self._register_referee(tid, embed_map.get(tid), video_frame)
                continue

            if tid in self.botsort_to_slot:
                sid = self.botsort_to_slot[tid]
                slot = self.slots[sid]
                if crop is not None:
                    slot.hist = _blend_hist(slot.hist, _color_hist(crop), alpha=0.7)
                    slot.n_updates += 1
                if tid in embed_map:
                    self._add_embedding(slot, embed_map[tid])
                slot.last_seen_frame = video_frame
                slot.last_bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
            else:
                if crop is None:
                    continue
                if self.next_slot > self.max_players:
                    continue
                slot = PlayerSlot(slot_id=self.next_slot)
                slot.hist = _color_hist(crop)
                slot.last_seen_frame = video_frame
                slot.last_bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                slot.state = SlotState.ACTIVE
                slot.active_track_id = tid
                slot.n_updates = 1
                if tid in embed_map:
                    self._add_embedding(slot, embed_map[tid])
                self.slots[self.next_slot] = slot
                self.botsort_to_slot[tid] = self.next_slot
                self.next_slot += 1

    # ------------------------------------------------------------------
    # Freeze + team clustering
    # ------------------------------------------------------------------

    def _freeze(self):
        self.is_frozen = True
        self._assign_teams()
        ta = sum(1 for s in self.slots.values() if s.team == "A")
        tb = sum(1 for s in self.slots.values() if s.team == "B")
        ne = sum(1 for s in self.slots.values() if s.mean_embed is not None)
        print(f"[Registry] Frozen: {len(self.slots)} slots | A={ta} B={tb} | {ne} with ReID")

    def _assign_teams(self):
        sids = [sid for sid, s in self.slots.items() if s.hist is not None]
        if len(sids) < 4:
            return
        hists = np.array([self.slots[s].hist for s in sids])
        n = len(hists)

        # Farthest-pair seeding
        dist = 1.0 - (hists @ hists.T)
        idx = np.unravel_index(np.argmax(dist), dist.shape)
        c0, c1 = hists[idx[0]].copy(), hists[idx[1]].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(10):
            d0 = 1.0 - (hists @ c0)
            d1 = 1.0 - (hists @ c1)
            labels = (d1 < d0).astype(int)
            m0, m1 = labels == 0, labels == 1
            if m0.sum() > 0: c0 = _l2norm(hists[m0].mean(0))
            if m1.sum() > 0: c1 = _l2norm(hists[m1].mean(0))

        self._centroid_a = c0
        self._centroid_b = c1
        for i, sid in enumerate(sids):
            self.slots[sid].team = "A" if labels[i] == 0 else "B"

    # ------------------------------------------------------------------
    # Active play identity matching
    # ------------------------------------------------------------------

    def _match_active(self, frame: np.ndarray, tracks: List, video_frame: int,
                      embed_map: Dict[int, np.ndarray]):
        """
        Pass 1: known BotSort tids → update their slot.
        Pass 2: unknown tids → identity score matching against dormant/unmatched slots.
        Hard rules: 1 tid per slot, 1 slot per tid, team-locked, no referee.
        """
        used_slots: set = set()
        unknown = []

        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            crop = _crop_torso(frame, t)

            # Route referees to ref registry
            if cls == REFEREE_CLASS or _is_referee_by_color(crop):
                self._register_referee(tid, embed_map.get(tid), video_frame)
                continue

            if tid in self.botsort_to_slot:
                sid = self.botsort_to_slot[tid]

                # Collision guard: two BotSort tids mapped to same slot
                if sid in used_slots:
                    del self.botsort_to_slot[tid]
                    team = self._team_from_crop(crop)
                    unknown.append((t, tid, embed_map.get(tid), team, crop))
                    continue

                used_slots.add(sid)
                slot = self.slots[sid]
                slot.state = SlotState.ACTIVE
                slot.active_track_id = tid
                slot.last_seen_frame = video_frame
                slot.last_bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                if crop is not None:
                    slot.hist = _blend_hist(slot.hist, _color_hist(crop))
                if tid in embed_map:
                    self._add_embedding(slot, embed_map[tid])
            else:
                team = self._team_from_crop(crop)
                unknown.append((t, tid, embed_map.get(tid), team, crop))

        if not unknown:
            return

        # Available slots = not active + not lost
        available = [(sid, slot) for sid, slot in self.slots.items()
                     if sid not in used_slots and slot.state != SlotState.LOST]
        if not available:
            return

        self._assign_unknown(unknown, available, used_slots, video_frame,
                             embed_map, threshold=0.50)

    # ------------------------------------------------------------------
    # Post-bench full re-identification
    # ------------------------------------------------------------------

    def _match_post_bench(self, frame: np.ndarray, tracks: List, video_frame: int,
                          embed_map: Dict[int, np.ndarray]):
        used_slots: set = set()
        unknown = []

        for t in tracks:
            if len(t) < 7:
                continue
            tid = int(t[4])
            cls = int(t[6])
            crop = _crop_torso(frame, t)
            if cls == REFEREE_CLASS or _is_referee_by_color(crop):
                self._register_referee(tid, embed_map.get(tid), video_frame)
                continue
            team = self._team_from_crop(crop)
            unknown.append((t, tid, embed_map.get(tid), team, crop))

        if not unknown:
            return

        available = [(sid, s) for sid, s in self.slots.items()
                     if s.state != SlotState.LOST]
        matched = self._assign_unknown(unknown, available, used_slots, video_frame,
                                       embed_map, threshold=0.60)
        n_unknown = len(unknown)
        print(f"[Registry] Post-bench: matched {matched}/{n_unknown} tracks to {len(available)} slots")

    # ------------------------------------------------------------------
    # Core assignment: Hungarian on identity score
    # ------------------------------------------------------------------

    def _assign_unknown(self, unknown, available, used_slots: set,
                        video_frame: int, embed_map: Dict[int, np.ndarray],
                        threshold: float) -> int:
        n_t, n_s = len(unknown), len(available)
        cost = np.ones((n_t, n_s))

        for i, (t, tid, emb, team, crop) in enumerate(unknown):
            bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
            for j, (sid, slot) in enumerate(available):
                cost[i, j] = self._identity_cost(
                    emb, team, bbox, video_frame, slot)

        rows, cols = _hungarian(cost)
        matched = 0
        for r, c in zip(rows, cols):
            if cost[r, c] >= threshold:
                continue
            _, tid, emb, team, crop = unknown[r]
            sid, slot = available[c]
            if sid in used_slots:
                continue
            self.botsort_to_slot[tid] = sid
            used_slots.add(sid)
            slot.state = SlotState.ACTIVE
            slot.active_track_id = tid
            slot.last_seen_frame = video_frame
            slot.last_bbox = [float(unknown[r][0][0]), float(unknown[r][0][1]),
                               float(unknown[r][0][2]), float(unknown[r][0][3])]
            if emb is not None:
                self._add_embedding(slot, emb)
            if crop is not None:
                slot.hist = _blend_hist(slot.hist, _color_hist(crop)) if slot.hist is not None \
                            else _color_hist(crop)
            matched += 1
        return matched

    def _identity_cost(self, emb: Optional[np.ndarray], team: str,
                       bbox: List[float], frame_idx: int,
                       slot: PlayerSlot) -> float:
        """
        Lower = better match.
        Hard blocks return 1.0 (maximum cost).
        """
        # Hard: team lock (never cross-team)
        if slot.team in ("A", "B") and team in ("A", "B") and slot.team != team:
            return 1.0

        score = 0.0

        # ReID embedding similarity (primary, 60%)
        reid_sim = _cosine(emb, slot.mean_embed)
        score += 0.60 * (1.0 - reid_sim)

        # Spatial continuity (20%) — IoU with last known bbox
        if slot.last_bbox is not None and bbox is not None:
            iou = _iou(bbox, slot.last_bbox)
            score += 0.20 * (1.0 - iou)
        else:
            score += 0.20  # no spatial info = max cost contribution

        # Temporal penalty (20%) — penalise slots not seen recently
        gap = max(0, frame_idx - slot.last_seen_frame)
        time_penalty = min(gap / 750.0, 1.0)   # saturate at 30s @ 25fps
        score += 0.20 * time_penalty

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Duplicate resolution — hard rule: 1 permanent ID per frame
    # ------------------------------------------------------------------

    def _resolve_duplicates(self):
        """If two BotSort tids point to same slot, keep the one seen most recently."""
        slot_to_tids: Dict[int, List[int]] = {}
        for tid, sid in self.botsort_to_slot.items():
            slot_to_tids.setdefault(sid, []).append(tid)

        for sid, tids in slot_to_tids.items():
            if len(tids) <= 1:
                continue
            # Keep highest tid (most recently spawned by BotSort → likely better)
            keep = max(tids)
            for tid in tids:
                if tid != keep:
                    del self.botsort_to_slot[tid]

    # ------------------------------------------------------------------
    # Team detection
    # ------------------------------------------------------------------

    def _team_from_crop(self, crop: Optional[np.ndarray]) -> str:
        if crop is None or self._centroid_a is None:
            return "UNK"
        hist = _color_hist(crop)
        sa = float(np.dot(hist, self._centroid_a))
        sb = float(np.dot(hist, self._centroid_b))
        return "A" if sa > sb else "B"

    # ------------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------------

    def _add_embedding(self, slot: PlayerSlot, emb: np.ndarray):
        slot.embeddings.append(emb.copy())
        bank = np.array(slot.embeddings)
        slot.mean_embed = _l2norm(bank.mean(0))


# ---------------------------------------------------------------------------
# VLMStateMachine — scene state + registry orchestration
# ---------------------------------------------------------------------------

class VLMStateMachine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.state = GameState.PLAY
        self.prev_state = GameState.PLAY
        self.registry = PlayerRegistry(max_players=30, reg_frames=30)
        self.frame_counter = 0
        self._freeze_to_play = False   # any freeze scene → play transition
        print("[VLM] Identity engine ready")

    def analyze(self, frame: np.ndarray, video_frame: int) -> GameState:
        self.frame_counter += 1
        if self.frame_counter % 10 != 0:
            return self.state
        self.prev_state = self.state
        self.state = self._classify(frame)
        if self.state != self.prev_state:
            print(f"[VLM] Frame {video_frame}: {self.prev_state.value} → {self.state.value}")
            if self.prev_state.is_freeze() and self.state.is_play():
                self._freeze_to_play = True
        return self.state

    def _classify(self, frame: np.ndarray) -> GameState:
        """
        Cheap HSV classifier.
        green > 40% → play
        green 25-40% → celebration (players on pitch, little green visible e.g. goal crowd)
        green < 25% → bench_shot (field not visible)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = mask.sum() / (frame.shape[0] * frame.shape[1] * 255)
        if green_ratio >= 0.40:
            return GameState.PLAY
        if green_ratio >= 0.25:
            return GameState.CELEBRATION   # freeze assignment, BotSort still runs
        return GameState.BENCH_SHOT        # full freeze

    def get_id_remap(self, frame: np.ndarray, tracks: List, video_frame: int,
                     embed_map: Dict[int, np.ndarray] = None) -> Dict[int, int]:
        if embed_map is None:
            embed_map = {}
        if self._freeze_to_play and len(tracks) > 0:
            self._freeze_to_play = False
            active_tids = {int(t[4]) for t in tracks if len(t) >= 5}
            self.registry.flush_stale_mappings(active_tids, current_frame=video_frame)
        return self.registry.update(frame, tracks, video_frame, embed_map)
