"""
Deep-EIoU tracker with GTA-Link post-processing.

Deep-EIoU: iterative IoU expansion for online association.
  - Expands bounding boxes proportionally before computing IoU cost.
  - Two-round matching: high-conf dets first, low-conf recovery second.
  - Appearance cost blended with expanded-IoU cost.

GTA-Link: global tracklet association post-process.
  - Runs after all frames processed.
  - Merges fragmented tracklets via embedding cosine similarity + temporal gap gate.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Kalman filter (simple constant-velocity)
# ---------------------------------------------------------------------------

class KalmanBox:
    """
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Observation: [cx, cy, w, h]
    """
    _I = np.eye(8)

    def __init__(self, bbox_xyxy: np.ndarray):
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w  = bbox_xyxy[2] - bbox_xyxy[0]
        h  = bbox_xyxy[3] - bbox_xyxy[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.P = np.diag([w*w, h*h, w*w, h*h, 100., 100., 10., 10.])
        # Motion model
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0
        # Observation model
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
        self.Q = np.diag([1., 1., 1., 1., 1., 1., 0.1, 0.1])
        self.R = np.diag([1., 1., 10., 10.])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_xyxy()

    def update(self, bbox_xyxy: np.ndarray):
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w  = bbox_xyxy[2] - bbox_xyxy[0]
        h  = bbox_xyxy[3] - bbox_xyxy[1]
        z = np.array([cx, cy, w, h])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P
        return self._to_xyxy()

    def _to_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    @property
    def mean_xyxy(self) -> np.ndarray:
        return self._to_xyxy()


# ---------------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------------

class DETrack:
    _next_id = 1

    def __init__(self, bbox_xyxy: np.ndarray, score: float, cls: int,
                 embed: Optional[np.ndarray] = None, frame: int = 0):
        self.track_id  = DETrack._next_id
        DETrack._next_id += 1
        self.kf        = KalmanBox(bbox_xyxy)
        self.score     = score
        self.cls       = cls
        self.hits      = 1
        self.age       = 0
        self.time_since_update = 0
        self.state     = "tentative"   # tentative → tracked → lost
        self.embed     = embed.copy() if embed is not None else None
        self.start_frame = frame
        self.last_frame  = frame
        # Embedding history for GTA-Link
        self._embed_bank: List[np.ndarray] = [] if embed is None else [embed.copy()]

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, bbox_xyxy: np.ndarray, score: float,
               embed: Optional[np.ndarray], frame: int):
        self.kf.update(bbox_xyxy)
        self.score  = score
        self.hits  += 1
        self.time_since_update = 0
        self.last_frame = frame
        if self.hits >= 3:
            self.state = "tracked"
        if embed is not None:
            self.embed = embed.copy()
            self._embed_bank.append(embed.copy())
            if len(self._embed_bank) > 30:
                self._embed_bank.pop(0)

    @property
    def mean_embed(self) -> Optional[np.ndarray]:
        if not self._embed_bank:
            return self.embed
        m = np.mean(self._embed_bank, axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    @property
    def bbox(self) -> np.ndarray:
        return self.kf.mean_xyxy

    def mark_lost(self):
        self.state = "lost"


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) → (N,M) IoU matrix."""
    ax1, ay1, ax2, ay2 = bboxes_a[:, 0], bboxes_a[:, 1], bboxes_a[:, 2], bboxes_a[:, 3]
    bx1, by1, bx2, by2 = bboxes_b[:, 0], bboxes_b[:, 1], bboxes_b[:, 2], bboxes_b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def _expand_boxes(bboxes: np.ndarray, scale: float) -> np.ndarray:
    """Expand boxes by scale factor around center (Deep-EIoU key step)."""
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w  = (bboxes[:, 2] - bboxes[:, 0]) * scale
    h  = (bboxes[:, 3] - bboxes[:, 1]) * scale
    return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)


def _embed_cost(tracks: List[DETrack], embeds: List[Optional[np.ndarray]]) -> np.ndarray:
    """Cosine distance matrix: (N tracks) x (M dets)."""
    N, M = len(tracks), len(embeds)
    cost = np.ones((N, M), dtype=float)
    for i, tr in enumerate(tracks):
        te = tr.mean_embed
        if te is None:
            continue
        for j, de in enumerate(embeds):
            if de is None:
                continue
            cost[i, j] = 1.0 - float(np.dot(te, de) / (np.linalg.norm(te) * np.linalg.norm(de) + 1e-8))
    return cost


def _greedy_assign(cost: np.ndarray, thresh: float) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Greedy min-cost matching. Returns (matched_r, matched_c, unmatched_r, unmatched_c)."""
    rows, cols = cost.shape
    pairs = sorted(
        ((cost[r, c], r, c) for r in range(rows) for c in range(cols)),
        key=lambda x: x[0]
    )
    used_r, used_c = set(), set()
    matched_r, matched_c = [], []
    for v, r, c in pairs:
        if v > thresh:
            break
        if r not in used_r and c not in used_c:
            matched_r.append(r)
            matched_c.append(c)
            used_r.add(r)
            used_c.add(c)
    unmatched_r = [r for r in range(rows) if r not in used_r]
    unmatched_c = [c for c in range(cols) if c not in used_c]
    return matched_r, matched_c, unmatched_r, unmatched_c


# ---------------------------------------------------------------------------
# Deep-EIoU Tracker
# ---------------------------------------------------------------------------

class DeepEIoUTracker:
    """
    Online multi-object tracker using iterative expanded IoU + appearance.

    Association rounds:
      Round 1: high-conf dets (score >= track_high_thresh)
               cost = 0.5 * (1 - expanded_iou) + 0.5 * embed_cost
               expand_scale = 1.5 (moderate expansion)
      Round 2: low-conf dets (score >= track_low_thresh)
               cost = 1 - expanded_iou only (byte-style recovery)
               expand_scale = 2.0 (larger expansion for fast movers)
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh:  float = 0.10,
        new_track_thresh:  float = 0.40,
        match_thresh:      float = 0.70,   # cost threshold round 1
        iou_only_thresh:   float = 0.80,   # cost threshold round 2
        max_age:           int   = 150,
        min_hits:          int   = 3,
        expand_r1:         float = 1.5,    # expansion round 1
        expand_r2:         float = 2.0,    # expansion round 2
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.new_track_thresh  = new_track_thresh
        self.match_thresh      = match_thresh
        self.iou_only_thresh   = iou_only_thresh
        self.max_age           = max_age
        self.min_hits          = min_hits
        self.expand_r1         = expand_r1
        self.expand_r2         = expand_r2

        self.track_buffer      = 30     # frames before tracked→lost demotion

        self.tracked:  List[DETrack] = []
        self.lost:     List[DETrack] = []
        self._frame    = 0

    def update(
        self,
        dets_xyxy:  np.ndarray,           # (N,4)
        scores:     np.ndarray,           # (N,)
        classes:    np.ndarray,           # (N,)
        embeds:     Optional[np.ndarray] = None,  # (N, D) or None
    ) -> List[DETrack]:
        """Returns list of active DETrack objects (state==tracked, hits>=min_hits)."""
        self._frame += 1

        # Predict all existing tracks
        for tr in self.tracked + self.lost:
            tr.predict()

        # Split detections by confidence
        high_mask = scores >= self.track_high_thresh
        low_mask  = (scores >= self.track_low_thresh) & ~high_mask

        hi_bboxes  = dets_xyxy[high_mask]
        hi_scores  = scores[high_mask]
        hi_cls     = classes[high_mask]
        hi_embeds  = embeds[high_mask] if embeds is not None else [None] * high_mask.sum()

        lo_bboxes  = dets_xyxy[low_mask]
        lo_scores  = scores[low_mask]
        lo_cls     = classes[low_mask]

        # Split tracked into active (recently matched) vs stale (drifting)
        # Bug #1 fix: demote stale tracks to lost after track_buffer frames
        active = []
        demoted_to_lost = []
        for tr in self.tracked:
            if tr.time_since_update <= 1:
                active.append(tr)
            elif tr.time_since_update > self.track_buffer:
                # Dormant demotion: track hasn't matched in 30 frames → lost
                tr.mark_lost()
                demoted_to_lost.append(tr)
            else:
                active.append(tr)  # stale but within buffer, still participable

        # ---- Round 1: high-conf dets vs ALL active/stale tracked ----
        unmatched_tracks_r1 = list(range(len(active)))
        unmatched_dets_r1   = list(range(len(hi_bboxes)))

        if len(active) > 0 and len(hi_bboxes) > 0:
            tr_bboxes = np.array([tr.bbox for tr in active])
            exp_tr    = _expand_boxes(tr_bboxes, self.expand_r1)
            exp_det   = _expand_boxes(hi_bboxes,  self.expand_r1)

            iou_cost  = 1.0 - _iou_matrix(exp_tr, exp_det)
            emb_list  = [e if e is not None else None for e in hi_embeds]
            app_cost  = _embed_cost(active, emb_list)

            has_embed = np.array([tr.mean_embed is not None for tr in active])
            cost = np.where(has_embed[:, None], 0.5 * iou_cost + 0.5 * app_cost, iou_cost)

            mr, mc, unmatched_tracks_r1, unmatched_dets_r1 = _greedy_assign(cost, self.match_thresh)
            for ri, ci in zip(mr, mc):
                emb = hi_embeds[ci] if embeds is not None else None
                active[ri].update(hi_bboxes[ci], hi_scores[ci], emb, self._frame)

        # ---- Round 2: low-conf dets vs unmatched tracks (IoU only) ----
        r2_tracks = [active[i] for i in unmatched_tracks_r1]
        matched_r2_positions = set()
        if len(r2_tracks) > 0 and len(lo_bboxes) > 0:
            tr_bboxes = np.array([tr.bbox for tr in r2_tracks])
            exp_tr    = _expand_boxes(tr_bboxes, self.expand_r2)
            exp_det   = _expand_boxes(lo_bboxes,  self.expand_r2)
            iou_cost  = 1.0 - _iou_matrix(exp_tr, exp_det)

            mr2, mc2, _, _ = _greedy_assign(iou_cost, self.iou_only_thresh)
            for ri, ci in zip(mr2, mc2):
                r2_tracks[ri].update(lo_bboxes[ci], lo_scores[ci], None, self._frame)
                matched_r2_positions.add(ri)

        # Tracks still unmatched after both rounds
        finally_unmatched = [
            active[unmatched_tracks_r1[pos]]
            for pos in range(len(unmatched_tracks_r1))
            if pos not in matched_r2_positions
        ]

        # ---- Round 3 (Bug #2 fix): try unmatched high-conf dets against lost pool ----
        # Recover lost tracks before spawning new ones to prevent duplicates
        used_det_indices = set()  # det indices consumed by lost recovery
        if len(unmatched_dets_r1) > 0 and len(self.lost) > 0:
            r3_dets_idx = unmatched_dets_r1
            r3_bboxes = hi_bboxes[r3_dets_idx] if len(r3_dets_idx) > 0 else np.empty((0, 4))
            if len(r3_bboxes) > 0:
                tr_bboxes = np.array([tr.bbox for tr in self.lost])
                exp_tr  = _expand_boxes(tr_bboxes, self.expand_r2)
                exp_det = _expand_boxes(r3_bboxes, self.expand_r2)
                iou_cost = 1.0 - _iou_matrix(exp_tr, exp_det)

                # Also use appearance if available
                r3_embeds = [hi_embeds[ci] if embeds is not None else None for ci in r3_dets_idx]
                app_cost = _embed_cost(self.lost, r3_embeds)
                has_embed = np.array([tr.mean_embed is not None for tr in self.lost])
                cost_r3 = np.where(has_embed[:, None], 0.5 * iou_cost + 0.5 * app_cost, iou_cost)

                mr3, mc3, _, _ = _greedy_assign(cost_r3, self.match_thresh)
                for ri, ci in zip(mr3, mc3):
                    emb = hi_embeds[r3_dets_idx[ci]] if embeds is not None else None
                    self.lost[ri].update(hi_bboxes[r3_dets_idx[ci]], hi_scores[r3_dets_idx[ci]], emb, self._frame)
                    self.lost[ri].state = "tracked"
                    used_det_indices.add(r3_dets_idx[ci])

        # ---- Spawn new tracks ONLY from truly unmatched high-conf dets ----
        for ci in unmatched_dets_r1:
            if ci in used_det_indices:
                continue  # already recovered a lost track
            if hi_scores[ci] >= self.new_track_thresh:
                emb = hi_embeds[ci] if embeds is not None else None
                tr = DETrack(hi_bboxes[ci], hi_scores[ci], hi_cls[ci], emb, self._frame)
                self.tracked.append(tr)

        # ---- Move unmatched active tracks to lost ----
        lost_tids = set()
        for tr in finally_unmatched:
            tr.mark_lost()
            lost_tids.add(tr.track_id)
        for tr in demoted_to_lost:
            lost_tids.add(tr.track_id)
            
        self.lost.extend(demoted_to_lost)
        self.lost.extend(finally_unmatched)

        # ---- Process lost pool: recover matched, prune dead ----
        recovered_tids = set()
        still_lost = []
        for tr in self.lost:
            if tr.state == "tracked":
                self.tracked.append(tr)
                recovered_tids.add(tr.track_id)
            elif tr.time_since_update <= self.max_age:
                still_lost.append(tr)
        self.lost = still_lost

        # BUG FIX 1: Remove dead/lost tracks from tracked
        # Keep tentative + tracked tracks alive.
        # Only remove tracks that are explicitly lost or too stale.
        self.tracked = [
            tr for tr in self.tracked 
            if tr.state != "lost" 
            and tr.time_since_update <= self.track_buffer
            and tr.track_id not in lost_tids
        ]

        # Return only confirmed active tracks
        active_returns = [tr for tr in self.tracked if tr.hits >= self.min_hits and tr.state == "tracked"]
        
        # Guardrail: Log DeepEIoU state
        if self._frame % 30 == 0:
            print(f"[DeepEIoU F{self._frame}] tracked={len(self.tracked)} lost={len(self.lost)} returned={len(active_returns)}")
            
        return active_returns

    def reset(self):
        """Bug #3: Clear all track state on scene boundary (e.g. bench_shot→play)."""
        self.tracked.clear()
        self.lost.clear()

    @property
    def active_tracks(self) -> List[DETrack]:
        return [tr for tr in self.tracked if tr.hits >= self.min_hits and tr.state == "tracked"]


# ---------------------------------------------------------------------------
# GTA-Link: global tracklet association post-process
# ---------------------------------------------------------------------------

class GTALink:
    """
    After tracking completes, merge fragmented tracklets.

    Algorithm:
      1. Build tracklet summaries (mean_embed, time range, last/first position).
      2. For each tracklet pair (A, B) where A ends before B starts:
         - Check temporal gap <= max_gap frames.
         - Check cosine similarity of mean embeddings >= min_sim.
         - Check spatial plausibility (max pixel displacement per frame).
      3. Build a union-find and merge passing pairs.
      4. Return a mapping: old_track_id -> new_global_id.
    """

    def __init__(
        self,
        max_gap:      int   = 120,   # max frame gap to consider merging
        min_sim:      float = 0.65,  # min cosine similarity to merge
        max_px_per_f: float = 80.0,  # max pixel/frame spatial plausibility
    ):
        self.max_gap      = max_gap
        self.min_sim      = min_sim
        self.max_px_per_f = max_px_per_f

    def run(self, tracklets: Dict[int, dict]) -> Dict[int, int]:
        """
        tracklets: {track_id: {
            'mean_embed': np.ndarray,
            'start_frame': int,
            'end_frame': int,
            'first_bbox': [x1,y1,x2,y2],
            'last_bbox':  [x1,y1,x2,y2],
        }}
        Returns: {old_track_id: global_id}
        """
        ids = list(tracklets.keys())
        n   = len(ids)

        if n == 0:
            return {}

        # Union-Find
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Sort by end_frame for efficiency
        ids_sorted = sorted(ids, key=lambda i: tracklets[i]['end_frame'])

        merged = 0
        for ai, a_id in enumerate(ids_sorted):
            a = tracklets[a_id]
            a_emb = a.get('mean_embed')
            if a_emb is None:
                continue

            for b_id in ids_sorted[ai+1:]:
                b = tracklets[b_id]

                # Temporal gate: B must start after A ends
                gap = b['start_frame'] - a['end_frame']
                if gap < 0 or gap > self.max_gap:
                    continue

                # Appearance gate
                b_emb = b.get('mean_embed')
                if b_emb is None:
                    continue
                sim = float(np.dot(a_emb, b_emb) / (
                    np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-8))
                if sim < self.min_sim:
                    continue

                # Spatial plausibility
                a_last = a['last_bbox']
                b_first = b['first_bbox']
                if a_last is not None and b_first is not None and gap > 0:
                    ax = (a_last[0] + a_last[2]) / 2
                    ay = (a_last[1] + a_last[3]) / 2
                    bx = (b_first[0] + b_first[2]) / 2
                    by = (b_first[1] + b_first[3]) / 2
                    dist = np.sqrt((bx - ax)**2 + (by - ay)**2)
                    if dist / gap > self.max_px_per_f:
                        continue

                union(a_id, b_id)
                merged += 1

        # Build global ID mapping (root of union-find = global ID)
        roots = sorted(set(find(i) for i in ids))
        root_to_global = {r: gi+1 for gi, r in enumerate(roots)}
        result = {tid: root_to_global[find(tid)] for tid in ids}

        n_before = len(ids)
        n_after  = len(set(result.values()))
        print(f"[GTALink] Tracklets: {n_before} → {n_after} "
              f"(merged {n_before - n_after}, {merged} pairs linked)")
        return result


# ---------------------------------------------------------------------------
# Helper: build tracklet summaries from track_results.json frames
# ---------------------------------------------------------------------------

def build_tracklet_summaries(frames_data: List[dict]) -> Dict[int, dict]:
    """Build per-trackId summary from track_results frames list."""
    summaries: Dict[int, dict] = {}
    for frame in frames_data:
        fi = frame['frameIndex']
        for p in frame.get('players', []):
            tid = p['trackId']
            bbox = p.get('bbox')
            emb  = p.get('_embed')  # optional, may not be stored
            if tid not in summaries:
                summaries[tid] = {
                    'start_frame': fi,
                    'end_frame':   fi,
                    'first_bbox':  bbox,
                    'last_bbox':   bbox,
                    'mean_embed':  None,
                    '_embeds':     [],
                }
            s = summaries[tid]
            s['end_frame'] = fi
            s['last_bbox'] = bbox
            if emb is not None:
                s['_embeds'].append(np.array(emb))

    # Compute mean embeddings
    for s in summaries.values():
        if s['_embeds']:
            m = np.mean(s['_embeds'], axis=0)
            n = np.linalg.norm(m)
            s['mean_embed'] = m / n if n > 0 else m
        del s['_embeds']

    return summaries
