from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


DORMANT_TTL = 180          # frames before dormant → lost (~7s @ 25fps)
MAX_SLOTS = 22
COST_REJECT_THRESHOLD = 0.72
EMB_ALPHA = 0.25


@dataclass
class PlayerSlot:
    pid: str
    state: str = "lost"    # active | dormant | lost
    active_track_id: Optional[int] = None
    seen_this_frame: bool = False
    last_seen_frame: int = -10**9
    last_position: Optional[Tuple[float, float]] = None
    embedding: Optional[np.ndarray] = None
    stability_counter: int = 0  # Number of consecutive frames this slot has been active


    def update_embedding(self, emb: np.ndarray) -> None:
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        if self.embedding is None:
            self.embedding = emb
        else:
            self.embedding = (1.0 - EMB_ALPHA) * self.embedding + EMB_ALPHA * emb
            norm2 = np.linalg.norm(self.embedding)
            if norm2 > 0:
                self.embedding = self.embedding / norm2


class IdentityCore:
    def __init__(self, logger=None, debug_every: int = 30):
        self.logger = logger
        self.debug_every = debug_every
        self.slots: List[PlayerSlot] = [PlayerSlot(pid=f"P{i}") for i in range(1, MAX_SLOTS + 1)]
        
        # BUG FIX E: ReID Fallback log
        print("[ReID] using HSV fallback only for appearance matching")

        self.frame_id: int = -1
        self.assigned_this_frame: int = 0
        self.unmatched_tracks: int = 0
        self.unmatched_slots: int = 0

        # Snapshot taken at bench/freeze entry — used for revival on return
        self._bench_snapshot: Dict[str, dict] = {}  # pid -> {embedding, position, last_seen}
        self._soft_snapshot: Dict[str, dict] = {}   # pid -> {embedding, position, last_seen}
        self._recovery_frames_left: int = 0  # countdown after scene reset
        self._last_memory_skips: int = 0

    # ------------------------------------------------------------------
    # Scene reset
    # ------------------------------------------------------------------

    def reset_for_scene(self) -> None:
        """Clear active tracking state on scene boundary (bench_shot→play) 
        but preserve dormant memory and bench snapshot to revive later."""
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False
            if s.state in ("active", "dormant") and s.embedding is not None:
                s.state = "dormant"
            else:
                s.state = "lost"

        if len(self._bench_snapshot) > 0:
            print(f"[Identity] Reset: {len(self._bench_snapshot)} snapshot slots survived reset")
        else:
            print(f"[Identity] Reset: No snapshot exists to survive reset.")
            
        self._recovery_frames_left = 60  # permissive matching for 60 frames

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.assigned_this_frame = 0
        self.unmatched_tracks = 0
        self.unmatched_slots = 0
        if self._recovery_frames_left > 0:
            self._recovery_frames_left -= 1
        # CRITICAL: clear frame-local flags only — do NOT reset state/embedding
        for s in self.slots:
            s.active_track_id = None
            s.seen_this_frame = False

    def assign_tracks(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        memory_ok_tids: Optional[set] = None,
    ) -> Dict[int, str]:
        """
        Match current tracks to PID slots via Hungarian algorithm.
        Cost = 0.70 * embedding_distance + 0.25 * position_distance + recency_penalty.

        Args:
            memory_ok_tids: Set of track IDs allowed to update slot embeddings.
                If None, all tracks can update. If provided, tracks NOT in this
                set will still be assigned a PID but won't pollute identity memory.

        Returns {track_id -> PID string}.
        """
        if len(tracks) == 0:
            self.unmatched_slots = MAX_SLOTS
            return {}

        track_ids = [int(t.track_id) for t in tracks]
        n_t = len(track_ids)
        n_s = len(self.slots)

        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, slot in enumerate(self.slots):
                cost[i, j] = self._slot_cost(slot, t_emb, t_pos)

        r_idx, c_idx = linear_sum_assignment(cost)

        track_to_pid: Dict[int, str] = {}
        matched_tracks: set = set()
        matched_slots: set = set()
        memory_skips = 0

        for r, c in zip(r_idx, c_idx):
            cst = float(cost[r, c])
            if cst > COST_REJECT_THRESHOLD:
                continue

            tid = track_ids[r]
            slot = self.slots[c]

            track_to_pid[tid] = slot.pid
            slot.active_track_id = tid
            slot.seen_this_frame = True
            slot.state = "active"
            slot.last_seen_frame = self.frame_id
            slot.last_position = positions.get(tid)
            slot.stability_counter += 1

            # Prompt 11: Only update embedding if crop quality allows it
            emb = embeddings.get(tid)
            if emb is not None:
                if memory_ok_tids is None or tid in memory_ok_tids:
                    slot.update_embedding(emb)
                else:
                    memory_skips += 1

            self.assigned_this_frame += 1
            matched_tracks.add(r)
            matched_slots.add(c)

        self.unmatched_tracks = n_t - len(matched_tracks)
        self.unmatched_slots = n_s - len(matched_slots)
        self._last_memory_skips = memory_skips
        return track_to_pid

    def end_frame(self, frame_id: Optional[int] = None) -> None:
        if frame_id is not None:
            self.frame_id = frame_id
        for s in self.slots:
            if s.seen_this_frame:
                continue
            age = self.frame_id - s.last_seen_frame
            s.state = "dormant" if age <= DORMANT_TTL else "lost"
            s.active_track_id = None
            if not s.seen_this_frame:
                s.stability_counter = 0

    # ------------------------------------------------------------------
    # Bench snapshot / revival (Priority D)
    # ------------------------------------------------------------------

    def snapshot_scene(self, frame_id: int) -> int:
        """
        Call when entering bench/freeze. Saves all active+dormant slot state
        so we can attempt strong revival on return-to-play.
        Returns count of slots saved.
        """
        self._bench_snapshot = {}
        for s in self.slots:
            if s.state in ("active", "dormant") and s.embedding is not None:
                self._bench_snapshot[s.pid] = {
                    "embedding": s.embedding.copy(),
                    "position": s.last_position,
                    "last_seen": s.last_seen_frame,
                }
        saved = len(self._bench_snapshot)
        print(f"[Identity] SceneSnapshot: {saved} slots saved at frame {frame_id}")
        return saved
        
    def snapshot_soft(self, frame_id: int) -> int:
        """
        Call when entering soft-collapse. Saves state to revive tracks
        when the scene stabilizes.
        """
        self._soft_snapshot = {}
        for s in self.slots:
            if s.state in ("active", "dormant") and s.embedding is not None:
                self._soft_snapshot[s.pid] = {
                    "embedding": s.embedding.copy(),
                    "position": s.last_position,
                    "last_seen": s.last_seen_frame,
                }
        saved = len(self._soft_snapshot)
        print(f"[Identity] SoftSnapshot: {saved} slots saved at frame {frame_id}")
        return saved

    def revive_cost_matrix(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
    ) -> Dict[int, str]:
        """
        Called on first frame after bench→play. Matches tracks to snapshot
        embeddings first (stronger signal) before normal per-frame assignment.
        Returns partial {track_id -> PID} for revived tracks.
        """
        if not self._bench_snapshot or len(tracks) == 0:
            return {}

        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(self._bench_snapshot.keys())
        n_t = len(track_ids)
        n_s = len(snap_pids)

        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, pid in enumerate(snap_pids):
                snap = self._bench_snapshot[pid]
                emb_cost = 0.5
                pos_cost = 0.5

                if t_emb is not None and snap["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0:
                        e = e / n
                    cos = float(np.clip(np.dot(e, snap["embedding"]), -1.0, 1.0))
                    emb_cost = 1.0 - (cos + 1.0) * 0.5

                if t_pos is not None and snap["position"] is not None:
                    dx = float(t_pos[0] - snap["position"][0])
                    dy = float(t_pos[1] - snap["position"][1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    pos_cost = min(dist / 300.0, 1.0)  # looser after cut

                cost[i, j] = 0.75 * emb_cost + 0.25 * pos_cost

        r_idx, c_idx = linear_sum_assignment(cost)

        revived: Dict[int, str] = {}
        for r, c in zip(r_idx, c_idx):
            if cost[r, c] < 0.60:  # tighter threshold for revival
                revived[track_ids[r]] = snap_pids[c]

        print(f"[Identity] Revival: {len(revived)}/{n_t} tracks matched from scene snapshot")
        
        # Explicitly update revived slots so they are recognized by end_frame and assign_tracks
        for tid, pid in revived.items():
            for slot in self.slots:
                if slot.pid == pid:
                    slot.active_track_id = tid
                    slot.seen_this_frame = True
                    slot.state = "active"
                    slot.last_seen_frame = self.frame_id
                    if tid in positions:
                        slot.last_position = positions[tid]
                    slot.stability_counter += 1
                    if tid in embeddings:
                        slot.update_embedding(embeddings[tid])
                    self.assigned_this_frame += 1
                    break
                    
        self._bench_snapshot = {}  # consume once
        return revived

    def revive_from_soft_snapshot(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        is_first_recovery_frame: bool = False
    ) -> Dict[int, str]:
        """
        Matches tracks to soft snapshot. Removes matched from the soft snapshot progressively.
        """
        if not hasattr(self, "_soft_snapshot") or not self._soft_snapshot or len(tracks) == 0:
            return {}

        track_ids = [int(t.track_id) for t in tracks]
        snap_pids = list(self._soft_snapshot.keys())
        n_t = len(track_ids)
        n_s = len(snap_pids)

        cost = np.full((n_t, n_s), 1e3, dtype=np.float32)
        debug_costs = []

        for i, tid in enumerate(track_ids):
            t_emb = embeddings.get(tid)
            t_pos = positions.get(tid)
            for j, pid in enumerate(snap_pids):
                snap = self._soft_snapshot[pid]
                emb_cost = 0.5
                pos_cost = 0.5

                if t_emb is not None and snap["embedding"] is not None:
                    e = t_emb.astype(np.float32)
                    n = np.linalg.norm(e)
                    if n > 0: e = e / n
                    cos = float(np.clip(np.dot(e, snap["embedding"]), -1.0, 1.0))
                    emb_cost = 1.0 - (cos + 1.0) * 0.5

                if t_pos is not None and snap["position"] is not None:
                    dx = float(t_pos[0] - snap["position"][0])
                    dy = float(t_pos[1] - snap["position"][1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    pos_cost = min(dist / 300.0, 1.0) 

                cst = 0.85 * emb_cost + 0.15 * pos_cost
                cost[i, j] = cst
                if is_first_recovery_frame:
                    debug_costs.append({
                        "tid": tid, "pid": pid, "cost": float(cst), 
                        "emb_cost": emb_cost, "pos_cost": pos_cost
                    })

        r_idx, c_idx = linear_sum_assignment(cost)

        revived: Dict[int, str] = {}
        for r, c in zip(r_idx, c_idx):
            cst = cost[r, c]
            accepted = cst < 0.65  # Threshold for soft revival inside same scene
            if accepted:
                revived[track_ids[r]] = snap_pids[c]

        if is_first_recovery_frame:
            debug_costs.sort(key=lambda x: x["cost"])
            for d in debug_costs[:5]:
                accepted = d["cost"] < 0.65
                print(f"[ReviveCost] tid={d['tid']} pid={d['pid']} cost={d['cost']:.3f} emb={d['emb_cost']:.3f} pos={d['pos_cost']:.3f} accepted={accepted}")

        # Explicit lock on the slots
        for tid, pid in revived.items():
            for slot in self.slots:
                if slot.pid == pid:
                    slot.active_track_id = tid
                    slot.seen_this_frame = True
                    slot.state = "active"
                    slot.last_seen_frame = self.frame_id
                    if tid in positions:
                        slot.last_position = positions[tid]
                    slot.stability_counter += 1
                    if tid in embeddings:
                        slot.update_embedding(embeddings[tid])
                    self.assigned_this_frame += 1
                    break
            
            # Consume from soft snapshot progressively
            if pid in self._soft_snapshot:
                del self._soft_snapshot[pid]
                
        return revived

    # ------------------------------------------------------------------
    # Metrics / logging
    # ------------------------------------------------------------------

    def state_counts(self) -> Tuple[int, int, int]:
        active = sum(1 for s in self.slots if s.state == "active")
        dormant = sum(1 for s in self.slots if s.state == "dormant")
        lost = sum(1 for s in self.slots if s.state == "lost")
        return active, dormant, lost

    def maybe_log(self, detections_count: int, tracks_count: int) -> None:
        if self.frame_id % self.debug_every != 0:
            return
        active, dormant, lost = self.state_counts()
        lost = min(lost, MAX_SLOTS)
        print(
            f"[Frame {self.frame_id}] det={detections_count} tracks={tracks_count} "
            f"assigned={self.assigned_this_frame} active={active} dormant={dormant} lost={lost} "
            f"unmatched_tracks={self.unmatched_tracks} unmatched_slots={self.unmatched_slots} "
            f"memory_skips={self._last_memory_skips}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _slot_cost(
        self,
        slot: PlayerSlot,
        t_emb: Optional[np.ndarray],
        t_pos: Optional[Tuple[float, float]],
    ) -> float:
        # Recovery mode: lost slots with no embedding eagerly accept tracks
        if self._recovery_frames_left > 0 and slot.state == "lost" and slot.embedding is None:
            return 0.30

        emb_cost = 0.5
        pos_cost = 0.5

        if t_emb is not None and slot.embedding is not None:
            e = t_emb.astype(np.float32)
            n = np.linalg.norm(e)
            if n > 0:
                e = e / n
            cos = float(np.clip(np.dot(e, slot.embedding), -1.0, 1.0))
            emb_cost = 1.0 - (cos + 1.0) * 0.5

        if t_pos is not None and slot.last_position is not None:
            dx = float(t_pos[0] - slot.last_position[0])
            dy = float(t_pos[1] - slot.last_position[1])
            dist = (dx * dx + dy * dy) ** 0.5
            pos_cost = min(dist / 200.0, 1.0)

        recency = min(max(self.frame_id - slot.last_seen_frame, 0), DORMANT_TTL)
        recency_penalty = recency / float(DORMANT_TTL) * 0.15

        # Architecture Change 2: Identity Lock / Anti-Swap logic
        # If slot has been stable for >= 10 frames, give it a large "continuity lock" discount
        # to prevent same-team swaps when appearance is ambiguous.
        lock_discount = 0.0
        if slot.stability_counter >= 10:
            lock_discount = 0.15

        # Bug Fix F: Lower HSV weight but use it securely. Give position even less weight during recovery!
        if self._recovery_frames_left > 0:
            # Recovery Mode: prioritize appearance almost exclusively
            final_cost = 0.90 * emb_cost + 0.05 * pos_cost + recency_penalty - lock_discount
        else:
            # Active tracking
            if recency == 0:
                # Strong continuity prior if active
                final_cost = 0.50 * emb_cost + 0.35 * pos_cost - 0.10 - lock_discount
            else:
                final_cost = 0.65 * emb_cost + 0.25 * pos_cost + recency_penalty - (lock_discount * 0.5)
                
        return float(max(0.0, min(1.0, final_cost)))
