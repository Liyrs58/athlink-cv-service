"""
Tracklet Graph Resolver — split identity histories into tracklets
and propose deterministic corrections when evidence is strong.

A tracklet is a contiguous segment of a PID's history between conflict
boundaries (or start/end of video). The resolver builds a graph of
tracklets and proposes correction actions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter


# ---------------------------------------------------------------------------
# Correction actions
# ---------------------------------------------------------------------------

class CorrectionAction(str, Enum):
    SWAP_PID_AFTER_FRAME = "swap_pid_after_frame"
    SPLIT_TRACKLET = "split_tracklet"
    MERGE_TRACKLETS = "merge_tracklets"
    ASSIGN_PID_TO_TRACKLET = "assign_pid_to_tracklet"
    REJECT_REVIVAL = "reject_revival"
    MARK_UNKNOWN = "mark_unknown"
    SUPPRESS_PID = "suppress_pid"


# ---------------------------------------------------------------------------
# Tracklet
# ---------------------------------------------------------------------------

@dataclass
class Tracklet:
    tracklet_id: str
    pid: str
    raw_track_id: int
    start_frame: int
    end_frame: int
    team_id_mode: Optional[int] = None
    role_mode: str = "player"
    first_bbox: Optional[List[float]] = None
    last_bbox: Optional[List[float]] = None
    mean_velocity: float = 0.0
    assignment_sources: List[str] = field(default_factory=list)
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    frame_count: int = 0
    entries: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "tracklet_id": self.tracklet_id,
            "pid": self.pid,
            "raw_track_id": self.raw_track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "team_id_mode": self.team_id_mode,
            "role_mode": self.role_mode,
            "first_bbox": self.first_bbox,
            "last_bbox": self.last_bbox,
            "mean_velocity": round(self.mean_velocity, 2),
            "assignment_sources": self.assignment_sources,
            "confidence_stats": self.confidence_stats,
            "frame_count": self.frame_count,
        }
        return d


# ---------------------------------------------------------------------------
# Correction proposal
# ---------------------------------------------------------------------------

@dataclass
class CorrectionProposal:
    action: str
    window_id: str
    confidence: float
    reason_codes: List[str] = field(default_factory=list)
    pid_a: Optional[str] = None
    pid_b: Optional[str] = None
    apply_from_frame: Optional[int] = None
    apply_to_frame: Optional[int] = None
    tracklet_id: Optional[str] = None
    new_pid: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "action": self.action,
            "window_id": self.window_id,
            "confidence": round(self.confidence, 3),
            "reason_codes": self.reason_codes,
        }
        if self.pid_a is not None:
            d["pid_a"] = self.pid_a
        if self.pid_b is not None:
            d["pid_b"] = self.pid_b
        if self.apply_from_frame is not None:
            d["apply_from_frame"] = self.apply_from_frame
        if self.apply_to_frame is not None:
            d["apply_to_frame"] = self.apply_to_frame
        if self.tracklet_id is not None:
            d["tracklet_id"] = self.tracklet_id
        if self.new_pid is not None:
            d["new_pid"] = self.new_pid
        return d


def _bbox_center(bbox: list) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TrackletGraphResolver:
    """
    Splits PID histories into tracklets at conflict boundaries,
    then proposes corrections.
    """

    def build_tracklets(
        self,
        track_results: dict,
        conflict_windows: List[dict],
    ) -> List[Tracklet]:
        """
        Build tracklets from track_results, splitting at conflict boundaries.
        """
        frames = track_results.get("frames", [])
        # Build per-PID ordered history
        pid_history: Dict[str, List[dict]] = {}
        for frame_data in frames:
            fi = frame_data.get("frameIndex", -1)
            for p in frame_data.get("players", []):
                pid = p.get("playerId")
                if pid is None:
                    continue
                entry = {
                    "frame": fi,
                    "bbox": p.get("bbox", [0, 0, 0, 0]),
                    "trackId": p.get("trackId"),
                    "rawTrackId": p.get("rawTrackId"),
                    "assignment_source": p.get("assignment_source", "unknown"),
                    "identity_confidence": p.get("identity_confidence", 0.0),
                    "team_id": p.get("team_id"),
                    "role": p.get("role", "player"),
                    "is_official": p.get("is_official", False),
                }
                pid_history.setdefault(pid, []).append(entry)

        # Identify split frames for each PID from conflicts
        pid_split_frames: Dict[str, Set[int]] = {}
        for cw in conflict_windows:
            for pid in cw.get("pids", []):
                pid_split_frames.setdefault(pid, set()).add(cw.get("start_frame", -1))

        tracklets: List[Tracklet] = []
        tracklet_counter = 0

        for pid, history in pid_history.items():
            history.sort(key=lambda h: h["frame"])
            splits = sorted(pid_split_frames.get(pid, set()))

            # Split history into segments
            segments: List[List[dict]] = []
            current_segment: List[dict] = []

            for entry in history:
                # Check if this frame crosses a split point
                if splits and current_segment:
                    last_frame = current_segment[-1]["frame"]
                    for sp in splits:
                        if last_frame < sp <= entry["frame"]:
                            segments.append(current_segment)
                            current_segment = []
                            break
                current_segment.append(entry)

            if current_segment:
                segments.append(current_segment)

            for seg in segments:
                if not seg:
                    continue
                tracklet_counter += 1

                # Compute stats
                sources = [e.get("assignment_source", "unknown") for e in seg]
                source_counter = Counter(sources)
                confs = [e.get("identity_confidence", 0.0) for e in seg]
                team_ids = [e.get("team_id") for e in seg if e.get("team_id") is not None]
                team_mode = Counter(team_ids).most_common(1)[0][0] if team_ids else None
                roles = [e.get("role", "player") for e in seg]
                role_mode = Counter(roles).most_common(1)[0][0] if roles else "player"

                # Velocity
                velocities = []
                for i in range(1, len(seg)):
                    dt = seg[i]["frame"] - seg[i - 1]["frame"]
                    if dt > 0:
                        c1 = _bbox_center(seg[i - 1]["bbox"])
                        c2 = _bbox_center(seg[i]["bbox"])
                        velocities.append(_euclidean(c1, c2) / dt)
                mean_vel = sum(velocities) / len(velocities) if velocities else 0.0

                # rawTrackId mode
                raw_tids = [e.get("rawTrackId", e.get("trackId", -1)) for e in seg]
                raw_tid_mode = Counter(raw_tids).most_common(1)[0][0] if raw_tids else -1

                tracklets.append(Tracklet(
                    tracklet_id=f"tl_{tracklet_counter:05d}",
                    pid=pid,
                    raw_track_id=raw_tid_mode,
                    start_frame=seg[0]["frame"],
                    end_frame=seg[-1]["frame"],
                    team_id_mode=team_mode,
                    role_mode=role_mode,
                    first_bbox=seg[0]["bbox"],
                    last_bbox=seg[-1]["bbox"],
                    mean_velocity=mean_vel,
                    assignment_sources=list(source_counter.keys()),
                    confidence_stats={
                        "min": round(min(confs), 3) if confs else 0.0,
                        "max": round(max(confs), 3) if confs else 0.0,
                        "mean": round(sum(confs) / len(confs), 3) if confs else 0.0,
                    },
                    frame_count=len(seg),
                    entries=seg,
                ))

        return tracklets

    def propose_corrections(
        self,
        tracklets: List[Tracklet],
        conflict_windows: List[dict],
    ) -> List[CorrectionProposal]:
        """
        Propose deterministic correction actions based on tracklet analysis
        and conflict windows.
        """
        proposals: List[CorrectionProposal] = []

        # Index tracklets by PID
        pid_tracklets: Dict[str, List[Tracklet]] = {}
        for tl in tracklets:
            pid_tracklets.setdefault(tl.pid, []).append(tl)

        for cw in conflict_windows:
            window_id = cw.get("window_id", "unknown")
            ctypes = cw.get("conflict_types", [])
            pids = cw.get("pids", [])
            start_frame = cw.get("start_frame", 0)
            end_frame = cw.get("end_frame", 0)
            evidence = cw.get("evidence", {})

            # TEAM_FLIP with two PIDs → propose swap
            if "TEAM_FLIP" in ctypes and len(pids) >= 2:
                proposals.append(CorrectionProposal(
                    action=CorrectionAction.SWAP_PID_AFTER_FRAME.value,
                    window_id=window_id,
                    pid_a=pids[0],
                    pid_b=pids[1],
                    apply_from_frame=start_frame,
                    confidence=0.80,
                    reason_codes=["TEAM_FLIP", "VELOCITY_CONTINUITY"],
                ))

            # GOALKEEPER_TO_OUTFIELD → mark unknown after frame
            if "GOALKEEPER_TO_OUTFIELD" in ctypes and len(pids) >= 1:
                proposals.append(CorrectionProposal(
                    action=CorrectionAction.MARK_UNKNOWN.value,
                    window_id=window_id,
                    pid_a=pids[0],
                    apply_from_frame=end_frame,
                    confidence=0.95,
                    reason_codes=["GOALKEEPER_TO_OUTFIELD"],
                ))

            # DUPLICATE_PID → suppress the lower-confidence duplicate
            if "DUPLICATE_PID" in ctypes and len(pids) >= 1:
                proposals.append(CorrectionProposal(
                    action=CorrectionAction.SUPPRESS_PID.value,
                    window_id=window_id,
                    pid_a=pids[0],
                    apply_from_frame=start_frame,
                    apply_to_frame=end_frame,
                    confidence=0.90,
                    reason_codes=["DUPLICATE_PID"],
                ))

            # PID_TELEPORT with single PID → reject the revival
            if "PID_TELEPORT" in ctypes and len(pids) >= 1:
                proposals.append(CorrectionProposal(
                    action=CorrectionAction.REJECT_REVIVAL.value,
                    window_id=window_id,
                    pid_a=pids[0],
                    apply_from_frame=end_frame,
                    confidence=0.85,
                    reason_codes=["PID_TELEPORT"],
                ))

            # TRACKLET_FRAGMENTATION → merge
            if "TRACKLET_FRAGMENTATION" in ctypes and len(pids) >= 1:
                pid = pids[0]
                tls = pid_tracklets.get(pid, [])
                if len(tls) >= 2:
                    proposals.append(CorrectionProposal(
                        action=CorrectionAction.MERGE_TRACKLETS.value,
                        window_id=window_id,
                        pid_a=pid,
                        tracklet_id=tls[0].tracklet_id,
                        confidence=0.75,
                        reason_codes=["TRACKLET_FRAGMENTATION"],
                    ))

        return proposals

    def resolve(
        self,
        track_results: dict,
        conflict_windows: List[dict],
    ) -> dict:
        """
        Full resolution pipeline: build tracklets → propose corrections.
        Returns a patch plan.
        """
        tracklets = self.build_tracklets(track_results, conflict_windows)
        proposals = self.propose_corrections(tracklets, conflict_windows)

        return {
            "tracklets": [t.to_dict() for t in tracklets],
            "corrections": [p.to_dict() for p in proposals],
            "patches_proposed": len(proposals),
        }
