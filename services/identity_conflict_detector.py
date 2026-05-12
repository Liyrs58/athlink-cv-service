"""
Identity Conflict Detector — deterministic post-hoc scan of tracking output.

Scans track_results.json frame-by-frame and emits typed conflict windows
that downstream components (tracklet resolver, VLM supervisor, patch service)
can consume.

Architecture rule: this module is **read-only** over raw tracking output.
It never mutates tracks. It only emits conflict descriptors.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Conflict taxonomy
# ---------------------------------------------------------------------------

class ConflictType(str, Enum):
    TEAM_FLIP = "TEAM_FLIP"
    ROLE_FLIP = "ROLE_FLIP"
    GOALKEEPER_TO_OUTFIELD = "GOALKEEPER_TO_OUTFIELD"
    DUPLICATE_PID = "DUPLICATE_PID"
    OVERLAP_GT_0_30 = "OVERLAP_GT_0_30"
    CAMERA_COMPENSATED_CENTROID_JUMP = "CAMERA_COMPENSATED_CENTROID_JUMP"
    IMPLAUSIBLE_SPEED = "IMPLAUSIBLE_SPEED"
    REV_RELOCK_AFTER_OCCLUSION = "REV_RELOCK_AFTER_OCCLUSION"
    PID_TELEPORT = "PID_TELEPORT"
    TRACKLET_FRAGMENTATION = "TRACKLET_FRAGMENTATION"
    IDENTITY_SOURCE_DOWNGRADE = "IDENTITY_SOURCE_DOWNGRADE"
    HIGH_CONGESTION_RELOCK = "HIGH_CONGESTION_RELOCK"


# ---------------------------------------------------------------------------
# Evidence & Window containers
# ---------------------------------------------------------------------------

@dataclass
class ConflictEvidence:
    team_before: Optional[str] = None
    team_after: Optional[str] = None
    role_before: Optional[str] = None
    role_after: Optional[str] = None
    max_overlap_iou: float = 0.0
    max_speed_mps: float = 0.0
    centroid_jump_px: float = 0.0
    motion_class: Optional[str] = None
    source_before: Optional[str] = None
    source_after: Optional[str] = None
    duplicate_frame: Optional[int] = None


@dataclass
class ConflictWindow:
    window_id: str
    start_frame: int
    end_frame: int
    pids: List[str]
    raw_track_ids: List[int]
    conflict_types: List[str]
    severity: str = "low"  # low | medium | high
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "pids": self.pids,
            "raw_track_ids": self.raw_track_ids,
            "conflict_types": self.conflict_types,
            "severity": self.severity,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_center(bbox: list) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _severity(types: List[str]) -> str:
    """Derive severity from conflict type mix."""
    high = {
        ConflictType.TEAM_FLIP, ConflictType.ROLE_FLIP,
        ConflictType.GOALKEEPER_TO_OUTFIELD, ConflictType.PID_TELEPORT,
    }
    medium = {
        ConflictType.DUPLICATE_PID, ConflictType.IMPLAUSIBLE_SPEED,
        ConflictType.HIGH_CONGESTION_RELOCK,
        ConflictType.CAMERA_COMPENSATED_CENTROID_JUMP,
    }
    for t in types:
        if t in {ct.value for ct in high}:
            return "high"
    for t in types:
        if t in {ct.value for ct in medium}:
            return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Team / role inference from tracking data
# ---------------------------------------------------------------------------

def _infer_team_from_class(cls_val: int) -> Optional[str]:
    """
    Class mapping from YOLO:
      0 = person (generic)
      2 = player (our main class)
    No reliable team info from class alone — we'll use HSV / bbox position
    heuristics or the team_id field if present.
    """
    return None


def _infer_role(player: dict) -> str:
    """Infer role from player data.  is_official=True → referee."""
    if player.get("is_official"):
        return "referee"
    if player.get("role"):
        return player["role"]
    return "player"


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class IdentityConflictDetector:
    """
    Deterministic scan of tracking output for identity conflicts.

    Parameters
    ----------
    iou_threshold : float
        IoU above which two bboxes create an overlap conflict. Default 0.30.
    speed_threshold_px_per_frame : float
        Max plausible centroid movement per frame in pixels. Default 60.
    centroid_jump_threshold : float
        Camera-compensated centroid jump threshold in pixels. Default 150.
    team_confidence_low : float
        Below this, team assignment is considered unreliable. Default 0.6.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        speed_threshold_px_per_frame: float = 60.0,
        centroid_jump_threshold: float = 150.0,
        team_confidence_low: float = 0.6,
    ):
        self.iou_threshold = iou_threshold
        self.speed_threshold = speed_threshold_px_per_frame
        self.centroid_jump_threshold = centroid_jump_threshold
        self.team_confidence_low = team_confidence_low

    def detect(
        self,
        track_results: dict,
        camera_motion: Optional[dict] = None,
        identity_metrics: Optional[dict] = None,
    ) -> List[ConflictWindow]:
        """
        Scan all frames and return a list of ConflictWindow objects.

        Parameters
        ----------
        track_results : dict
            The parsed track_results.json with keys: jobId, frames, total_frames.
        camera_motion : dict, optional
            Camera motion data keyed by frame index.
        identity_metrics : dict, optional
            Global identity metrics (informational).
        """
        frames = track_results.get("frames", [])
        conflicts: List[ConflictWindow] = []
        window_counter = 0

        # Build per-PID history for temporal checks
        pid_history: Dict[str, List[dict]] = {}  # pid -> [{frame, bbox, ...}]
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
                    "class": p.get("class"),
                    "is_official": p.get("is_official", False),
                    "team_id": p.get("team_id"),
                    "role": _infer_role(p),
                }
                pid_history.setdefault(pid, []).append(entry)

        # Camera motion dict keyed by frame
        cam_by_frame: Dict[int, dict] = {}
        if camera_motion:
            if isinstance(camera_motion, list):
                for entry in camera_motion:
                    cam_by_frame[entry.get("frame", -1)] = entry
            elif isinstance(camera_motion, dict):
                if "frames" in camera_motion:
                    for entry in camera_motion["frames"]:
                        cam_by_frame[entry.get("frame", -1)] = entry
                else:
                    cam_by_frame = camera_motion

        # --- Check 1: Duplicate PID in same frame ---
        for frame_data in frames:
            fi = frame_data.get("frameIndex", -1)
            pid_counts: Dict[str, list] = {}
            for p in frame_data.get("players", []):
                pid = p.get("playerId")
                if pid is None:
                    continue
                pid_counts.setdefault(pid, []).append(p)

            for pid, plist in pid_counts.items():
                if len(plist) > 1:
                    tids = [p.get("rawTrackId", p.get("trackId", -1)) for p in plist]
                    window_counter += 1
                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=fi,
                        end_frame=fi,
                        pids=[pid],
                        raw_track_ids=tids,
                        conflict_types=[ConflictType.DUPLICATE_PID.value],
                        severity="medium",
                        evidence={"duplicate_frame": fi, "count": len(plist)},
                    ))

        # --- Check 2: Pairwise overlap in each frame ---
        for frame_data in frames:
            fi = frame_data.get("frameIndex", -1)
            players = [p for p in frame_data.get("players", []) if p.get("playerId")]
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    a, b = players[i], players[j]
                    iou = _bbox_iou(
                        a.get("bbox", [0, 0, 0, 0]),
                        b.get("bbox", [0, 0, 0, 0]),
                    )
                    if iou > self.iou_threshold:
                        window_counter += 1
                        conflicts.append(ConflictWindow(
                            window_id=f"conflict_{window_counter:05d}",
                            start_frame=fi,
                            end_frame=fi,
                            pids=[a["playerId"], b["playerId"]],
                            raw_track_ids=[
                                a.get("rawTrackId", a.get("trackId", -1)),
                                b.get("rawTrackId", b.get("trackId", -1)),
                            ],
                            conflict_types=[ConflictType.OVERLAP_GT_0_30.value],
                            severity="low",
                            evidence={"max_overlap_iou": round(iou, 4)},
                        ))

        # --- Check 3: Per-PID temporal checks ---
        for pid, history in pid_history.items():
            history.sort(key=lambda h: h["frame"])

            for k in range(1, len(history)):
                prev = history[k - 1]
                curr = history[k]
                dt_frames = curr["frame"] - prev["frame"]
                if dt_frames <= 0:
                    continue

                # 3a: Team flip
                if (prev.get("team_id") is not None
                        and curr.get("team_id") is not None
                        and prev["team_id"] != curr["team_id"]):
                    window_counter += 1
                    evidence = {
                        "team_before": str(prev["team_id"]),
                        "team_after": str(curr["team_id"]),
                    }

                    ctypes = [ConflictType.TEAM_FLIP.value]

                    # Check for goalkeeper -> outfield
                    if prev.get("role") == "goalkeeper" and curr.get("role") != "goalkeeper":
                        ctypes.append(ConflictType.GOALKEEPER_TO_OUTFIELD.value)
                        evidence["role_before"] = "goalkeeper"
                        evidence["role_after"] = curr.get("role", "player")

                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=prev["frame"],
                        end_frame=curr["frame"],
                        pids=[pid],
                        raw_track_ids=[
                            prev.get("rawTrackId", -1),
                            curr.get("rawTrackId", -1),
                        ],
                        conflict_types=ctypes,
                        severity=_severity(ctypes),
                        evidence=evidence,
                    ))

                # 3b: Role flip (non-team)
                if (prev.get("role") and curr.get("role")
                        and prev["role"] != curr["role"]):
                    if prev["role"] == "goalkeeper" and curr["role"] != "goalkeeper":
                        window_counter += 1
                        conflicts.append(ConflictWindow(
                            window_id=f"conflict_{window_counter:05d}",
                            start_frame=prev["frame"],
                            end_frame=curr["frame"],
                            pids=[pid],
                            raw_track_ids=[
                                prev.get("rawTrackId", -1),
                                curr.get("rawTrackId", -1),
                            ],
                            conflict_types=[ConflictType.GOALKEEPER_TO_OUTFIELD.value],
                            severity="high",
                            evidence={
                                "role_before": prev["role"],
                                "role_after": curr["role"],
                            },
                        ))

                # 3c: Centroid jump / implausible speed
                prev_center = _bbox_center(prev["bbox"])
                curr_center = _bbox_center(curr["bbox"])
                dist = _euclidean(prev_center, curr_center)
                speed = dist / max(dt_frames, 1)

                # Camera compensation
                cam = cam_by_frame.get(curr["frame"], {})
                compensated_dist = dist
                if cam:
                    dx = cam.get("dx", 0.0)
                    dy = cam.get("dy", 0.0)
                    comp_prev = (prev_center[0] + dx, prev_center[1] + dy)
                    compensated_dist = _euclidean(comp_prev, curr_center)

                if compensated_dist > self.centroid_jump_threshold:
                    window_counter += 1
                    motion_class = cam.get("motion_class", "unknown")
                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=prev["frame"],
                        end_frame=curr["frame"],
                        pids=[pid],
                        raw_track_ids=[
                            prev.get("rawTrackId", -1),
                            curr.get("rawTrackId", -1),
                        ],
                        conflict_types=[ConflictType.CAMERA_COMPENSATED_CENTROID_JUMP.value],
                        severity="medium",
                        evidence={
                            "centroid_jump_px": round(compensated_dist, 1),
                            "motion_class": motion_class,
                        },
                    ))

                if speed > self.speed_threshold:
                    window_counter += 1
                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=prev["frame"],
                        end_frame=curr["frame"],
                        pids=[pid],
                        raw_track_ids=[
                            prev.get("rawTrackId", -1),
                            curr.get("rawTrackId", -1),
                        ],
                        conflict_types=[ConflictType.IMPLAUSIBLE_SPEED.value],
                        severity="medium",
                        evidence={"max_speed_mps": round(speed, 2)},
                    ))

                # 3d: Identity source downgrade (locked -> revived/unassigned)
                source_rank = {"locked": 3, "revived": 2, "provisional": 1, "unassigned": 0, "unknown": 0}
                prev_rank = source_rank.get(prev.get("assignment_source", "unknown"), 0)
                curr_rank = source_rank.get(curr.get("assignment_source", "unknown"), 0)
                if prev_rank >= 3 and curr_rank <= 1:
                    window_counter += 1
                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=prev["frame"],
                        end_frame=curr["frame"],
                        pids=[pid],
                        raw_track_ids=[
                            prev.get("rawTrackId", -1),
                            curr.get("rawTrackId", -1),
                        ],
                        conflict_types=[ConflictType.IDENTITY_SOURCE_DOWNGRADE.value],
                        severity="low",
                        evidence={
                            "source_before": prev.get("assignment_source"),
                            "source_after": curr.get("assignment_source"),
                        },
                    ))

                # 3e: REV -> LOCK after occlusion (high congestion relock)
                if (prev.get("assignment_source") == "revived"
                        and curr.get("assignment_source") == "locked"
                        and dt_frames <= 10):
                    # Check if there was overlap in the gap frames
                    window_counter += 1
                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=prev["frame"],
                        end_frame=curr["frame"],
                        pids=[pid],
                        raw_track_ids=[
                            prev.get("rawTrackId", -1),
                            curr.get("rawTrackId", -1),
                        ],
                        conflict_types=[ConflictType.REV_RELOCK_AFTER_OCCLUSION.value],
                        severity="low",
                        evidence={
                            "source_before": "revived",
                            "source_after": "locked",
                            "gap_frames": dt_frames,
                        },
                    ))

                # 3f: PID teleport (rawTrackId change with large distance)
                if (prev.get("rawTrackId") != curr.get("rawTrackId")
                        and compensated_dist > self.centroid_jump_threshold * 0.5):
                    window_counter += 1
                    conflicts.append(ConflictWindow(
                        window_id=f"conflict_{window_counter:05d}",
                        start_frame=prev["frame"],
                        end_frame=curr["frame"],
                        pids=[pid],
                        raw_track_ids=[
                            prev.get("rawTrackId", -1),
                            curr.get("rawTrackId", -1),
                        ],
                        conflict_types=[ConflictType.PID_TELEPORT.value],
                        severity="high",
                        evidence={
                            "centroid_jump_px": round(compensated_dist, 1),
                            "raw_tid_before": prev.get("rawTrackId"),
                            "raw_tid_after": curr.get("rawTrackId"),
                        },
                    ))

        return conflicts

    def detect_and_save(
        self,
        track_results: dict,
        output_path: str | Path,
        camera_motion: Optional[dict] = None,
        identity_metrics: Optional[dict] = None,
    ) -> dict:
        """Run detection and write identity_conflicts.json. Returns manifest dict."""
        conflicts = self.detect(track_results, camera_motion, identity_metrics)
        result = {
            "job_id": track_results.get("jobId", "unknown"),
            "conflict_windows_total": len(conflicts),
            "high_severity_conflicts": sum(1 for c in conflicts if c.severity == "high"),
            "medium_severity_conflicts": sum(1 for c in conflicts if c.severity == "medium"),
            "team_flip_conflicts": sum(
                1 for c in conflicts if ConflictType.TEAM_FLIP.value in c.conflict_types
            ),
            "role_flip_conflicts": sum(
                1 for c in conflicts if ConflictType.ROLE_FLIP.value in c.conflict_types
                or ConflictType.GOALKEEPER_TO_OUTFIELD.value in c.conflict_types
            ),
            "duplicate_pid_conflicts": sum(
                1 for c in conflicts if ConflictType.DUPLICATE_PID.value in c.conflict_types
            ),
            "windows": [c.to_dict() for c in conflicts],
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return result
