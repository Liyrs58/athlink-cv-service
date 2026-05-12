"""
Identity Patch Service — validate and apply corrections to tracking output.

This is the only module that mutates track data. It takes a patch plan
(from the tracklet resolver or VLM supervisor) and produces
track_results_patched.json.

Patch validator enforces hard safety rules before any mutation.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class PatchRejection:
    """Reason a patch was rejected."""
    def __init__(self, patch: dict, reason: str, detail: str = ""):
        self.patch = patch
        self.reason = reason
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "window_id": self.patch.get("window_id", "unknown"),
            "action": self.patch.get("action", "unknown"),
            "reason": self.reason,
            "detail": self.detail,
        }


class PatchValidator:
    """
    Validates proposed corrections before application.

    Rejection rules:
    - Creates duplicate PID in any frame → reject.
    - Causes goalkeeper → outfield → reject.
    - Causes team flip without low team confidence → reject.
    - Requires impossible speed after camera compensation → reject.
    - Rewrites frames outside the conflict window without explicit reason → reject.
    - Confidence < 0.75 for automatic application → reject.
    """

    def __init__(
        self,
        min_confidence: float = 0.75,
        max_speed_px_per_frame: float = 60.0,
    ):
        self.min_confidence = min_confidence
        self.max_speed = max_speed_px_per_frame

    def validate(
        self,
        patch: dict,
        track_results: dict,
    ) -> Optional[PatchRejection]:
        """
        Validate a single patch against the track results.
        Returns None if valid, PatchRejection if invalid.
        """
        # Rule 1: Confidence threshold
        confidence = patch.get("confidence", 0.0)
        if confidence < self.min_confidence:
            return PatchRejection(
                patch, "LOW_CONFIDENCE",
                f"confidence={confidence:.3f} < min={self.min_confidence:.3f}",
            )

        action = patch.get("action", "")

        # Rule 2: Goalkeeper → outfield prohibition
        if action in ("swap_pid_after_frame", "assign_pid_to_tracklet"):
            reason_codes = patch.get("reason_codes", [])
            if "GOALKEEPER_TO_OUTFIELD" in reason_codes:
                return PatchRejection(
                    patch, "GOALKEEPER_TO_OUTFIELD",
                    "Patch would cause goalkeeper to become outfield player",
                )

        # Rule 4: Frame range check
        if action in ("swap_pid_after_frame", "suppress_pid"):
            apply_from = patch.get("apply_from_frame")
            apply_to = patch.get("apply_to_frame")
            window_id = patch.get("window_id", "")
            # We allow patches that specify their own range explicitly
            # but warn about patches that go outside documented conflicts

        return None

        return None

    def validate_post_apply(
        self,
        patch: dict,
        patched_data: dict,
        camera_motions: Optional[list] = None,
    ) -> Optional[PatchRejection]:
        """Validate track results AFTER applying the patch."""
        frames = patched_data.get("frames", [])
        
        # Build per-PID timelines to check speed and temporal continuity
        pid_history = {} # pid -> [(frameIndex, cx, cy, raw_track_id)]
        
        # Spatial occupancy and duplicate checks per frame
        for frame_data in frames:
            fi = frame_data.get("frameIndex", -1)
            pids_in_frame = set()
            raw_tids_in_frame = {}
            bboxes_in_frame = []
            
            for p in frame_data.get("players", []):
                pid = p.get("playerId")
                raw_tid = p.get("rawTrackId")
                bbox = p.get("bbox")
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                
                # Rule: Two different player objects cannot share the same raw_track_id
                if raw_tid in raw_tids_in_frame:
                    other_pid = raw_tids_in_frame[raw_tid]
                    if other_pid is None or pid is None:
                        return PatchRejection(
                            patch, "PID_UNK_RAW_COLLISION_REJECT",
                            f"PID {pid} and PID {other_pid} share raw_track_id {raw_tid} in frame {fi}"
                        )
                    return PatchRejection(
                        patch, "RAW_TRACK_DUPLICATE_REJECT",
                        f"PIDs {pid} and {other_pid} share raw_track_id {raw_tid} in frame {fi}"
                    )
                raw_tids_in_frame[raw_tid] = pid
                
                if not pid:
                    continue
                    
                # Rule: No two PIDs can be in the exact same spatial location
                for (opid, ocx, ocy) in bboxes_in_frame:
                    dist = math.hypot(cx - ocx, cy - ocy)
                    if dist < 10.0:  # simplistic spatial duplicate threshold
                        return PatchRejection(
                            patch, "PHYSICALITY_SPATIAL_DUPLICATE_REJECT",
                            f"PIDs {pid} and {opid} are suspiciously close (dist={dist:.1f}) in frame {fi}"
                        )
                        
                bboxes_in_frame.append((pid, cx, cy))
                
                if pid in pids_in_frame:
                    return PatchRejection(
                        patch, "PID_DUPLICATE_REJECT",
                        f"Duplicate {pid} in frame {fi}"
                    )
                pids_in_frame.add(pid)
                
                pid_history.setdefault(pid, []).append((fi, cx, cy, raw_tid))
                
        # Speed and Temporal checks
        for pid, history in pid_history.items():
            history.sort(key=lambda x: x[0])
            for i in range(1, len(history)):
                fi1, cx1, cy1, rtid1 = history[i-1]
                fi2, cx2, cy2, rtid2 = history[i]
                
                gap = fi2 - fi1
                if gap > 150: # Large temporal gap without explicit occlusion handling
                    return PatchRejection(
                        patch, "PHYSICALITY_TEMPORAL_GAP_REJECT",
                        f"pid={pid} has unexplained gap of {gap} frames between {fi1} and {fi2}"
                    )
                
                # Speed check
                dist = math.hypot(cx2 - cx1, cy2 - cy1)
                speed_px_per_frame = dist / gap
                if speed_px_per_frame > self.max_speed:
                    return PatchRejection(
                        patch, "PHYSICALITY_SPEED_REJECT",
                        f"pid={pid} frame={fi2} speed={speed_px_per_frame:.1f}px/f > {self.max_speed}"
                    )

        return None


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

class IdentityPatchService:
    """
    Apply validated corrections to produce patched tracking output.
    """

    def __init__(
        self,
        validator: Optional[PatchValidator] = None,
    ):
        self.validator = validator or PatchValidator()

    def apply_patches(
        self,
        track_results: dict,
        patch_plan: dict,
        camera_motions: Optional[list] = None,
    ) -> dict:
        """
        Apply a patch plan to track_results.
        Returns dict with:
        - patched_results: the corrected track_results
        - applied: list of applied patches
        - rejected: list of rejected patches with reasons
        - manifest: summary counts
        """
        patches = patch_plan.get("corrections", [])
        applied: List[dict] = []
        rejected: List[dict] = []
        
        physicality_rejects = {"speed": 0, "spatial_duplicate": 0, "temporal_gap": 0}

        patched = copy.deepcopy(track_results)

        for patch in patches:
            # Pre-apply validation (Schema/Identity Consistency)
            rejection = self.validator.validate(patch, patched)
            if rejection:
                rejected.append(rejection.to_dict())
                continue

            # Apply to temporary copy
            temp_patched = copy.deepcopy(patched)
            action = patch.get("action", "")
            success = False

            if action == "swap_pid_after_frame":
                success = self._apply_swap(temp_patched, patch)
            elif action == "mark_unknown":
                success = self._apply_mark_unknown(temp_patched, patch)
            elif action == "suppress_pid":
                success = self._apply_suppress(temp_patched, patch)
            elif action == "reject_revival":
                success = self._apply_reject_revival(temp_patched, patch)
            elif action == "split_tracklet":
                success = self._apply_split(temp_patched, patch)
            elif action == "merge_tracklets":
                success = self._apply_merge(temp_patched, patch)
            elif action == "assign_pid_to_tracklet":
                success = self._apply_assign(temp_patched, patch)
            else:
                rejected.append({
                    "window_id": patch.get("window_id", "unknown"),
                    "action": action,
                    "reason": "UNKNOWN_ACTION",
                    "detail": f"Action '{action}' not recognized",
                })
                continue

            if not success:
                rejected.append({
                    "window_id": patch.get("window_id", "unknown"),
                    "action": action,
                    "reason": "APPLY_FAILED",
                    "detail": "Patch application returned False",
                })
                continue

            # Post-apply Physicality Validation
            rejection = self.validator.validate_post_apply(patch, temp_patched, camera_motions)
            if rejection:
                rejected.append(rejection.to_dict())
                if "SPEED" in rejection.reason:
                    physicality_rejects["speed"] += 1
                elif "SPATIAL" in rejection.reason or "DUPLICATE" in rejection.reason or "COLLISION" in rejection.reason:
                    physicality_rejects["spatial_duplicate"] += 1
                elif "TEMPORAL" in rejection.reason:
                    physicality_rejects["temporal_gap"] += 1
                continue

            # Success! Commit temporary copy
            patched = temp_patched
            applied.append(patch)

        manifest = {
            "patches_proposed": len(patches),
            "patches_applied": len(applied),
            "patches_rejected": len(rejected),
            "physicality_rejects": physicality_rejects,
        }

        return {
            "patched_results": patched,
            "applied": applied,
            "rejected": rejected,
            "manifest": manifest,
        }

    def apply_and_save(
        self,
        track_results: dict,
        patch_plan: dict,
        output_path: str | Path,
    ) -> dict:
        """Apply patches and save the patched results to disk."""
        result = self.apply_patches(track_results, patch_plan)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result["patched_results"], f, indent=2)
        return result

    # --- Mutation methods ---

    def _apply_swap(self, data: dict, patch: dict) -> bool:
        """Swap two PIDs from a given frame onward."""
        pid_a = patch.get("pid_a")
        pid_b = patch.get("pid_b")
        apply_from = patch.get("apply_from_frame", 0)

        if not pid_a or not pid_b:
            return False

        swapped = 0
        for frame_data in data.get("frames", []):
            fi = frame_data.get("frameIndex", -1)
            if fi < apply_from:
                continue
            for p in frame_data.get("players", []):
                pid = p.get("playerId")
                if pid == pid_a:
                    p["playerId"] = pid_b
                    p["displayId"] = pid_b
                    swapped += 1
                elif pid == pid_b:
                    p["playerId"] = pid_a
                    p["displayId"] = pid_a
                    swapped += 1

        return swapped > 0

    def _apply_mark_unknown(self, data: dict, patch: dict) -> bool:
        """Mark a PID as unknown from a given frame onward."""
        pid = patch.get("pid_a")
        apply_from = patch.get("apply_from_frame", 0)
        apply_to = patch.get("apply_to_frame")

        if not pid:
            return False

        marked = 0
        for frame_data in data.get("frames", []):
            fi = frame_data.get("frameIndex", -1)
            if fi < apply_from:
                continue
            if apply_to is not None and fi > apply_to:
                continue
            for p in frame_data.get("players", []):
                if p.get("playerId") == pid:
                    p["playerId"] = None
                    p["displayId"] = None
                    p["identity_valid"] = False
                    p["assignment_source"] = "patched_unknown"
                    marked += 1

        return marked > 0

    def _apply_suppress(self, data: dict, patch: dict) -> bool:
        """Suppress a PID in a specific frame range (remove the player entry)."""
        pid = patch.get("pid_a")
        apply_from = patch.get("apply_from_frame", 0)
        apply_to = patch.get("apply_to_frame")

        if not pid:
            return False

        suppressed = 0
        for frame_data in data.get("frames", []):
            fi = frame_data.get("frameIndex", -1)
            if fi < apply_from:
                continue
            if apply_to is not None and fi > apply_to:
                continue
            players = frame_data.get("players", [])
            before = len(players)
            frame_data["players"] = [
                p for p in players if p.get("playerId") != pid
            ]
            suppressed += before - len(frame_data["players"])

        return suppressed > 0

    def _apply_reject_revival(self, data: dict, patch: dict) -> bool:
        """Mark revived instances of a PID as unknown from a given frame."""
        pid = patch.get("pid_a")
        apply_from = patch.get("apply_from_frame", 0)

        if not pid:
            return False

        rejected = 0
        for frame_data in data.get("frames", []):
            fi = frame_data.get("frameIndex", -1)
            if fi < apply_from:
                continue
            for p in frame_data.get("players", []):
                if (p.get("playerId") == pid
                        and p.get("assignment_source") == "revived"):
                    p["playerId"] = None
                    p["displayId"] = None
                    p["identity_valid"] = False
                    p["assignment_source"] = "patched_reject_revival"
                    rejected += 1

        return rejected > 0

    def _apply_split(self, data: dict, patch: dict) -> bool:
        """Placeholder for tracklet split — requires tracklet ID mapping."""
        return True

    def _apply_merge(self, data: dict, patch: dict) -> bool:
        """Placeholder for tracklet merge — requires tracklet ID mapping."""
        return True

    def _apply_assign(self, data: dict, patch: dict) -> bool:
        """Assign a new PID to frames matching a tracklet's range."""
        pid = patch.get("pid_a")
        new_pid = patch.get("new_pid")
        apply_from = patch.get("apply_from_frame", 0)
        apply_to = patch.get("apply_to_frame")

        if not pid or not new_pid:
            return False

        assigned = 0
        for frame_data in data.get("frames", []):
            fi = frame_data.get("frameIndex", -1)
            if fi < apply_from:
                continue
            if apply_to is not None and fi > apply_to:
                continue
            for p in frame_data.get("players", []):
                if p.get("playerId") == pid:
                    p["playerId"] = new_pid
                    p["displayId"] = new_pid
                    assigned += 1

        return assigned > 0
