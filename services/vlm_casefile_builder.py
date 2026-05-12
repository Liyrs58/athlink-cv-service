"""
VLM Case-File Builder — export unresolved conflicts as case files
for an optional VLM (Vision-Language Model) supervisor.

For each unresolved medium/high severity conflict, this module exports:
  - before/during/after crops for involved players
  - contact sheet (composite image)
  - velocity vectors, team IDs, role labels
  - candidate correction actions
  - machine-readable case.json

No live API calls are made. This builds the **offline data contract** only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Case file structure
# ---------------------------------------------------------------------------

VLM_RESPONSE_SCHEMA = {
    "window_id": "string",
    "decision": "accept_patch | reject_patch | needs_human",
    "action": "string (correction action to apply)",
    "confidence": "float 0.0-1.0",
    "reason_codes": ["list of reason code strings"],
    "analyst_summary": "short explanation",
}


def _bbox_center(bbox: list) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class VLMCaseFileBuilder:
    """
    Build case files for VLM review of unresolved identity conflicts.
    """

    def __init__(self, video_path: Optional[str] = None):
        self.video_path = video_path

    def build_case_files(
        self,
        conflict_windows: List[dict],
        track_results: dict,
        corrections: List[dict],
        output_dir: str | Path,
        severity_filter: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Build case files for conflicts matching the severity filter.
        Returns list of case file paths created.
        """
        if severity_filter is None:
            severity_filter = ["medium", "high"]

        output_dir = Path(output_dir)
        case_paths: List[str] = []

        frames_by_index: Dict[int, dict] = {}
        for f in track_results.get("frames", []):
            frames_by_index[f.get("frameIndex", -1)] = f

        # Index corrections by window_id
        corrections_by_window: Dict[str, List[dict]] = {}
        for c in corrections:
            wid = c.get("window_id", "unknown")
            corrections_by_window.setdefault(wid, []).append(c)

        for cw in conflict_windows:
            severity = cw.get("severity", "low")
            if severity not in severity_filter:
                continue

            window_id = cw.get("window_id", "unknown")
            case_dir = output_dir / window_id
            case_dir.mkdir(parents=True, exist_ok=True)

            # Build the case JSON
            case_data = self._build_case_json(
                cw, frames_by_index,
                corrections_by_window.get(window_id, []),
            )

            case_path = case_dir / "case.json"
            with open(case_path, "w") as f:
                json.dump(case_data, f, indent=2)
            case_paths.append(str(case_path))

            # Build contact sheet if video is available
            if self.video_path and HAS_CV2:
                try:
                    self._build_contact_sheet(cw, case_dir)
                except Exception as e:
                    print(f"[VLM] Contact sheet failed for {window_id}: {e}")

        return case_paths

    def _build_case_json(
        self,
        conflict: dict,
        frames_by_index: Dict[int, dict],
        candidate_corrections: List[dict],
    ) -> dict:
        """Build the machine-readable case.json for a single conflict."""
        window_id = conflict.get("window_id", "unknown")
        start_frame = conflict.get("start_frame", 0)
        end_frame = conflict.get("end_frame", 0)
        pids = conflict.get("pids", [])

        # Extract player data around the conflict
        context_frames = []
        for offset in [-10, -5, 0]:
            fi = start_frame + offset
            frame = frames_by_index.get(fi)
            if frame:
                players = [
                    p for p in frame.get("players", [])
                    if p.get("playerId") in pids
                ]
                context_frames.append({
                    "frameIndex": fi,
                    "label": "before" if offset < 0 else "conflict_start",
                    "players": players,
                })

        # During conflict
        mid_frame = (start_frame + end_frame) // 2
        frame = frames_by_index.get(mid_frame)
        if frame:
            players = [
                p for p in frame.get("players", [])
                if p.get("playerId") in pids
            ]
            context_frames.append({
                "frameIndex": mid_frame,
                "label": "during",
                "players": players,
            })

        # After conflict
        for offset in [0, 5, 10]:
            fi = end_frame + offset
            frame = frames_by_index.get(fi)
            if frame:
                players = [
                    p for p in frame.get("players", [])
                    if p.get("playerId") in pids
                ]
                context_frames.append({
                    "frameIndex": fi,
                    "label": "after" if offset > 0 else "conflict_end",
                    "players": players,
                })

        # Velocity vectors for involved PIDs
        velocity_data = {}
        for pid in pids:
            positions = []
            for f in sorted(frames_by_index.values(), key=lambda x: x.get("frameIndex", 0)):
                fi = f.get("frameIndex", -1)
                if abs(fi - start_frame) > 30:
                    continue
                for p in f.get("players", []):
                    if p.get("playerId") == pid:
                        positions.append({
                            "frame": fi,
                            "center": list(_bbox_center(p.get("bbox", [0, 0, 0, 0]))),
                        })
            velocity_data[pid] = positions

        case = {
            "window_id": window_id,
            "conflict": conflict,
            "context_frames": context_frames,
            "velocity_data": velocity_data,
            "candidate_corrections": candidate_corrections,
            "vlm_prompt": self._build_prompt(conflict, candidate_corrections),
            "expected_response_schema": VLM_RESPONSE_SCHEMA,
        }

        return case

    def _build_prompt(self, conflict: dict, corrections: List[dict]) -> str:
        """Build the VLM prompt for this conflict."""
        window_id = conflict.get("window_id", "unknown")
        pids = conflict.get("pids", [])
        ctypes = conflict.get("conflict_types", [])
        evidence = conflict.get("evidence", {})

        prompt = (
            f"Identity conflict {window_id} detected in football tracking.\n"
            f"Players involved: {', '.join(pids)}\n"
            f"Conflict types: {', '.join(ctypes)}\n"
            f"Evidence: {json.dumps(evidence, indent=2)}\n\n"
            f"Candidate corrections:\n"
        )

        for c in corrections:
            prompt += f"  - {c.get('action')}: {json.dumps(c, indent=4)}\n"

        prompt += (
            "\nPlease analyze the contact sheet and context frames. "
            "Respond with strict JSON only:\n"
            "{\n"
            '  "window_id": "...",\n'
            '  "decision": "accept_patch | reject_patch | needs_human",\n'
            '  "action": "...",\n'
            '  "confidence": 0.0,\n'
            '  "reason_codes": [],\n'
            '  "analyst_summary": "short explanation"\n'
            "}\n"
        )

        return prompt

    def _build_contact_sheet(self, conflict: dict, output_dir: Path) -> None:
        """Extract frames around the conflict and build a contact sheet."""
        if not HAS_CV2 or not self.video_path:
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        start = conflict.get("start_frame", 0)
        end = conflict.get("end_frame", 0)
        sample_frames = [
            max(0, start - 10),
            max(0, start - 5),
            start,
            (start + end) // 2,
            end,
            end + 5,
            end + 10,
        ]

        crops: List = []
        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                # Resize for contact sheet
                h, w = frame.shape[:2]
                scale = min(320.0 / w, 240.0 / h)
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                crops.append(small)

        cap.release()

        if not crops:
            return

        # Build grid
        max_h = max(c.shape[0] for c in crops)
        max_w = max(c.shape[1] for c in crops)
        cols = min(4, len(crops))
        rows = math.ceil(len(crops) / cols)

        sheet = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
        for idx, crop in enumerate(crops):
            r = idx // cols
            c = idx % cols
            y0 = r * max_h
            x0 = c * max_w
            sheet[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop

        cv2.imwrite(str(output_dir / "contact_sheet.jpg"), sheet)
