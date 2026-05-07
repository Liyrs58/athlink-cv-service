"""End-to-end pipeline runner: tracking -> teams -> pitch -> analytics -> match report.

Each analytics service already exists but is only triggered via individual HTTP
routes. This module calls them in order, persists their dict outputs to the
per-feature paths the match_report aggregator expects, and finally builds the
unified match_report.json.

A single failed step is logged but does not abort the rest of the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def _ensure_frames_dir(repo_root: Path, job_id: str, video_path: str, sample: int = 60) -> Path:
    """Make sure team_service has a frames_dir with a few jpgs to crop from."""
    frames_dir = repo_root / "temp" / job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.glob("*.jpg")):
        return frames_dir

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames_dir
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // sample) if total else 5
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            cv2.imwrite(str(frames_dir / f"frame_{fi:06d}.jpg"), frame)
        fi += 1
    cap.release()
    return frames_dir


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _safe(name: str, fn: Callable, *args, **kwargs) -> Tuple[bool, Any]:
    try:
        out = fn(*args, **kwargs)
        return True, out
    except Exception as e:
        logger.warning("[FullPipeline] %s failed: %s", name, e)
        print(f"[FullPipeline] {name} failed: {e}")
        return False, None


def run_full_pipeline(
    video_path: str,
    job_id: str,
    *,
    device: str = "cuda",
    frame_stride_track: int = 1,
    frame_stride_pitch: int = 5,
    skip_tracking: bool = False,
) -> Dict[str, Any]:
    """Run every step needed to produce a complete match_report.json.

    Returns a manifest of steps and their success state.
    """
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "temp" / job_id
    base.mkdir(parents=True, exist_ok=True)

    completed: List[str] = []
    failed: List[Tuple[str, str]] = []

    def _record(name: str, ok: bool, err: Optional[str] = None) -> None:
        if ok:
            completed.append(name)
            print(f"[FullPipeline] ✓ {name}")
        else:
            failed.append((name, err or "unknown"))

    # 1. Tracking
    track_results: Optional[List[dict]] = None
    if skip_tracking and (base / "tracking" / "track_results.json").exists():
        with open(base / "tracking" / "track_results.json") as f:
            track_results = json.load(f).get("frames", [])
        _record("tracking", True)
    else:
        from services.tracker_core import run_tracking
        ok, out = _safe(
            "tracking",
            run_tracking,
            video_path=video_path, job_id=job_id,
            frame_stride=frame_stride_track, device=device,
        )
        _record("tracking", ok, None if ok else "see log")
        if ok:
            track_results = out

    # 2. Convert frames -> tracks-with-trajectory schema (used by team_service + analytics)
    converted_tracks: List[dict] = []
    if track_results:
        from services.pitch_service import _frames_to_tracks_schema
        converted = _frames_to_tracks_schema(track_results)
        converted_tracks = converted.get("tracks", [])

    # 3. Teams
    if converted_tracks:
        from services.team_service import assign_teams
        frames_dir = _ensure_frames_dir(repo_root, job_id, video_path)
        team_dir = base / "tracking"
        ok, _ = _safe(
            "teams",
            assign_teams,
            tracks=converted_tracks,
            frames_dir=str(frames_dir),
            job_id=job_id,
            output_dir=str(team_dir),
        )
        _record("teams", ok)

    # 4. Pitch (writes pitch_map.json itself)
    from services.pitch_service import map_pitch
    ok, _ = _safe(
        "pitch",
        map_pitch,
        video_path=video_path, job_id=job_id,
        frame_stride=frame_stride_pitch,
    )
    _record("pitch", ok)

    # 5-11. Analytics services — call, persist returned dict at the path
    # match_report_service.py expects.
    analytics_steps: List[Tuple[str, str, Callable, Tuple[Any, ...]]] = []
    try:
        from services.heatmap_service import compute_heatmaps
        analytics_steps.append(("heatmap", "heatmap/heatmap.json", compute_heatmaps, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import heatmap_service failed: {e}")
    try:
        from services.pass_network_service import compute_pass_network
        analytics_steps.append(("pass_network", "pass_network/pass_network.json", compute_pass_network, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import pass_network_service failed: {e}")
    try:
        from services.pressing_service import compute_pressing
        analytics_steps.append(("pressing", "pressing/pressing.json", compute_pressing, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import pressing_service failed: {e}")
    try:
        from services.xg_service import compute_xg
        analytics_steps.append(("xg", "xg/xg_results.json", compute_xg, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import xg_service failed: {e}")
    try:
        from services.event_service import detect_events as _detect_events
        analytics_steps.append(("events", "events/event_timeline.json", _detect_events, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import event_service failed: {e}")
    try:
        from services.formation_service import compute_formations
        analytics_steps.append(("formation", "formation/formation.json", compute_formations, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import formation_service failed: {e}")
    try:
        from services.tactics_service import analyze_tactics
        analytics_steps.append(("tactics", "tactics/tactics_results.json", analyze_tactics, (job_id,)))
    except Exception as e:
        print(f"[FullPipeline] import tactics_service failed: {e}")

    for name, rel_path, fn, args in analytics_steps:
        ok, out = _safe(name, fn, *args)
        if ok and isinstance(out, (dict, list)):
            _write_json(base / rel_path, out)
        _record(name, ok)

    # 12. Match report (always last)
    from services.match_report_service import build_match_report
    ok, _ = _safe("match_report", build_match_report, job_id)
    _record("match_report", ok)

    manifest = {
        "jobId": job_id,
        "completedSteps": completed,
        "failedSteps": [{"step": s, "error": e} for s, e in failed],
        "outputDir": str(base),
    }
    _write_json(base / "pipeline_manifest.json", manifest)
    return manifest
