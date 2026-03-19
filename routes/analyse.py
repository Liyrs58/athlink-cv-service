from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
import shutil, uuid, os
import numpy as np
from services.tracking_service import run_tracking
from services.team_separation_service import cluster_teams
from services.reid_service import merge_fragmented_tracks
from services.homography_service import get_frame_calibration
from services.game_brain import detect_situation, extract_situation_events
from services.interpretation_service import interpret_events
from services.velocity_service import compute_all_velocities, get_team_velocity_summary
from services.shape_service import compute_shape_summary
from services.memory_service import store_match, get_historical_context, get_match_count, get_player_history
from services.trajectory_service import DevelopmentTrajectory
from services.physics_corrector import PhysicsCorrector
from services.confidence_service import (
    score_track_confidence,
    score_physical_metric,
    build_data_confidence_summary,
)
from services.job_queue_service import create_job, submit_job
from services.observer_brain import ObserverBrain
from services.fatigue_clock_service import FatigueClock
from services.voronoi_service import compute_voronoi_control
from services.entropy_service import compute_team_entropy

router = APIRouter()

def numpy_safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [numpy_safe(i) for i in obj]
    return obj

# Keep alias for any other code that may reference make_json_safe
make_json_safe = numpy_safe

def _run_analysis_pipeline(job_id: str, temp_path: str, skip_cleanup: bool = False, validate: bool = False, pre_confirmed_labels: dict = None):
    """Background task — runs the full analysis pipeline."""
    try:
        # Pitch calibration
        calibration = get_frame_calibration(temp_path)

        r = run_tracking(job_id=job_id, video_path=temp_path, frame_stride=2)
        tracks = r.get("tracks", [])
        frame_metadata = r.get("frame_metadata", [])

        # ReID merge
        tracks_before_merge = len(tracks)
        tracks = merge_fragmented_tracks(tracks, temp_path)
        tracks_after_merge = len(tracks)

        # Physics corrections — apply hard constraints
        corrector = PhysicsCorrector()
        correction_report = corrector.apply_all_constraints(
            tracks=tracks,
            frame_metadata=frame_metadata,
            calibration=calibration,
        )
        tracks = correction_report["corrected_tracks"]
        calibration = correction_report["calibration"]
        corrections_applied = correction_report["corrections_applied"]

        # Apply coach-confirmed labels from stream session
        if pre_confirmed_labels:
            for track in tracks:
                tid = track.get("trackId")
                if tid in pre_confirmed_labels:
                    track["coach_label"] = pre_confirmed_labels[tid]
                    track["coach_confirmed"] = True

        # Team separation
        team_sep = cluster_teams(tracks, temp_path)

        frame_results = []
        for meta in frame_metadata:
            frame_idx = meta.get("frameIndex", 0)
            active_with_bbox = []
            for t in tracks:
                if t.get("firstSeen", 0) <= frame_idx <= t.get("lastSeen", 0):
                    traj = t.get("trajectory", [])
                    closest = min(traj, key=lambda e: abs(e["frameIndex"] - frame_idx), default=None) if traj else None
                    if closest and abs(closest["frameIndex"] - frame_idx) <= 4:
                        active_with_bbox.append({
                            "trackId": t.get("trackId"),
                            "bbox": closest["bbox"],
                        })
            result = detect_situation(tracks=active_with_bbox, ball=None, frame_idx=frame_idx)
            frame_results.append({"frameIndex": frame_idx, "situation": result["situation"]})

        events = extract_situation_events(frame_results, fps=25.0, frame_stride=2)
        velocities = compute_all_velocities(tracks, calibration=calibration, video_path=temp_path)
        vel_summary = get_team_velocity_summary(velocities) or {}
        shape_summary = compute_shape_summary(tracks, frame_metadata, calibration=calibration) or {}

        voronoi = compute_voronoi_control(tracks, frame_metadata, calibration)
        entropy = compute_team_entropy(tracks, frame_metadata, calibration)

        # Part 1+4: Build confidence summary
        data_confidence = build_data_confidence_summary(tracks, vel_summary, shape_summary)

        # Fatigue Clock — Fourier-based fatigue analysis per player
        fatigue_clock = FatigueClock()
        fatigue_result = fatigue_clock.analyse_all_players(tracks, calibration=calibration)

        # Observer Brain — continuous belief update across all frames
        brain = ObserverBrain()
        brain_summary = brain.process_full_match(tracks, frame_metadata, calibration)

        memory = get_historical_context()
        analysis = interpret_events(
            events, tracks, job_id, vel_summary, shape_summary,
            velocities, memory, team_separation=team_sep,
            data_confidence=data_confidence,
            brain_summary=brain_summary,
            voronoi=voronoi,
            entropy=entropy,
            calibration=calibration,
        )
        analysis_text = analysis[0]["analysis"] if analysis else ""

        # Development Trajectory — player arc prediction across matches
        trajectory_engine = DevelopmentTrajectory()
        memory_data = {"player_history": get_player_history()}
        trajectory_summary = trajectory_engine.compute_team_trajectories(
            memory_data=memory_data,
            current_tracks=tracks,
            current_velocities=velocities,
            current_fatigue=fatigue_result,
        )

        store_match(
            job_id,
            {"total_tracks": len(tracks)},
            {"events": events},
            vel_summary,
            shape_summary,
            analysis_text,
            player_history=trajectory_summary.get("updated_player_history", {}),
        )

        # Part 4: Build per-player physical with confidence
        track_team_map = {}
        for t in tracks:
            tid = t.get("trackId")
            if tid is not None:
                track_team_map[tid] = t.get("teamId", -1)

        t0_name = team_sep.get("team_0_colour_name", "A")
        t1_name = team_sep.get("team_1_colour_name", "B")

        players_physical = []
        for v in velocities:
            trk = next((t for t in tracks if t.get("trackId") == v["track_id"]), None)
            if not trk:
                continue
            tc = score_track_confidence(trk)
            conf_level = tc["level"]

            max_spd_kmh = round(v["max_speed_ms"] * 3.6, 1)
            dist = v["distance_metres"]
            sprints = v["sprint_count"]

            # Distance range based on confidence
            spd_metric = score_physical_metric(max_spd_kmh, conf_level, "speed")
            dist_metric = score_physical_metric(dist, conf_level, "distance")

            pct = {"high": 0.12, "medium": 0.25, "low": 0.40}.get(conf_level, 0.25)
            dist_lo = round(dist * (1 - pct), 0)
            dist_hi = round(dist * (1 + pct), 0)

            team_id = track_team_map.get(v["track_id"], -1)
            team_name = t0_name if team_id == 0 else (t1_name if team_id == 1 else "unassigned")

            # Skip stationary/ghost tracks — no real movement detected
            if dist == 0 and sprints == 0 and max_spd_kmh < 5.0:
                continue

            # Optical flow cross-check: downgrade speed confidence if unreliable
            effective_conf = conf_level
            if v.get("speed_confidence_downgraded") and conf_level == "high":
                effective_conf = "medium"

            display_label = trk.get("coach_label") if trk.get("coach_confirmed") else None

            players_physical.append({
                "track_id": v["track_id"],
                "display_label": display_label,
                "team": team_name,
                "confidence": conf_level,
                "speed_confidence": effective_conf,
                "sprints": sprints,
                "distance_metres": dist,
                "distance_range": f"{dist_lo:.0f}-{dist_hi:.0f}m",
                "max_speed_kmh": max_spd_kmh,
                "max_speed_display": spd_metric["display_value"],
                "unreliable_speed_frames": v.get("unreliable_speed_frames", 0),
            })

        # Multi-pass validation (only if ?validate=true)
        validation_result = None
        if validate:
            from services.multi_pass_validator import run_multi_pass_validation
            validation_result = run_multi_pass_validation(
                video_path=temp_path,
                job_id=job_id,
                players_physical=players_physical,
            )

        # Total-level confidence for physical summary
        total_sprints = vel_summary.get("total_sprints", 0)
        max_speed_kmh = vel_summary.get("max_speed_kmh", 0)
        high_count = data_confidence.get("high_confidence_players", 0)
        total_players = high_count + data_confidence.get("medium_confidence_players", 0) + data_confidence.get("low_confidence_players", 0)
        phys_conf = "high" if total_players > 0 and high_count / total_players >= 0.6 else "medium"

        base_result = {
            "job_id": job_id,
            "matches_in_memory": get_match_count(),
            "tracking": {
                "total_tracks": len(tracks),
                "confirmed_tracks": sum(1 for t in tracks if isinstance(t.get("confirmed_detections", 0), (int, float)) and t.get("confirmed_detections", 0) >= 5),
                "frames_processed": r.get("framesProcessed", 0),
                "team_0_count": team_sep.get("team_0_players", 0),
                "team_1_count": team_sep.get("team_1_players", 0),
            },
            "team_separation": team_sep,
            "data_confidence": data_confidence,
            "situations": {
                "events": events,
                "counts": {s: sum(1 for f in frame_results if f["situation"] == s) for s in set(f["situation"] for f in frame_results)}
            },
            "physical": {
                "total_sprints": total_sprints,
                "total_sprints_confidence": phys_conf,
                "max_speed_kmh": max_speed_kmh,
                "max_speed_confidence": phys_conf,
                "max_speed_margin": score_physical_metric(max_speed_kmh, phys_conf, "speed")["margin_of_error"],
                "total_distance_metres": vel_summary.get("total_distance_metres", 0),
                "players_analysed": sum(
                    1 for p in players_physical
                    if p["distance_metres"] > 5 and p["confidence"] in ("high", "medium")
                ),
                "top_sprinter_id": vel_summary.get("top_sprinter_id"),
                "top_runner_id": vel_summary.get("top_runner_id"),
                "top_runner_distance": vel_summary.get("top_runner_distance"),
                "players": players_physical,
            },
            "shape": shape_summary,
            "calibration": {
                "method": calibration.get("method"),
                "visible_fraction": calibration.get("visible_fraction"),
                "pixels_per_metre": calibration.get("pixels_per_metre"),
            },
            "reid": {
                "tracks_before_merge": tracks_before_merge,
                "tracks_after_merge": tracks_after_merge,
                "fragments_merged": tracks_before_merge - tracks_after_merge,
            },
            "corrections_applied": corrections_applied,
            "fatigue_clock": fatigue_result,
            "voronoi": {
                "status": voronoi.get("status"),
                "team_0_control_pct": voronoi.get("team_0_control_pct"),
                "team_1_control_pct": voronoi.get("team_1_control_pct"),
                "dominant_team": voronoi.get("dominant_team"),
                "control_margin": voronoi.get("control_margin"),
                "frames_analysed": voronoi.get("frames_analysed"),
            },
            "entropy": {
                "status": entropy.get("status"),
                "team_0": entropy.get("team_0", {}),
                "team_1": entropy.get("team_1", {}),
                "shape_collapse_events": entropy.get("shape_collapse_events", []),
            },
            "brain": {
                "verdict": brain_summary.get("brain_verdict", ""),
                "tracking_health": brain_summary.get("tracking_health", {}),
                "match_phases": brain_summary.get("match_phases", []),
                "metrics_to_trust": brain_summary.get("metrics_to_trust", []),
                "metrics_to_question": brain_summary.get("metrics_to_question", []),
                "anomalies_summary": brain_summary.get("anomalies_summary", ""),
            },
            "trajectory": {
                "players_with_trajectory": trajectory_summary.get(
                    "players_with_trajectory", 0),
                "trajectories": trajectory_summary.get("trajectories", {}),
                "history_size": trajectory_summary.get("player_history_size", {}),
            },
            "analysis": analysis_text,
            "validation": validation_result,
        }
        result = numpy_safe(base_result)
        return result
    finally:
        if not skip_cleanup and os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/analyse")
async def analyse_video(
    video: UploadFile = File(...),
    validate: bool = Query(False, description="Run 3-pass validation for confirmed metrics"),
):
    """Submit video for async background analysis. Returns immediately with job_id."""
    job_id = str(uuid.uuid4())[:8]
    temp_path = f"/tmp/{job_id}_{video.filename}"

    # Write video to temp file
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create job in queue
    create_job(job_id)

    # Submit async task
    submit_job(job_id, _run_analysis_pipeline, job_id, temp_path, validate=validate)

    # Return immediately with poll URL
    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "poll_url": f"/api/v1/jobs/status/{job_id}"
    })
