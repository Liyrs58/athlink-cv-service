from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
import shutil, uuid, os, logging
import cv2
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
from services.job_queue_service import create_job, submit_job, get_job
from services.runpod_service import is_runpod_available, run_on_runpod
from services.observer_brain import ObserverBrain
from services.fatigue_clock_service import FatigueClock
from services.voronoi_service import compute_voronoi_control
from services.entropy_service import compute_team_entropy
from services.ball_tracking_service import BallTracker, PossessionDetector, PassDetector
from services.video_annotator import VideoAnnotator
from services.camera_compensator import CameraCompensator
from services.speed_estimator import SpeedEstimator
from services.brain_service import generate_brain_report

router = APIRouter()
logger = logging.getLogger(__name__)

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

def _convert_tracks_to_at_format(tracks, ball_data, total_frames):
    """
    Convert Athlink track format to Abdullah Tarek's format.

    Athlink format: list of track dicts with trajectory entries.
    Abdullah format: {
        "players": [{track_id: {"bbox": [...], "team": int}}, ...per frame],
        "ball": [{1: {"bbox": [...]}}, ...per frame]
    }
    """
    at_tracks = {
        "players": [{} for _ in range(total_frames)],
        "ball": [{} for _ in range(total_frames)],
    }

    # Map players
    for t in tracks:
        tid = t.get("trackId", 0)
        team_id = t.get("teamId", -1)
        for entry in t.get("trajectory", []):
            fi = entry.get("frameIndex", 0)
            bbox = entry.get("bbox", [])
            if fi < total_frames and len(bbox) >= 4:
                at_tracks["players"][fi][tid] = {
                    "bbox": bbox,
                    "team": team_id,
                    "team_color": {0: (0, 0, 255), 1: (0, 255, 0), -1: (128, 128, 128)}.get(team_id, (128, 128, 128)),
                }

    # Map ball positions
    if ball_data and not ball_data.get("error"):
        positions = ball_data.get("positions", {})
        for fi_str, pos in positions.items():
            fi = int(fi_str)
            if fi < total_frames:
                # Convert center point to bbox (ball is small, ~10px)
                cx, cy = pos["x"], pos["y"]
                r = 8
                at_tracks["ball"][fi][1] = {
                    "bbox": [cx - r, cy - r, cx + r, cy + r],
                }

    return at_tracks


def infer_position(positions: list, frame_h: int, team_id: int) -> str:
    """
    Infer position from average trajectory position.

    Splits pitch into thirds by x-coordinate (world coords):
    - Back third (0-35m): Defender / Goalkeeper
    - Middle third (35-70m): Midfielder
    - Front third (70-105m): Forward/Striker

    Goalkeeper: avg position in back 15% AND very low distance covered (<20m total)

    Returns: "Goalkeeper" / "Defender" / "Midfielder" / "Forward" / "Unknown"
    """
    try:
        if not positions or len(positions) == 0:
            return "Unknown"

        # Extract world x-coordinates from positions
        world_xs = []
        for pos in positions:
            if isinstance(pos, dict):
                # If position dict has world_x, use it
                if "world_x" in pos:
                    world_xs.append(float(pos["world_x"]))
                # Otherwise try to infer from pixel x and visible_fraction
                elif "pixel_x" in pos and "visible_fraction" in pos:
                    vis_frac = float(pos["visible_fraction"])
                    px = float(pos["pixel_x"])
                    world_x = (px / 1920.0) * (105.0 * vis_frac)
                    world_xs.append(world_x)

        if not world_xs:
            return "Unknown"

        avg_x = sum(world_xs) / len(world_xs)

        # Determine zone
        if avg_x < 35:
            zone = "back"
        elif avg_x < 70:
            zone = "middle"
        else:
            zone = "front"

        # Map zone to position
        if zone == "back":
            return "Defender"
        elif zone == "middle":
            return "Midfielder"
        else:
            return "Forward"
    except Exception:
        return "Unknown"

def _get_video_duration(path: str) -> float:
    """Return video duration in seconds, or 0 on error."""
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return frames / fps
    except Exception:
        return 0

def _run_on_runpod_wrapper(job_id: str, temp_path: str, skip_cleanup: bool = False, validate: bool = False):
    """Try RunPod GPU processing. Only fall back to local for videos under 60s."""
    duration = _get_video_duration(temp_path)
    try:
        result = run_on_runpod(temp_path, job_id)
        logger.info("RunPod completed job %s successfully", job_id)
        if not skip_cleanup and os.path.exists(temp_path):
            os.remove(temp_path)
        return result
    except Exception as e:
        error_msg = str(e)
        # Do not fall back if RunPod itself failed — that's a real error
        if "RunPod job failed" in error_msg or "RunPod error" in error_msg:
            raise
        # Do not fall back for long videos — CPU fallback is unusable
        if duration > 60:
            raise Exception(f"RunPod unavailable for job {job_id} ({duration:.0f}s video). Not falling back to CPU. Error: {error_msg}")
        logger.warning("RunPod network/timeout failure for short job %s (%.0fs) — falling back to local. Error: %s", job_id, duration, error_msg)
        return _run_analysis_pipeline(job_id, temp_path, skip_cleanup=skip_cleanup, validate=validate)


def _run_analysis_pipeline(job_id: str, temp_path: str, skip_cleanup: bool = False, validate: bool = False, pre_confirmed_labels: dict = None, progress_path: str = None):
    """Background task — runs the full analysis pipeline."""
    try:
        # Pitch calibration
        calibration = get_frame_calibration(temp_path)

        r = run_tracking(job_id=job_id, video_path=temp_path, frame_stride=2, progress_path=progress_path)
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

        # ── Ball tracking ─────────────────────────────────────────
        ball_data = {"error": "Ball tracking unavailable"}
        possession_data = {}
        pass_data = {}
        pass_detector = None
        try:
            ball_tracker = BallTracker()
            ball_tracker.load_model()
            if ball_tracker.model is not None:
                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ball_tracker.detect(frame, total_frames)
                    total_frames += 1
                cap.release()

                ball_positions = ball_tracker.get_positions()
                rate = ball_tracker.tracking_rate(total_frames)
                ball_data = {
                    "positions": {str(k): v for k, v in ball_positions.items()},
                    "tracked_frames": len(ball_positions),
                    "total_frames": total_frames,
                    "tracking_rate": round(rate, 1),
                }
                logger.info("Ball tracked in %d/%d frames (%.1f%%)", len(ball_positions), total_frames, rate)

                # Possession + pass detection
                ppm = 1.0
                if calibration and calibration.get("pixels_per_metre"):
                    ppm = max(calibration["pixels_per_metre"], 0.1)

                possession_det = PossessionDetector()
                pass_detector = PassDetector()

                for fi in range(total_frames):
                    bp = ball_tracker.get_position_at(fi)
                    # Build player list for this frame
                    frame_players = []
                    for t in tracks:
                        if t.get("firstSeen", 0) <= fi <= t.get("lastSeen", 0):
                            traj = t.get("trajectory", [])
                            closest = min(traj, key=lambda e: abs(e["frameIndex"] - fi), default=None) if traj else None
                            if closest and abs(closest["frameIndex"] - fi) <= 4:
                                bbox = closest.get("bbox", [])
                                if len(bbox) >= 4:
                                    frame_players.append({
                                        "track_id": t.get("trackId"),
                                        "cx": (bbox[0] + bbox[2]) / 2.0,
                                        "cy": (bbox[1] + bbox[3]) / 2.0,
                                        "team_id": t.get("teamId", -1),
                                    })

                    poss = possession_det.update(bp, frame_players, fi, ppm)
                    pass_detector.update(poss, bp, fi, fps, ppm)

                poss_pct = possession_det.get_team_possession_pct()
                possession_data = {
                    "team_0_pct": poss_pct.get(0, 0.0),
                    "team_1_pct": poss_pct.get(1, 0.0),
                    "events": possession_det.get_possession_events(),
                }
                pass_data = {
                    "total": len(pass_detector.get_passes()),
                    "per_player": pass_detector.get_passes_per_player(),
                    "events": pass_detector.get_passes(),
                }
        except Exception as e:
            logger.warning("Ball tracking failed: %s", e)
            ball_data = {"error": f"Ball tracking failed: {e}"}

        # ── Video annotation ────────────────────────────────────
        annotated_video_path = None
        annotation_possession = None
        try:
            cap2 = cv2.VideoCapture(temp_path)
            ann_fps = cap2.get(cv2.CAP_PROP_FPS) or 25.0
            ann_frames = []
            while True:
                ret2, f2 = cap2.read()
                if not ret2:
                    break
                ann_frames.append(f2)
            cap2.release()

            if ann_frames and len(ann_frames) > 5:
                ann_total = len(ann_frames)
                at_tracks = _convert_tracks_to_at_format(tracks, ball_data, ann_total)

                # Camera compensation
                compensator = CameraCompensator(ann_frames[0])
                camera_movement = compensator.get_camera_movement(ann_frames)
                at_tracks = compensator.adjust_positions(at_tracks, camera_movement)

                # Speed estimation with camera correction
                speed_est = SpeedEstimator()
                at_tracks = speed_est.calculate(at_tracks, ann_fps)

                # Ball possession
                annotator = VideoAnnotator()
                at_tracks, team_ball_control = annotator.assign_ball_possession(at_tracks)
                annotation_possession = annotator.get_team_ball_control_pct(team_ball_control)

                # Generate annotated video
                annotated_video_path = f'/tmp/{job_id}_annotated.mp4'
                annotator.annotate_video(
                    temp_path,
                    at_tracks,
                    annotated_video_path,
                    camera_movement,
                    team_ball_control,
                )

                logger.info("Video annotation complete: %s", annotated_video_path)
        except Exception as e:
            logger.error("Video annotation failed: %s", e)
            annotated_video_path = None

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
            ball_data=ball_data,
            possession_data=possession_data,
            pass_data=pass_data,
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

            # Infer position from trajectory
            traj = trk.get("trajectory", [])
            positions = []
            vis_frac = calibration.get("visible_fraction", 0.55) if calibration else 0.55
            for entry in traj:
                bbox = entry.get("bbox", [])
                if len(bbox) >= 4:
                    px = (bbox[0] + bbox[2]) / 2.0
                    positions.append({"pixel_x": px, "visible_fraction": vis_frac})
            position = infer_position(positions, frame_metadata[0].get("frameHeight", 1080) if frame_metadata else 1080, team_id)

            players_physical.append({
                "track_id": v["track_id"],
                "display_label": display_label,
                "position": position,
                "team": team_name,
                "confidence": conf_level,
                "speed_confidence": effective_conf,
                "sprints": sprints,
                "distance_metres": dist,
                "distance_range": f"{dist_lo:.0f}-{dist_hi:.0f}m",
                "max_speed_kmh": max_spd_kmh,
                "max_speed_display": spd_metric["display_value"],
                "unreliable_speed_frames": v.get("unreliable_speed_frames", 0),
                "passes": pass_detector.get_passes_per_player().get(v["track_id"], 0) if pass_detector else 0,
            })

        # Build match_feed from situation events and sprint data
        match_feed = []

        # Add situation events
        for event in events:
            start_time = event.get("start_time", 0)
            situation = event.get("situation", "UNKNOWN")
            time_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            description = "Open play resumes" if situation == "OPEN_PLAY" else "Play stopped"
            match_feed.append({
                "time": time_str,
                "type": "PHASE_CHANGE",
                "situation": situation,
                "description": f"{description} at {start_time:.1f}s"
            })

        # Add sprint events from velocity_summary
        for player in players_physical:
            if player.get("sprints", 0) > 0:
                max_speed = player.get("max_speed_kmh", 0)
                display_label = player.get("display_label") or f"{player['team'].capitalize()} P{player['track_id']}"
                position = player.get("position", "Unknown")
                team = player.get("team", "unknown")
                # Approximate timestamp from distance ratio (simplified)
                match_feed.append({
                    "time": "N/A",
                    "type": "SPRINT",
                    "player_label": display_label,
                    "position": position,
                    "team": team,
                    "speed": f"{max_speed} km/h",
                    "description": f"{display_label} ({position}) sprint — {max_speed} km/h"
                })

        # Sort by time (phase changes only have timestamps)
        match_feed_with_time = [f for f in match_feed if f["time"] != "N/A"]
        match_feed_no_time = [f for f in match_feed if f["time"] == "N/A"]
        match_feed_with_time.sort(key=lambda x: (int(x["time"].split(":")[0]), int(x["time"].split(":")[1])))
        match_feed = match_feed_with_time + match_feed_no_time

        # Build stats_table
        stats_rows = []
        for player in players_physical:
            if player["confidence"] not in ("high", "medium"):
                continue
            if player["distance_metres"] <= 5:
                continue
            stats_rows.append({
                "display_label": player.get("display_label") or f"{player['team'].capitalize()} P{player['track_id']}",
                "position": player.get("position", "Unknown"),
                "team": player["team"],
                "distance_range": player["distance_range"],
                "top_speed": f"{player['max_speed_kmh']} km/h ±10%",
                "sprints": player.get("sprints", 0),
                "passes": player.get("passes", 0),
                "confidence": player["confidence"]
            })
        stats_rows.sort(key=lambda x: x["distance_range"].split("-")[0], reverse=True)

        stats_table = {
            "headers": ["Player", "Position", "Team", "Distance", "Top Speed", "Sprints", "Passes", "Confidence"],
            "rows": stats_rows
        }

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
            "ball": ball_data,
            "possession": possession_data,
            "passes": pass_data,
            "match_feed": match_feed,
            "stats_table": stats_table,
            "annotated_video_path": annotated_video_path,
            "annotation_possession": annotation_possession,
            "analysis": analysis_text,
            "validation": validation_result,
        }
        result = numpy_safe(base_result)

        # Three-layer AI brain report replaces generic interpretation
        try:
            brain_report = generate_brain_report(job_id, temp_path, result)
            result["analysis"] = brain_report
        except Exception as e:
            logger.error("Brain report failed for %s: %s — keeping interpret_events output", job_id, e)

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

    # Try RunPod GPU first, fall back to local CPU
    if is_runpod_available():
        logger.info("RunPod available — routing job %s to GPU", job_id)
        submit_job(job_id, _run_on_runpod_wrapper, job_id, temp_path, validate=validate)
    else:
        logger.info("RunPod not available — processing job %s locally", job_id)
        submit_job(job_id, _run_analysis_pipeline, job_id, temp_path, validate=validate)

    # Return immediately with poll URL
    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "poll_url": f"/api/v1/jobs/status/{job_id}"
    })


@router.get("/jobs/{job_id}/video")
async def get_annotated_video(job_id: str):
    """Return the annotated video — from Supabase URL (RunPod) or local file (CPU fallback)."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, RedirectResponse

    # First check if job result has a Supabase URL (RunPod path)
    job = get_job(job_id)
    if job:
        result = job.get("result", {})
        if isinstance(result, dict):
            video_url = result.get("annotated_video_url")
            if video_url:
                return RedirectResponse(url=video_url, status_code=302)

    # Fall back to local file (CPU processing path)
    video_path = f'/tmp/{job_id}_annotated.mp4'
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type='video/mp4',
            filename=f'athlink_analysis_{job_id}.mp4',
        )

    raise HTTPException(status_code=404, detail="Annotated video not found. Processing may still be in progress.")
