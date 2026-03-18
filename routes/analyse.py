from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, uuid, os
from services.tracking_service import run_tracking
from services.team_separation_service import cluster_teams
from services.reid_service import merge_fragmented_tracks
from services.homography_service import get_frame_calibration
from services.game_brain import detect_situation, extract_situation_events
from services.interpretation_service import interpret_events
from services.velocity_service import compute_all_velocities, get_team_velocity_summary
from services.shape_service import compute_shape_summary
from services.memory_service import store_match, get_historical_context, get_match_count

router = APIRouter()

@router.post("/analyse")
async def analyse_video(video: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    temp_path = f"/tmp/{job_id}_{video.filename}"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        # Pitch calibration — estimate visible fraction and pixels_per_metre
        calibration = get_frame_calibration(temp_path)

        r = run_tracking(job_id=job_id, video_path=temp_path, frame_stride=2)
        tracks = r.get("tracks", [])
        frame_metadata = r.get("frame_metadata", [])

        # ReID — merge fragmented tracks before team separation
        tracks_before_merge = len(tracks)
        tracks = merge_fragmented_tracks(tracks, temp_path)
        tracks_after_merge = len(tracks)

        # Team separation — assign teamId to each track using jersey colours
        team_sep = cluster_teams(tracks, temp_path)

        frame_results = []
        for meta in frame_metadata:
            frame_idx = meta.get("frameIndex", 0)
            # Build per-frame track snapshots with top-level bbox for detect_situation
            active_with_bbox = []
            for t in tracks:
                if t.get("firstSeen", 0) <= frame_idx <= t.get("lastSeen", 0):
                    traj = t.get("trajectory", [])
                    # Find the trajectory entry closest to this frame
                    closest = min(traj, key=lambda e: abs(e["frameIndex"] - frame_idx), default=None) if traj else None
                    if closest and abs(closest["frameIndex"] - frame_idx) <= 4:
                        active_with_bbox.append({
                            "trackId": t.get("trackId"),
                            "bbox": closest["bbox"],
                        })
            result = detect_situation(tracks=active_with_bbox, ball=None, frame_idx=frame_idx)
            frame_results.append({"frameIndex": frame_idx, "situation": result["situation"]})

        events = extract_situation_events(frame_results, fps=25.0, frame_stride=2)
        velocities = compute_all_velocities(tracks, calibration=calibration)
        vel_summary = get_team_velocity_summary(velocities) or {}
        shape_summary = compute_shape_summary(tracks, frame_metadata, calibration=calibration) or {}
        memory = get_historical_context()
        analysis = interpret_events(
            events, tracks, job_id, vel_summary, shape_summary,
            velocities, memory, team_separation=team_sep,
        )
        analysis_text = analysis[0]["analysis"] if analysis else ""
        store_match(job_id, {"total_tracks": len(tracks)}, {"events": events}, vel_summary, shape_summary, analysis_text)

        return JSONResponse({
            "job_id": job_id,
            "matches_in_memory": get_match_count(),
            "tracking": {
                "total_tracks": len(tracks),
                "confirmed_tracks": sum(1 for t in tracks if isinstance(t.get("confirmed_detections", 0), (int, float)) and t.get("confirmed_detections",0) >= 5),
                "frames_processed": r.get("framesProcessed", 0),
                "team_0_count": team_sep.get("team_0_players", 0),
                "team_1_count": team_sep.get("team_1_players", 0),
            },
            "team_separation": team_sep,
            "situations": {
                "events": events,
                "counts": {s: sum(1 for f in frame_results if f["situation"]==s) for s in set(f["situation"] for f in frame_results)}
            },
            "physical": vel_summary,
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
            "analysis": analysis_text,
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
