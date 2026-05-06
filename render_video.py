"""
Render annotated video.

Box colour communicates IDENTITY confidence, not team:
  - GREEN   : locked or hungarian-stable identity (identity_valid=True, source != revived)
  - YELLOW  : revived from snapshot, not yet promoted to lock
  - GREY    : unassigned / uncertain — no internal tracker id is shown

Label format:
  P7 locked     P7 revived     P7 hungarian     ?
"""

import cv2
import json
from pathlib import Path

GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
GREY = (128, 128, 128)
RED = (0, 0, 255)


def _color_for(p):
    if not p.get("identity_valid", False):
        return GREY
    src = p.get("assignment_source", "")
    if src == "revived":
        return YELLOW
    return GREEN


def _label_for(p):
    display_id = p.get("displayId")
    identity_valid = p.get("identity_valid", False)
    if isinstance(display_id, str) and display_id.startswith("U T"):
        display_id = None
    if not identity_valid:
        if isinstance(display_id, int) or (isinstance(display_id, str) and display_id.isdigit()):
            display_id = None
    if display_id:
        src = p.get("assignment_source", "")
        if identity_valid and src:
            return f"{display_id} {src}"
        return str(display_id)
    if identity_valid:
        pid = p.get("playerId") or f"P{p.get('trackId', '?')}"
        src = p.get("assignment_source", "?")
        return f"{pid} {src}"
    if p.get("assignment_pending", False):
        return "?"
    return None


def draw_annotations(frame, players, frame_idx, summary_overlay=None):
    for p in players:
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        color = _color_for(p)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = _label_for(p)
        if label:
            cv2.putText(
                frame, label, (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

    cv2.putText(
        frame, f"F{frame_idx}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    if summary_overlay:
        cv2.putText(
            frame, summary_overlay, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
    return frame


def render_video(video_path, results_json, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    with open(results_json) as f:
        data = json.load(f)

    frames_data = {f["frameIndex"]: f["players"] for f in data.get("frames", [])}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        players = frames_data.get(frame_idx, [])
        n_locked = sum(1 for p in players if p.get("assignment_source") == "locked")
        n_revived = sum(1 for p in players if p.get("assignment_source") == "revived")
        n_hung = sum(1 for p in players if p.get("assignment_source") == "hungarian")
        n_unassigned = sum(1 for p in players if not p.get("identity_valid", False))
        overlay = (
            f"locked={n_locked} revived={n_revived} "
            f"hungarian={n_hung} uncertain={n_unassigned}"
        )
        frame = draw_annotations(frame, players, frame_idx, overlay)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    import sys
    job_id = sys.argv[1] if len(sys.argv) > 1 else "stride5_test"
    video_path = Path("/Users/rudra/Downloads/1b16c594_villa_psg_40s_new.mp4")
    json_path = Path(f"temp/{job_id}/tracking/track_results.json")
    out_path = Path(f"temp/{job_id}/annotated.mp4")
    render_video(video_path, json_path, out_path)
