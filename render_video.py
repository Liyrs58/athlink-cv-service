import cv2
import json
import numpy as np
from pathlib import Path

TEAM_COLORS = {
    0: (0, 0, 255),      # Red in BGR (Villa)
    1: (255, 0, 0),      # Blue in BGR (PSG)
    -1: (128, 128, 128)  # Gray for unknown
}

def draw_annotations(frame, players):
    for p in players:
        bbox = p.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        tid = p.get('trackId', '?')
        team = p.get('teamId', -1)
        color = TEAM_COLORS.get(team, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"T{tid} T{team}"
        cv2.putText(frame, label, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def render_video(video_path, results_json, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    with open(results_json) as f:
        data = json.load(f)

    frames_data = {f['frameIndex']: f['players'] for f in data.get('frames', [])}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        frame = draw_annotations(frame, players)
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
    out_path = Path("/Users/rudra/Desktop/test video 1 deepseeek.mp4")
    render_video(video_path, json_path, out_path)
