"""
Render annotated video — UEFA-style overlay.

Box colour communicates TEAM, not identity confidence.
Stroke style communicates identity confidence:
  - SOLID  : locked (high confidence)
  - DASHED : revived / hungarian (medium)
  - DOTTED : unassigned / pending (low)

Officials are drawn as thin orange boxes with no label so they don't clutter the identity scene.
"""

import cv2
import json
from pathlib import Path  # noqa: F401

# Team palette (BGR) — picked to read well on green grass
TEAM_HOME = (255, 99, 29)   # blue (matches mood board)
TEAM_AWAY = (37, 197, 34)   # green (matches mood board)
TEAM_FALLBACK = (200, 200, 200)
REF_COLOR = (0, 165, 255)   # thin orange


def _team_color(p, team_map):
    """Resolve a per-track team colour from team_results.json mapping."""
    if p.get("is_official", False):
        return REF_COLOR
    tid = p.get("rawTrackId") or p.get("trackId")
    team = team_map.get(int(tid)) if tid is not None else None
    if team in (0, "0", "home"):
        return TEAM_HOME
    if team in (1, "1", "away"):
        return TEAM_AWAY
    return TEAM_FALLBACK


def _stroke_for(p):
    """Return ("solid"|"dashed"|"dotted", thickness) based on identity confidence."""
    if p.get("is_official", False):
        return "solid", 1
    src = p.get("assignment_source", "")
    if src == "locked":
        return "solid", 2
    if src in ("revived", "hungarian"):
        return "dashed", 2
    return "dotted", 1


def _draw_rect(frame, x1, y1, x2, y2, color, style, thickness):
    if style == "solid":
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return
    # dashed / dotted: draw segments along each edge
    seg_len = 6 if style == "dashed" else 2
    gap = 4 if style == "dashed" else 4

    def _line(p, q):
        # Bresenham-style stepping along p->q with seg_len/gap pattern
        x_a, y_a = p
        x_b, y_b = q
        dx = x_b - x_a
        dy = y_b - y_a
        length = max(abs(dx), abs(dy))
        if length == 0:
            return
        sx = dx / length
        sy = dy / length
        i = 0
        while i < length:
            j = min(i + seg_len, length)
            cv2.line(
                frame,
                (int(x_a + sx * i), int(y_a + sy * i)),
                (int(x_a + sx * j), int(y_a + sy * j)),
                color, thickness,
            )
            i = j + gap

    _line((x1, y1), (x2, y1))
    _line((x2, y1), (x2, y2))
    _line((x2, y2), (x1, y2))
    _line((x1, y2), (x1, y1))


def _label_for(p):
    if p.get("is_official", False):
        return None  # silent — refs get a thin box only
    if p.get("assignment_pending", False):
        return "?"
    if not p.get("identity_valid", False):
        return None

    pid = p.get("playerId") or p.get("displayId")
    if not pid:
        tid = p.get("trackId", "?")
        pid = f"P{tid}"

    source = p.get("assignment_source")
    if source:
        return f"{pid} {source}"
    return str(pid)


def draw_annotations(frame, entries, frame_idx, team_map, summary_overlay=None):
    for p in entries:
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        color = _team_color(p, team_map)
        style, thickness = _stroke_for(p)
        _draw_rect(frame, x1, y1, x2, y2, color, style, thickness)
        label = _label_for(p)
        if label:
            # White pill with team-color text, drawn just below the bbox top
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            pad = 3
            bx1, by1 = x1, max(y1 - th - 2 * pad, 0)
            bx2, by2 = x1 + tw + 2 * pad, by1 + th + 2 * pad
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(
                frame, label, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
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


def _load_team_map(results_json):
    """Find team_results.json next to track_results.json and build {trackId: team}."""
    p = Path(results_json).parent.parent / "tracking" / "team_results.json"
    if not p.exists():
        # Sometimes results_json is already in tracking/, try sibling
        p = Path(results_json).parent / "team_results.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        out = {}
        for entry in data.get("tracks", []) or data.get("teams", []) or []:
            tid = entry.get("trackId") or entry.get("track_id")
            team = entry.get("teamId") if "teamId" in entry else entry.get("team_id")
            if tid is not None:
                out[int(tid)] = team
        return out
    except Exception:
        return {}


def render_video(video_path, results_json, output_path,
                 frame_dir=None, sample_frames=None, verbose=False, **_kwargs):
    """
    Render annotated video.
    Optional frame_dir + sample_frames extracts JPGs at those frame indices.
    Extra kwargs are accepted for forward-compat with notebook callers.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    with open(results_json) as f:
        data = json.load(f)

    team_map = _load_team_map(results_json)

    # Combine players + officials per frame; officials get is_official tag preserved
    frames_data = {}
    for frec in data.get("frames", []):
        idx = frec["frameIndex"]
        entries = list(frec.get("players", []))
        for o in frec.get("officials", []):
            entries.append({**o, "is_official": True})
        frames_data[idx] = entries

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    sample_set = set(sample_frames or [])
    if frame_dir and sample_set:
        Path(frame_dir).mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        entries = frames_data.get(frame_idx, [])
        n_locked = sum(1 for p in entries if p.get("assignment_source") == "locked")
        n_revived = sum(1 for p in entries if p.get("assignment_source") == "revived")
        n_hung = sum(1 for p in entries if p.get("assignment_source") == "hungarian")
        n_unassigned = sum(1 for p in entries if not p.get("identity_valid", False) and not p.get("is_official", False))
        n_refs = sum(1 for p in entries if p.get("is_official", False))
        overlay = (
            f"locked={n_locked} revived={n_revived} "
            f"hungarian={n_hung} uncertain={n_unassigned} refs={n_refs}"
        )
        annotated = draw_annotations(frame, entries, frame_idx, team_map, overlay)
        out.write(annotated)
        if frame_dir and frame_idx in sample_set:
            cv2.imwrite(str(Path(frame_dir) / f"frame_{frame_idx:05d}.jpg"), annotated)
            extracted += 1
        frame_idx += 1

    cap.release()
    out.release()
    if verbose:
        print(f"[render_video] {frame_idx} frames written to {output_path}")
        if extracted:
            print(f"[render_video] {extracted} sample frames saved to {frame_dir}")
    print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    import sys
    job_id = sys.argv[1] if len(sys.argv) > 1 else "stride5_test"
    video_path = Path("/Users/rudra/Downloads/1b16c594_villa_psg_40s_new.mp4")
    json_path = Path(f"temp/{job_id}/tracking/track_results.json")
    out_path = Path(f"temp/{job_id}/annotated.mp4")
    render_video(video_path, json_path, out_path)
