#!/usr/bin/env python3.14
"""
Self-contained 3D replay renderer.

Usage:
    python3.14 scripts/render_3d_replay.py <jobId> [--preview]

What it does:
    1. Builds the export JSON from temp/<jobId>/ outputs
    2. Starts a CORS-enabled HTTP server serving /api/v1/export/<jobId>
    3. Runs `npx remotion render` (or `remotion preview` if --preview) pointing
       at the local server. Remotion's Chromium fetches the export, loads the
       scene, and renders to temp/<jobId>/replay3d/replay.mp4.

Requires Python 3.10+ (for PEP 604 types in export_service chain). Uses the
system Python 3.14 — NOT the Python 3.9 .venv.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.export_service import build_export  # noqa: E402


SERVER_PORT = 8765
BRIDGE_DIR = REPO_ROOT / "nextjs-bridge"
REMOTION_ENTRY = "remotion/index.ts"
COMPOSITION_ID = "PitchReplay"


def _cors_headers(h: BaseHTTPRequestHandler) -> None:
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "*")


def _make_handler(export_payload: bytes, job_id: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep stdout quiet
            pass

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            _cors_headers(self)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path == f"/api/v1/export/{job_id}":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(export_payload)))
                _cors_headers(self)
                self.end_headers()
                self.wfile.write(export_payload)
                return
            self.send_response(404)
            _cors_headers(self)
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')

    return Handler


def _start_server(export_payload: bytes, job_id: str) -> ThreadingHTTPServer:
    handler_cls = _make_handler(export_payload, job_id)
    server = ThreadingHTTPServer(("127.0.0.1", SERVER_PORT), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[serve] http://127.0.0.1:{SERVER_PORT}/api/v1/export/{job_id}  "
          f"({len(export_payload)/1024:.0f} KB)")
    return server


def _validate_pitch(job_id: str) -> None:
    p = REPO_ROOT / "temp" / job_id / "pitch" / "pitch_map.json"
    if not p.exists():
        raise SystemExit(f"[error] pitch_map.json missing for {job_id}. Run /pitch/map first.")
    data = json.loads(p.read_text())
    if not data.get("calibration_valid"):
        raise SystemExit(
            f"[error] Calibration is invalid for {job_id}. "
            "World coords would be unreliable — refusing to render."
        )


def _build_export_bytes(job_id: str) -> bytes:
    print(f"[build] export_service.build_export({job_id!r})…")
    payload = build_export(job_id)
    vm = payload.get("videoMeta", {})
    print(f"[build] videoMeta: fps={vm.get('fps')} frameCount={vm.get('frameCount')} "
          f"duration={vm.get('durationSeconds')}s")
    print(f"[build] frames={len(payload.get('frames', []))}  "
          f"ball_points={len(payload.get('ball', []))}")
    return json.dumps(payload).encode()


def _run_remotion_render(job_id: str, fps: float | None) -> Path:
    """Render PNG sequence via Remotion, then stitch to MP4 with static ffmpeg.
    Remotion's bundled ffmpeg requires macOS 15; we avoid it entirely."""
    out_dir = REPO_ROOT / "temp" / job_id / "replay3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = out_dir / "frames"
    if seq_dir.exists():
        import shutil as _sh
        _sh.rmtree(seq_dir)
    seq_dir.mkdir(parents=True)
    mp4 = out_dir / "replay.mp4"

    props = {
        "jobId": job_id,
        "exportUrl": f"http://127.0.0.1:{SERVER_PORT}/api/v1/export/{job_id}",
    }
    if fps is not None:
        props["videoFps"] = fps

    cmd = [
        "npx", "remotion", "render",
        REMOTION_ENTRY, COMPOSITION_ID,
        str(seq_dir),
        f"--props={json.dumps(props)}",
        f"--gl={os.environ.get('REPLAY3D_GL', 'angle')}",
        "--sequence", "--image-format=png",
        "--log=info",
    ]

    print(f"[render] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BRIDGE_DIR))
    if result.returncode != 0:
        raise SystemExit(f"[error] Remotion render exited {result.returncode}")

    # Stitch PNGs → MP4 via static ffmpeg
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    png_fps = fps if fps is not None else _fps_from_export(job_id)
    stitch = [
        ffmpeg, "-y",
        "-framerate", str(png_fps),
        "-i", str(seq_dir / "element-%d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast",
        str(mp4),
    ]
    print(f"[stitch] {' '.join(stitch)}")
    r2 = subprocess.run(stitch, capture_output=True, text=True)
    if r2.returncode != 0:
        # Try alternate filename pattern (Remotion uses 0-padded by default)
        alt = [
            ffmpeg, "-y",
            "-framerate", str(png_fps),
            "-pattern_type", "glob",
            "-i", str(seq_dir / "*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            str(mp4),
        ]
        print(f"[stitch-retry] {' '.join(alt)}")
        r2 = subprocess.run(alt, capture_output=True, text=True)
        if r2.returncode != 0:
            print(r2.stderr[-2000:])
            raise SystemExit(f"[error] ffmpeg stitch failed: exit {r2.returncode}")

    if not mp4.exists() or mp4.stat().st_size == 0:
        raise SystemExit("[error] produced no MP4")
    print(f"[done] {mp4}  ({mp4.stat().st_size/1e6:.1f} MB)")
    return mp4


def _fps_from_export(job_id: str) -> float:
    p = build_export(job_id).get("videoMeta", {}).get("fps") or 25.0
    return float(p)


def _run_remotion_preview(job_id: str) -> None:
    props = {
        "jobId": job_id,
        "exportUrl": f"http://127.0.0.1:{SERVER_PORT}/api/v1/export/{job_id}",
    }
    cmd = [
        "npx",
        "remotion",
        "preview",
        REMOTION_ENTRY,
        f"--props={json.dumps(props)}",
    ]
    print(f"[preview] {' '.join(cmd)}")
    # Foreground — Ctrl+C stops both preview and server.
    subprocess.run(cmd, cwd=str(BRIDGE_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--preview", action="store_true",
                    help="Launch Remotion preview server instead of rendering MP4.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override the composition fps (default: derive from videoMeta).")
    ap.add_argument("--skip-calibration-check", action="store_true",
                    help="Render even if calibration_valid=False (dev only).")
    args = ap.parse_args()

    if not args.skip_calibration_check:
        _validate_pitch(args.job_id)

    payload = _build_export_bytes(args.job_id)
    server = _start_server(payload, args.job_id)

    # Give server a moment to bind
    time.sleep(0.3)

    try:
        if args.preview:
            _run_remotion_preview(args.job_id)
        else:
            _run_remotion_render(args.job_id, args.fps)
    except KeyboardInterrupt:
        print("\n[main] interrupted — shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
