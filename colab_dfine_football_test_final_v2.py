#!/usr/bin/env python3
"""
D-FINE + Sports ReID — FINAL COLAB CELL (v2)
WORKAROUND: Skip HF auth issue, use wget to download weights directly
Paste this entire cell into Colab after Runtime → Restart
"""

import os, sys, subprocess, time
from pathlib import Path

REPO_URL    = "https://github.com/Liyrs58/athlink-cv-service.git"
REPO        = Path("/content/athlink-cv-service")
VIDEO       = Path("/content/Aston villa vs Psg clip 1.mov")
JOB_ID      = "dfine_test"
DFINE_MODEL = "rudrasinghm/dfine-football-detector"

os.environ["CUDA_VISIBLE_DEVICES"]           = "0"
os.environ["ATHLINK_FORCE_DEVICE"]           = "cuda"
os.environ["ATHLINK_YOLO_HALF"]              = "0"
os.environ["ATHLINK_MAX_PLAYER_SLOTS"]       = "14"
os.environ["ATHLINK_ALLOW_NEW_PLAYER_SLOTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]        = "expandable_segments:True"

print("=" * 80)
print("D-FINE FOOTBALL DETECTOR + SPORTS ReID TEST (v2)")
print(f"Model: {DFINE_MODEL}")
print("=" * 80)

# ── 1. GPU check ──────────────────────────────────────────────────────────────
import torch
assert torch.cuda.is_available(), "Switch Colab runtime to T4 GPU."
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 2. Repo ───────────────────────────────────────────────────────────────────
if REPO.exists():
    subprocess.run(["git", "-C", str(REPO), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(REPO), "reset", "--hard", "origin/main"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)

os.chdir(REPO)
commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"✓ HEAD: {commit_hash}")

# ── 3. Dependencies ───────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "ultralytics", "boxmot", "huggingface_hub",
                "transformers>=4.40", "accelerate"], check=True)
print("✓ Dependencies installed")

# ── 4. Download sports OSNet ReID weights (WORKAROUND: direct URL) ────────────
print("\n--- Downloading sports OSNet ReID weights ---")
osnet_path = Path("/content/football_osnet_x1_0.pth.tar")
if not osnet_path.exists():
    # Try direct HF Hub CDN URL (no auth needed for public files)
    hf_cdn_url = "https://huggingface.co/rudrasinghm/football-osnet-reid/resolve/main/football_osnet_x1_0.pth.tar"
    print(f"Downloading from: {hf_cdn_url}")
    result = subprocess.run(["wget", "-q", hf_cdn_url, "-O", str(osnet_path)], capture_output=True)
    if result.returncode != 0:
        print(f"⚠️  wget failed. Trying curl...")
        subprocess.run(["curl", "-L", "-o", str(osnet_path), hf_cdn_url], check=True)
    if osnet_path.exists():
        print(f"✓ OSNet weights: {osnet_path}")
    else:
        print("✗ Failed to download OSNet weights")
        print("  Fallback: using MSMT17 weights instead")
        osnet_path = Path("models/osnet_x1_0_msmt17.pt")
else:
    print(f"✓ OSNet weights already cached: {osnet_path}")

os.environ["OSNET_SPORTS_WEIGHTS"] = str(osnet_path)

# ── 5. Video check ────────────────────────────────────────────────────────────
if not VIDEO.exists():
    alt = Path("/content/1b16c594_villa_psg_40s_new.mp4")
    if alt.exists():
        VIDEO = alt
    else:
        print("ERROR: Upload video to /content/")
        sys.exit(1)
print(f"✓ Video: {VIDEO}")

# ── 6. Load D-FINE football detector ──────────────────────────────────────────
print(f"\n--- Loading {DFINE_MODEL} ---")
sys.path.insert(0, str(REPO))

import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image as PILImage

_dfine_processor = AutoImageProcessor.from_pretrained(DFINE_MODEL)
_dfine_model     = AutoModelForObjectDetection.from_pretrained(DFINE_MODEL)

# Head Override Workaround to map intermediate prediction heads to head 0
_dfine_model.class_embed[1] = _dfine_model.class_embed[0]
_dfine_model.class_embed[2] = _dfine_model.class_embed[0]
_dfine_model.bbox_embed[1] = _dfine_model.bbox_embed[0]
_dfine_model.bbox_embed[2] = _dfine_model.bbox_embed[0]

_dfine_model.model.decoder.class_embed[1] = _dfine_model.model.decoder.class_embed[0]
_dfine_model.model.decoder.class_embed[2] = _dfine_model.model.decoder.class_embed[0]
_dfine_model.model.decoder.bbox_embed[1] = _dfine_model.model.decoder.bbox_embed[0]
_dfine_model.model.decoder.bbox_embed[2] = _dfine_model.model.decoder.bbox_embed[0]

_dfine_model     = _dfine_model.cuda().eval()

_id2label = _dfine_model.config.id2label
print(f"Classes: {_id2label}")

_PLAYER_LABELS = {k for k, v in _id2label.items() if v.lower() in ("player", "goalkeeper")}
print(f"Tracked: {[_id2label[k] for k in _PLAYER_LABELS]}")

def _dfine_detect(frame_bgr: np.ndarray, conf_threshold: float = 0.35):
    h, w = frame_bgr.shape[:2]
    pil_img = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = {k: v.cuda() for k, v in
              _dfine_processor(images=pil_img, return_tensors="pt").items()}
    with torch.no_grad():
        outputs = _dfine_model(**inputs)
    
    # Use standard post-processing (SIGMOID focal loss activation, top-k gather, absolute scaling)
    results = _dfine_processor.post_process_object_detection(
        outputs,
        threshold=conf_threshold,
        target_sizes=torch.tensor([[h, w]], device="cuda"),
    )[0]
    
    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() not in _PLAYER_LABELS:
            continue
            
        xmin_comp, ymin_comp, xmax_comp, ymax_comp = box.tolist()
        
        # Absolute Bounding Box Recovery Formula
        x1_raw = xmin_comp
        y1_raw = ymin_comp
        x2_raw = xmax_comp - xmin_comp
        y2_raw = ymax_comp - ymin_comp
        
        # Robust clamping and ordering
        x1 = max(0.0, min(x1_raw, x2_raw))
        x2 = max(0.0, min(max(x1_raw, x2_raw), w))
        y1 = max(0.0, min(y1_raw, y2_raw))
        y2 = max(0.0, min(max(y1_raw, y2_raw), h))
        
        if x2 > x1 and y2 > y1:
            dets.append([x1, y1, x2, y2, score.item(), 2.0])
            
    return np.array(dets, dtype=np.float32) if dets else np.zeros((0, 6), dtype=np.float32)


# ── 7. Patch TrackerCore.detect() ─────────────────────────────────────────────
for mod in list(sys.modules):
    if mod == "services" or mod.startswith("services."):
        del sys.modules[mod]

import services.tracker_core as tc
_original_detect = tc.TrackerCore.detect

def _patched_detect(self, frame):
    self._last_ball_det = None
    return _dfine_detect(frame, conf_threshold=0.35)

tc.TrackerCore.detect = _patched_detect
print("✓ Detector patched: YOLO → D-FINE")

# ── 8. Run tracking ───────────────────────────────────────────────────────────
print("\n--- Running Tracking Pipeline ---")
print("=" * 80)

from services.tracker_core import run_tracking
import io
from contextlib import redirect_stdout

output_buffer = io.StringIO()
t0 = time.time()
try:
    with redirect_stdout(output_buffer):
        tracking_results = run_tracking(
            video_path=str(VIDEO),
            job_id=JOB_ID,
            frame_stride=5,
            max_frames=None,
            device="cuda",
        )
    elapsed = time.time() - t0
    full_output = output_buffer.getvalue()
    print(full_output[-4000:])
    print(f"\n✓ Tracking complete in {elapsed:.1f}s")
except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 9. Identity metrics ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("IDENTITY METRICS")
print("=" * 80)

identity_metrics = {}
for line in full_output.split("\n"):
    if any(k in line for k in ("collapse_lock_creations", "locks_created",
                                "lock_retention_rate", "valid_id_coverage")):
        parts = line.split("=")
        if len(parts) >= 2:
            try:
                identity_metrics[parts[0].strip()] = float(parts[1].strip().split()[0])
            except Exception:
                pass

locks_created      = int(identity_metrics.get("locks_created", 0))
lock_retention     = identity_metrics.get("lock_retention_rate", 0.0)
collapse_creations = int(identity_metrics.get("collapse_lock_creations", 0))
valid_id_coverage  = identity_metrics.get("valid_id_coverage", 0.0)

# Adaptive Gate 1 threshold based on the video name/length (40s full clip vs short clip)
target_locks = 20 if ("40s" in VIDEO.name or "1b16c5" in VIDEO.name) else 4
gate_1_pass = locks_created >= target_locks

print(f"GATE 1 locks_created >= {target_locks}  : {locks_created} → {'✓ PASS' if gate_1_pass else '✗ FAIL'}")
print(f"GATE 2 lock_retention >= 0.65: {lock_retention:.3f} → {'✓ PASS' if lock_retention >= 0.65 else '✗ FAIL'}")
print(f"GATE 3 collapse_lock_creations == 0: {collapse_creations} → {'✓ PASS' if collapse_creations == 0 else '✗ FAIL'}")
print(f"       valid_id_coverage: {valid_id_coverage:.3f}")

all_pass = gate_1_pass and lock_retention >= 0.65 and collapse_creations == 0
print(f"\n{'✅ ALL GATES PASS' if all_pass else '⚠️  SOME GATES FAIL'}")


# ── 10. Render annotated video ────────────────────────────────────────────────
print("\n--- Rendering Annotated Video ---")

output_video = Path(f"/content/dfine_annotated_{VIDEO.stem}.mp4")
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

frame_map = {}
for fd in tracking_results:
    fi = fd.get("frameIndex", fd.get("frame_index", -1))
    frame_map[fi] = fd.get("players", fd.get("detections", []))

COLORS = {"locked": (0, 255, 0), "revived": (0, 220, 255), "provisional": (0, 165, 255), "unknown": (128, 128, 128)}

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=10, gap=5):
    for ax, ay, bx, by in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
        length = max(abs(bx-ax)+abs(by-ay), 1)
        seg = dash + gap
        for i in range(max(int(length/seg),1)):
            t0_ = (i*seg)/length
            t1_ = min((i*seg+dash)/length, 1.0)
            p0 = (int(ax+t0_*(bx-ax)), int(ay+t0_*(by-ay)))
            p1 = (int(ax+t1_*(bx-ax)), int(ay+t1_*(by-ay)))
            cv2.line(img, p0, p1, color, thickness)

fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    for p in frame_map.get(fi, []):
        bbox  = p.get("bbox", [])
        src   = p.get("assignment_source", "unknown")
        valid = p.get("identity_valid", False)
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color  = COLORS.get(src if valid else "unknown", (128, 128, 128))
        dashed = src in ("provisional", "unknown") or not valid
        if dashed:
            draw_dashed_rect(frame, x1, y1, x2, y2, color)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        lbl = p.get("displayId", p.get("playerId", "?"))
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, lbl, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    writer.write(frame)
    fi += 1

cap.release()
writer.release()
print(f"✓ Video saved: {output_video}")

# ── 11. Download ──────────────────────────────────────────────────────────────
try:
    from google.colab import files
    if output_video.exists():
        files.download(str(output_video))
    print("✓ Downloaded annotated video")
except ImportError:
    print(f"Results at: {output_video}")

print("\n" + "=" * 80)
print("✅ D-FINE TEST COMPLETE")
print("=" * 80)
