#!/usr/bin/env python3
"""
PASTE THIS ENTIRE CELL INTO COLAB.
RT-DETRv2-r50 detector test — full pipeline on Aston Villa vs PSG clip 1.mov
Replaces YOLO with RT-DETRv2-r50 (GPU) for person detection, keeps all other
pipeline components (BoT-SORT/Deep-EIoU, identity, ReID) unchanged.
"""

import os, sys, shutil, subprocess, json, time
from pathlib import Path

# --- Configuration ---
REPO_URL = "https://github.com/liys58/athlink-cv-service.git"
REPO = Path("/content/athlink-cv-service")
VIDEO = Path("/content/Aston villa vs Psg clip 1.mov")
JOB_ID = "rtdetr_test"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ATHLINK_FORCE_DEVICE"] = "cuda"
os.environ["ATHLINK_YOLO_HALF"] = "0"
os.environ["ATHLINK_MAX_PLAYER_SLOTS"] = "14"
os.environ["ATHLINK_ALLOW_NEW_PLAYER_SLOTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Set your HF token here or via Colab secrets
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # set in Colab secrets or paste below

print("=" * 80)
print("RT-DETRv2-r50 FULL PIPELINE TEST")
print("=" * 80)

print("\n--- 1. GPU & Repo Setup ---")
import torch
assert torch.cuda.is_available(), "Switch Colab runtime to T4 GPU."
print(f"GPU: {torch.cuda.get_device_name(0)}")

if REPO.exists():
    print("Refreshing repository...")
    subprocess.run(["git", "-C", str(REPO), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(REPO), "reset", "--hard", "origin/main"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)

os.chdir(REPO)
commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"HEAD: {commit_hash}")

print("\n--- 2. Dependencies ---")
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "ultralytics", "boxmot", "huggingface_hub",
                "transformers", "accelerate"], check=True)
try:
    import torchreid
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "git+https://github.com/KaiyangZhou/deep-person-reid.git"], check=True)
    import torchreid
print("✓ Dependencies OK")

# Link OSNet weights
models_dir = REPO / "models"
models_dir.mkdir(exist_ok=True)
src_weights = Path("/content/osnet_x1_0_msmt17.pt")
dst_weights = models_dir / "osnet_x1_0_msmt17.pt"
if src_weights.exists():
    shutil.copy2(src_weights, dst_weights)
    print("✓ OSNet weights linked")

# Check video
if not VIDEO.exists():
    print(f"⚠ Video not found at {VIDEO}")
    alt = Path("/content/1b16c594_villa_psg_40s_new.mp4")
    if alt.exists():
        VIDEO = alt
        print(f"Using fallback: {VIDEO}")
    else:
        print("ERROR: Upload 'Aston villa vs Psg clip 1.mov' to /content/")
        sys.exit(1)
print(f"Video: {VIDEO}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Patch TrackerCore to use RT-DETRv2-r50 instead of YOLO
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3. Patching detector → RT-DETRv2-r50 ---")

sys.path.insert(0, str(REPO))
for mod in list(sys.modules):
    if mod == "services" or mod.startswith("services."):
        del sys.modules[mod]

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image as PILImage
import numpy as np
import cv2

# Load RT-DETRv2-r50 once globally
print("Loading RT-DETRv2-r50 (this downloads ~160MB first time)...")
_rtdetr_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
_rtdetr_model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
_rtdetr_model = _rtdetr_model.cuda()
_rtdetr_model.eval()

# Find person class id in RT-DETRv2 id2label
_person_ids = {k for k, v in _rtdetr_model.config.id2label.items() if "person" in v.lower()}
print(f"✓ RT-DETRv2-r50 loaded | person class ids: {_person_ids}")

def _rtdetr_detect(frame_bgr: np.ndarray, conf_threshold: float = 0.35):
    """RT-DETRv2-r50 drop-in for YOLO detect. Returns np.array (N,6) [x1,y1,x2,y2,conf,cls=0]."""
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    inputs = _rtdetr_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _rtdetr_model(**inputs)

    target_sizes = torch.tensor([[h, w]], device="cuda")
    results = _rtdetr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=conf_threshold
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() in _person_ids:
            x1, y1, x2, y2 = box.tolist()
            # Clamp to frame
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                detections.append([x1, y1, x2, y2, score.item(), 0])

    return np.array(detections, dtype=np.float32) if detections else np.zeros((0, 6), dtype=np.float32)


# Monkey-patch TrackerCore._detect to use RT-DETRv2
import services.tracker_core as tc

_orig_detect = tc.TrackerCore._detect

def _patched_detect(self, frame):
    """Drop-in replacement for YOLO _detect.
    Returns np.array (N,6) [x1,y1,x2,y2,conf,cls=2.0] — same as original."""
    self._last_ball_det = None  # no ball detection from RT-DETR
    dets = _rtdetr_detect(frame, conf_threshold=0.35)
    if len(dets) == 0:
        return np.empty((0, 6))
    # Force class=2.0 (player) to match roboflow-normalised downstream expectations
    dets[:, 5] = 2.0
    return dets

tc.TrackerCore._detect = _patched_detect
print("✓ Detector patched: YOLO → RT-DETRv2-r50")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run Tracking Pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 4. Running Tracking Pipeline ---")
print("=" * 80)

from services.tracker_core import run_tracking
import io
from contextlib import redirect_stdout

output_buffer = io.StringIO()
frame_stride = 5
max_frames = None

t0 = time.time()
try:
    with redirect_stdout(output_buffer):
        tracking_results = run_tracking(
            video_path=str(VIDEO),
            job_id=JOB_ID,
            frame_stride=frame_stride,
            max_frames=max_frames,
            device="cuda"
        )
    elapsed = time.time() - t0
    full_output = output_buffer.getvalue()
    print(full_output[-4000:])
    print(f"\n✓ Tracking complete in {elapsed:.1f}s")

except Exception as e:
    print(f"✗ Tracking failed: {e}")
    import traceback; traceback.print_exc()
    print(output_buffer.getvalue()[-5000:])
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Identity Metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("IDENTITY METRICS")
print("=" * 80)

identity_metrics = {}
for line in full_output.split('\n'):
    if any(key in line for key in ['collapse_lock_creations', 'locks_created',
                                    'lock_retention_rate', 'valid_id_coverage']):
        parts = line.split('=')
        if len(parts) >= 2:
            try:
                identity_metrics[parts[0].strip()] = float(parts[1].strip().split()[0])
            except:
                pass

locks_created      = int(identity_metrics.get('locks_created', 0))
lock_retention     = identity_metrics.get('lock_retention_rate', 0.0)
collapse_creations = int(identity_metrics.get('collapse_lock_creations', 0))
valid_id_coverage  = identity_metrics.get('valid_id_coverage', 0.0)

print(f"\nGATE 1 locks_created >= 20 : {locks_created} → {'✓ PASS' if locks_created >= 20 else '✗ FAIL'}")
print(f"GATE 2 lock_retention >= 0.65: {lock_retention:.3f} → {'✓ PASS' if lock_retention >= 0.65 else '✗ FAIL'}")
print(f"GATE 3 collapse_lock_creations == 0: {collapse_creations} → {'✓ PASS' if collapse_creations == 0 else '✗ FAIL'}")
print(f"       valid_id_coverage: {valid_id_coverage:.3f}")

all_pass = locks_created >= 20 and lock_retention >= 0.65 and collapse_creations == 0
print(f"\n{'✅ ALL GATES PASS' if all_pass else '⚠️  INCOMPLETE'}")
print(f"Time: {elapsed:.1f}s | Frames: {len(tracking_results)} | Video: {VIDEO.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Render Annotated Video
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 5. Rendering Annotated Video ---")

output_video = Path(f"/content/rtdetr_annotated_{VIDEO.stem}.mp4")
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_video), fourcc, fps, (fw, fh))

# Build frame→players lookup
frame_map = {}
for frame_data in tracking_results:
    fi = frame_data.get("frameIndex", frame_data.get("frame_index", -1))
    frame_map[fi] = frame_data.get("players", frame_data.get("detections", []))

COLORS = {
    "locked":      (0, 255, 0),
    "revived":     (0, 220, 255),
    "provisional": (0, 165, 255),
    "unknown":     (128, 128, 128),
    "unassigned":  (60, 60, 60),
}
DASHED_SOURCES = {"provisional", "unknown", "unassigned"}

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=10, gap=5):
    pts = [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]
    for (ax,ay,bx,by) in pts:
        length = max(abs(bx-ax)+abs(by-ay), 1)
        seg = dash + gap
        n = max(int(length / seg), 1)
        for i in range(n):
            t0 = (i*seg) / length
            t1 = min((i*seg+dash) / length, 1.0)
            p0 = (int(ax+t0*(bx-ax)), int(ay+t0*(by-ay)))
            p1 = (int(ax+t1*(bx-ax)), int(ay+t1*(by-ay)))
            cv2.line(img, p0, p1, color, thickness)

fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    for p in frame_map.get(fi, []):
        pid   = p.get("playerId", "?")
        disp  = p.get("displayId", pid)
        bbox  = p.get("bbox", [])
        src   = p.get("assignment_source", "unknown")
        valid = p.get("identity_valid", False)
        if len(bbox) != 4:
            continue
        x1,y1,x2,y2 = [int(v) for v in bbox]
        color  = COLORS.get(src if valid else "unknown", (128,128,128))
        dashed = (src in DASHED_SOURCES) or (not valid)
        if dashed:
            draw_dashed_rect(frame, x1, y1, x2, y2, color)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = disp
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    writer.write(frame)
    fi += 1

cap.release()
writer.release()
print(f"✓ Video saved: {output_video}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Download Results
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google.colab import files
    print("\nDownloading results...")

    metrics_json = REPO / f"temp/{JOB_ID}/tracking/identity_metrics.json"
    if metrics_json.exists():
        files.download(str(metrics_json))

    if output_video.exists():
        files.download(str(output_video))

    print("✓ Downloaded")
except ImportError:
    print(f"\nResults at: {REPO}/temp/{JOB_ID}/  and  {output_video}")

print("\nRT-DETRv2-r50 TEST COMPLETE")
print("=" * 80)
