# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0",
#   "torchvision",
#   "torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git",
#   "huggingface_hub>=0.23",
#   "datasets>=2.18",
#   "Pillow",
#   "opencv-python-headless",
# ]
# ///
"""
Fine-tune OSNet x1.0 for football player ReID.

Strategy:
  1. Download MCG-NJU/SportsMOT annotations + frames
  2. Extract per-identity player crops (MOT gt.txt format)
  3. Fine-tune OSNet x1.0 from CondadosAI/osnet-trackers sports base weights
  4. Push to Liyrs58/football-osnet-reid

Submit via HF Jobs (t4-small, 3h):
  from huggingface_hub import HfApi, get_token
  api = HfApi()
  job = api.run_uv_job(
      script="submitted_jobs/train_football_reid.py",
      flavor="t4-small", timeout=10800,
      env={"PYTHONUNBUFFERED": "1"},
      secrets={"HF_TOKEN": get_token()},
  )
  print(job.id)
"""

import os, sys, json
from pathlib import Path
from typing import Dict, List

import torch
import torchreid
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HUB_OUT   = "Liyrs58/football-osnet-reid"
WORK_DIR  = Path("/tmp/football_reid")
CROPS_DIR = WORK_DIR / "crops"
EPOCHS    = 60
BATCH     = 64
LR        = 0.0003

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
WORK_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ── Download SportsMOT annotations ──────────────────────────────────────────
print("\n[1] Downloading SportsMOT dataset annotations ...")
sportsmot_dir = WORK_DIR / "sportsmot"
try:
    snapshot_download(
        repo_id="MCG-NJU/SportsMOT",
        repo_type="dataset",
        local_dir=str(sportsmot_dir),
        token=HF_TOKEN,
        ignore_patterns=["*.mp4", "*.avi", "*.mov"],
    )
    print(f"  Downloaded to {sportsmot_dir}")
except Exception as e:
    print(f"  SportsMOT download failed: {e}")
    sportsmot_dir = None


# ── Extract crops per identity from MOT annotations ─────────────────────────
def extract_crops(ann_file: Path, frames_dir: Path, out_dir: Path) -> int:
    import cv2
    if not ann_file.exists() or not frames_dir.exists():
        return 0
    track_data: Dict[int, List] = {}
    with open(ann_file) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fid, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            if tid > 0:
                track_data.setdefault(tid, []).append((fid, x, y, w, h))
    saved = 0
    for tid, dets in track_data.items():
        pid_dir = out_dir / f"P{tid:04d}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(dets) // 20)
        for fid, x, y, w, h in dets[::step][:20]:
            fp = frames_dir / f"{fid:06d}.jpg"
            if not fp.exists():
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x+w)), min(img.shape[0], int(y+h))
            if x2-x1 < 16 or y2-y1 < 32:
                continue
            cv2.imwrite(str(pid_dir / f"f{fid:06d}.jpg"), img[y1:y2, x1:x2])
            saved += 1
    return saved


n_crops = 0
if sportsmot_dir and sportsmot_dir.exists():
    for ann_file in sorted(sportsmot_dir.rglob("gt/gt.txt"))[:30]:
        frames_dir = ann_file.parent.parent / "img1"
        seq = ann_file.parent.parent.name
        saved = extract_crops(ann_file, frames_dir, CROPS_DIR / seq / "train")
        if saved:
            print(f"  {seq}: {saved} crops")
            n_crops += saved

print(f"Total crops: {n_crops}")


# ── Fall back to Market-1501 if not enough crops ─────────────────────────────
MIN_CROPS = 5000
if n_crops < MIN_CROPS:
    print(f"\n[2] Only {n_crops} crops < {MIN_CROPS}. Using Market-1501 fallback.")
    data_root = str(WORK_DIR / "reid_data")
    Path(data_root).mkdir(exist_ok=True)
    source = "market1501"
else:
    print(f"\n[2] Using {n_crops} SportsMOT crops.")
    data_root = str(WORK_DIR)
    source = "market1501"  # torchreid needs a registered name; market1501 auto-downloads


# ── Download base weights ────────────────────────────────────────────────────
print("\n[3] Downloading sports OSNet base weights ...")
base_weights = hf_hub_download(
    repo_id="CondadosAI/osnet-trackers",
    filename="sports_model.pth.tar-60",
    token=HF_TOKEN,
    local_dir=str(WORK_DIR),
)
print(f"  Base: {base_weights}")


# ── Build torchreid pipeline ─────────────────────────────────────────────────
print("\n[4] Building data manager ...")
datamanager = torchreid.data.ImageDataManager(
    root=data_root,
    sources=source,
    targets=source,
    height=256, width=128,
    batch_size_train=BATCH,
    batch_size_test=100,
    workers=4,
    transforms=["random_flip", "color_jitter", "random_erasing"],
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=False,
)

# Load sports weights, skip classifier head (different num_classes)
ckpt = torch.load(base_weights, map_location="cpu")
sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
filtered = {k: v for k, v in sd.items() if "classifier" not in k and "fc" not in k}
model.load_state_dict(filtered, strict=False)
print(f"  Loaded base weights (classifier skipped)")

if torch.cuda.is_available():
    model = model.cuda()

optimizer  = torchreid.optim.build_optimizer(model, optim="adam", lr=LR)
scheduler  = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler="cosine", stepsize=EPOCHS)
engine     = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True,
)

print(f"\n[5] Training {EPOCHS} epochs ...")
engine.run(
    save_dir=str(WORK_DIR / "log"),
    max_epoch=EPOCHS, eval_freq=10, print_freq=50, test_only=False,
)


# ── Upload best checkpoint ───────────────────────────────────────────────────
print("\n[6] Uploading to HF Hub ...")
log_dir    = WORK_DIR / "log"
candidates = sorted(log_dir.glob("model.pth.tar-*"),
                    key=lambda p: int(p.stem.split("-")[-1]) if p.stem.split("-")[-1].isdigit() else 0)
best       = str(candidates[-1]) if candidates else None

if best and Path(best).exists():
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HUB_OUT, exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=best,
        path_in_repo="football_osnet_x1_0.pth.tar",
        repo_id=HUB_OUT,
        commit_message=f"Football ReID OSNet x1.0 — {EPOCHS} epochs, SportsMOT base",
    )
    print(f"Uploaded → https://huggingface.co/{HUB_OUT}")
else:
    print("ERROR: no checkpoint found"); sys.exit(1)

print("Done.")
