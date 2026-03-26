"""
Fine-tune football_yolo11.pt on football-specific data.

Downloads from Roboflow (free, no login needed for public datasets) and
optionally from SoccerNet. Fine-tunes the existing football_yolo11.pt
(25M params, classes: ball/goalkeeper/player/referee) so you keep those
classes but improve accuracy on your use cases.

Usage:
    cd /Users/rudra/Desktop/athlink-cv-service
    source .venv/bin/activate
    python3 scripts/train_football_model.py

Output:
    models/football_yolo11_finetuned.pt   ← drop-in replacement

Requirements (already in your venv):
    ultralytics >= 8.1.0
    roboflow (pip install roboflow  — only needed if downloading from Roboflow)
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models" / "football_yolo11.pt"
OUTPUT_DIR = ROOT / "models" / "train_run"
DATASET_DIR = ROOT / "datasets"
DATASET_DIR.mkdir(exist_ok=True)

# ── Training hyper-params ─────────────────────────────────────────────────────
EPOCHS = 50           # increase to 100 for a full run; 50 is good for fine-tuning
IMG_SIZE = 640        # 1280 gives better small-ball detection but needs more VRAM
BATCH = 8             # reduce to 4 if you hit OOM on RunPod; increase to 16 on A100
PATIENCE = 20         # early-stop if no improvement for 20 epochs
WORKERS = 4


def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def download_roboflow_dataset() -> Path:
    """
    Downloads the Roboflow football-players-detection dataset (YOLOv8 format).
    Classes: ball, goalkeeper, player, referee  — matches your model exactly.

    Dataset: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
    ~3,600 annotated images, free public download.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow…")
        _pip_install("roboflow")
        from roboflow import Roboflow

    rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # free key at roboflow.com
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(2)
    dataset = version.download("yolov8", location=str(DATASET_DIR / "roboflow_football"))
    return Path(dataset.location) / "data.yaml"


def download_soccernet_dataset() -> Path:
    """
    Downloads SoccerNet tracking data (bboxes + tracklet IDs).
    Requires free registration at https://www.soccer-net.org/ to get a password.

    Set env var SOCCERNET_PASS before running.
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("Installing SoccerNet…")
        _pip_install("SoccerNet")
        from SoccerNet.Downloader import SoccerNetDownloader

    sn_dir = DATASET_DIR / "soccernet_tracking"
    sn_dir.mkdir(exist_ok=True)

    password = os.getenv("SOCCERNET_PASS", "")
    if not password:
        print("⚠  SOCCERNET_PASS env var not set — skipping SoccerNet download.")
        print("   Register free at https://www.soccer-net.org/data and set:")
        print("   export SOCCERNET_PASS=your_password")
        return None

    dl = SoccerNetDownloader(LocalDirectory=str(sn_dir))
    dl.password = password
    dl.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
    print("SoccerNet tracking data downloaded to", sn_dir)
    return sn_dir


def create_dataset_yaml(data_yaml_path: Path) -> Path:
    """
    If you want to merge multiple datasets, write a combined data.yaml here.
    For now just returns the Roboflow yaml directly.
    """
    return data_yaml_path


def train(data_yaml: Path):
    """Run YOLO fine-tuning."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Base model : {BASE_MODEL}")
    print(f"Dataset    : {data_yaml}")
    print(f"Epochs     : {EPOCHS}  |  img_size: {IMG_SIZE}  |  batch: {BATCH}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
        project=str(OUTPUT_DIR),
        name="finetune",
        exist_ok=True,
        # Fine-tuning settings — lower LR to avoid overwriting learned features
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Augmentation tuned for broadcast football footage
        hsv_h=0.015,    # slight hue shift — handles different kit colours under lights
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.0,     # football — don't flip upside down
        fliplr=0.5,
        mosaic=0.8,
        # Save best weights
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    # Copy best weights to models/
    best = OUTPUT_DIR / "finetune" / "weights" / "best.pt"
    dest = ROOT / "models" / "football_yolo11_finetuned.pt"
    if best.exists():
        import shutil
        shutil.copy(best, dest)
        print(f"\n✅ Fine-tuned model saved to: {dest}")
        print(f"   To use it, set: YOLO_MODEL_PATH={dest}")
        print(f"   Or rename it to football_yolo11.pt to use as default.")
    else:
        print(f"\n⚠  best.pt not found at {best} — check training output in {OUTPUT_DIR}")

    return results


def validate(model_path: Path, data_yaml: Path):
    """Quick validation pass to print mAP stats."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=IMG_SIZE, batch=BATCH)
    print("\n── Validation metrics ──")
    print(f"  mAP50     : {metrics.box.map50:.3f}")
    print(f"  mAP50-95  : {metrics.box.map:.3f}")
    print(f"  Precision : {metrics.box.mp:.3f}")
    print(f"  Recall    : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune football YOLO model")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use existing datasets/)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training, just validate the finetuned model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH = args.batch
    IMG_SIZE = args.imgsz

    # Step 1: Download dataset
    roboflow_yaml = DATASET_DIR / "roboflow_football" / "data.yaml"
    if not args.skip_download:
        print("Downloading Roboflow football dataset…")
        print("NOTE: You need a free Roboflow API key.")
        print("Get one at https://roboflow.com  then edit this script and replace YOUR_API_KEY_HERE\n")
        try:
            roboflow_yaml = download_roboflow_dataset()
        except Exception as e:
            print(f"Roboflow download failed: {e}")
            if not roboflow_yaml.exists():
                print("No local dataset found either. Exiting.")
                sys.exit(1)
            print(f"Using existing dataset at {roboflow_yaml}")

        # Optional: also download SoccerNet
        download_soccernet_dataset()
    else:
        if not roboflow_yaml.exists():
            print(f"Dataset not found at {roboflow_yaml}. Run without --skip-download first.")
            sys.exit(1)
        print(f"Using existing dataset: {roboflow_yaml}")

    data_yaml = create_dataset_yaml(roboflow_yaml)

    if args.validate_only:
        finetuned = ROOT / "models" / "football_yolo11_finetuned.pt"
        if not finetuned.exists():
            print("No finetuned model found. Run training first.")
            sys.exit(1)
        validate(finetuned, data_yaml)
    else:
        # Step 2: Train
        train(data_yaml)

        # Step 3: Validate
        finetuned = ROOT / "models" / "football_yolo11_finetuned.pt"
        if finetuned.exists():
            print("\nRunning validation on fine-tuned model…")
            validate(finetuned, data_yaml)
