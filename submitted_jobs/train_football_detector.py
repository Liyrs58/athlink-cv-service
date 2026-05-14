# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0",
#   "torchvision",
#   "transformers>=4.40",
#   "datasets>=2.18",
#   "huggingface_hub>=0.23",
#   "Pillow",
#   "albumentations>=1.4",
#   "accelerate>=0.27",
# ]
# ///
"""
Fine-tune D-FINE-small on martinjolif/football-player-detection.

4 classes: ball=0  goalkeeper=1  player=2  referee=3
Pushes trained model to Liyrs58/dfine-football-detector on Hub.

Submit via HF Jobs (t4-small, 2h):
  from huggingface_hub import HfApi, get_token
  api = HfApi()
  job = api.run_uv_job(
      script="submitted_jobs/train_football_detector.py",
      flavor="t4-small", timeout=7200,
      env={"PYTHONUNBUFFERED": "1"},
      secrets={"HF_TOKEN": get_token()},
  )
  print(job.id)
"""

import os, re
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_tree
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
import albumentations as A

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_ID   = "ustc-community/dfine-small-coco"
DATASET_ID = "martinjolif/football-player-detection"
HUB_OUT    = "Liyrs58/dfine-football-detector"

ID2LABEL = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Load image processor ────────────────────────────────────────────────────
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

# ── Download and parse all YOLO label files ─────────────────────────────────

def _fetch_labels(split_dir: str) -> Dict[str, List]:
    """
    Download every .txt in data/<split_dir>/labels/ and return
    {stem: {"bboxes": [[x1,y1,x2,y2], ...], "cats": [int, ...]}}
    Bboxes converted from YOLO normalized xywh → absolute pascal_voc xyxy
    using placeholder img size 1920x1080 (actual size corrected per image later).
    """
    labels: Dict[str, dict] = {}
    try:
        tree = list_repo_tree(
            DATASET_ID, repo_type="dataset", token=HF_TOKEN,
            path_in_repo=f"data/{split_dir}/labels",
        )
        paths = [item.path for item in tree if item.path.endswith(".txt")]
    except Exception as e:
        print(f"[warn] list_repo_tree failed for {split_dir}: {e}")
        paths = []

    for fpath in paths:
        stem = Path(fpath).stem
        try:
            local = hf_hub_download(
                DATASET_ID, filename=fpath,
                repo_type="dataset", token=HF_TOKEN,
            )
            lines = Path(local).read_text().strip().splitlines()
        except Exception:
            lines = []

        raw_boxes, cats = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            raw_boxes.append((cls_id, x_c, y_c, w, h))
            cats.append(cls_id)

        labels[stem] = {"raw": raw_boxes}

    print(f"[labels] {split_dir}: {len(labels)} files")
    return labels


def _to_pascal(x_c, y_c, w, h, iw, ih):
    x1 = max(0.0, (x_c - w / 2) * iw)
    y1 = max(0.0, (y_c - h / 2) * ih)
    x2 = min(float(iw), (x_c + w / 2) * iw)
    y2 = min(float(ih), (y_c + h / 2) * ih)
    return [x1, y1, x2, y2]


def build_records(hf_split, label_map: Dict[str, dict], split_name: str):
    """
    Pair each HF image row with its YOLO labels by filename stem.
    HF parquet rows don't carry filenames, so we sort both sides and zip.
    """
    stems = sorted(label_map.keys())
    records = []
    for idx, row in enumerate(hf_split):
        img: Image.Image = row["image"].convert("RGB")
        iw, ih = img.size
        stem = stems[idx] if idx < len(stems) else None
        bboxes, cats = [], []
        if stem and stem in label_map:
            for cls_id, x_c, y_c, w, h in label_map[stem]["raw"]:
                bbox = _to_pascal(x_c, y_c, w, h, iw, ih)
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if bw > 4 and bh > 4:
                    bboxes.append(bbox)
                    cats.append(cls_id)
        records.append({
            "image":    img,
            "image_id": idx,
            "objects":  {
                "bbox":     bboxes,
                "category": cats,
                "area":     [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes],
            },
        })
    print(f"[dataset] {split_name}: {len(records)} rows, "
          f"{sum(len(r['objects']['category']) for r in records)} total boxes")
    return records


# ── Augmentation ────────────────────────────────────────────────────────────
_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
    A.RandomScale(scale_limit=0.25, p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.15),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(
    format="pascal_voc", label_fields=["cats"], min_visibility=0.3,
))


def augment(record):
    img = np.array(record["image"])
    bboxes = record["objects"]["bbox"]
    cats   = record["objects"]["category"]
    if bboxes:
        try:
            out = _AUG(image=img, bboxes=bboxes, cats=cats)
            record["image"]              = Image.fromarray(out["image"])
            record["objects"]["bbox"]    = [list(b) for b in out["bboxes"]]
            record["objects"]["category"] = out["cats"]
            record["objects"]["area"]    = [
                (b[2]-b[0])*(b[3]-b[1]) for b in out["bboxes"]
            ]
        except Exception:
            pass
    return record


# ── Collator ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    images = [r["image"] for r in batch]
    targets = []
    for r in batch:
        anns = [
            {
                "image_id":    r["image_id"],
                "category_id": cat,
                "bbox":        bbox,
                "area":        (bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
                "iscrowd":     0,
            }
            for cat, bbox in zip(r["objects"]["category"], r["objects"]["bbox"])
        ]
        targets.append({"image_id": r["image_id"], "annotations": anns})
    return processor(images=images, annotations=targets, return_tensors="pt")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}  ({torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'})")

    # Load raw HF dataset (image column only)
    raw = load_dataset(DATASET_ID, token=HF_TOKEN, trust_remote_code=True)

    # Download label files for each split
    train_labels = _fetch_labels("train")
    val_labels   = _fetch_labels("valid")

    train_records = [augment(r) for r in build_records(raw["train"],      train_labels, "train")]
    val_records   =              build_records(raw["validation"], val_labels,   "val")

    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_list(train_records)
    val_ds   = HFDataset.from_list(val_records)

    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_ID,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        token=HF_TOKEN,
    )

    args = TrainingArguments(
        output_dir="dfine_football_out",
        num_train_epochs=40,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=True,
        hub_model_id=HUB_OUT,
        report_to="none",
        dataloader_num_workers=2,
        fp16=(dev == "cuda"),
    )

    if args.push_to_hub and not args.hub_token:
        args.hub_token = HF_TOKEN

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.push_to_hub(commit_message="D-FINE-small football: ball/goalkeeper/player/referee (40 epochs)")
    print(f"\nDone → https://huggingface.co/{HUB_OUT}")


if __name__ == "__main__":
    main()
