# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0",
#   "torchvision",
#   "torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git",
#   "huggingface_hub>=0.20",
#   "Pillow",
# ]
# ///
"""
Phase 2: Fine-tune OSNet x1.0 on sports player crops for football ReID.

Base weights: CondadosAI/osnet-trackers / sports_model.pth.tar-60
Output:       Liyrs58/football-osnet-reid / football_osnet_x1_0.pth.tar
"""

import os
import sys
import shutil
import tempfile

import torch
from huggingface_hub import hf_hub_download, HfApi, get_token

# ---------------------------------------------------------------------------
# 1. Download base sports weights
# ---------------------------------------------------------------------------
print("[ReID-Train] Downloading base sports weights from CondadosAI/osnet-trackers ...")
base_weights_path = hf_hub_download(
    repo_id="CondadosAI/osnet-trackers",
    filename="sports_model.pth.tar-60",
    local_dir="/tmp/reid_weights",
    local_dir_use_symlinks=False,
)
print(f"[ReID-Train] Base weights at: {base_weights_path}")

# ---------------------------------------------------------------------------
# 2. Build OSNet x1.0 via torchreid
# ---------------------------------------------------------------------------
import torchreid

datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="market1501",
    targets="market1501",
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=100,
    transforms=["random_flip", "color_jitter", "random_erasing"],
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=False,
)

# Load sports base weights (partial — head may differ due to num_classes)
print("[ReID-Train] Loading base sports weights ...")
checkpoint = torch.load(base_weights_path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)

# Strip classifier head keys that won't match
backbone_state = {
    k: v for k, v in state_dict.items()
    if not k.startswith("classifier") and not k.startswith("module.classifier")
}
missing, unexpected = model.load_state_dict(backbone_state, strict=False)
print(f"[ReID-Train] Backbone loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

model = model.cuda() if torch.cuda.is_available() else model

# ---------------------------------------------------------------------------
# 3. Optimizer + scheduler
# ---------------------------------------------------------------------------
optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003,
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="cosine",
    stepsize=60,
)

# ---------------------------------------------------------------------------
# 4. Engine — ImageSoftmaxEngine with label smoothing
# ---------------------------------------------------------------------------
SAVE_DIR = "/tmp/reid_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
)

engine.run(
    save_dir=SAVE_DIR,
    max_epoch=60,
    eval_freq=10,
    print_freq=50,
    test_only=False,
)

# ---------------------------------------------------------------------------
# 5. Upload best checkpoint to HF Hub
# ---------------------------------------------------------------------------
best_ckpt = os.path.join(SAVE_DIR, "model.pth.tar-60")
if not os.path.exists(best_ckpt):
    # Fall back to last epoch checkpoint
    candidates = sorted(
        [f for f in os.listdir(SAVE_DIR) if f.startswith("model.pth.tar")],
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    if candidates:
        best_ckpt = os.path.join(SAVE_DIR, candidates[0])

output_filename = "football_osnet_x1_0.pth.tar"
output_path = os.path.join(SAVE_DIR, output_filename)
if best_ckpt != output_path:
    shutil.copy2(best_ckpt, output_path)

hf_token = os.environ.get("HF_TOKEN") or get_token()
api = HfApi(token=hf_token)

TARGET_REPO = "Liyrs58/football-osnet-reid"
api.create_repo(repo_id=TARGET_REPO, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo=output_filename,
    repo_id=TARGET_REPO,
    repo_type="model",
    commit_message="Upload football-specific OSNet x1.0 ReID weights (Phase 2)",
)

print(f"[ReID-Train] Done. Uploaded to {TARGET_REPO}/{output_filename}")
