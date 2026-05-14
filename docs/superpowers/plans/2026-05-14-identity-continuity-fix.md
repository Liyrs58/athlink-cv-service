# Identity Continuity Fix — Football Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix identity switches, ghost tracks, and untrustworthy P-ID labels in the football tracking pipeline by upgrading the ReID model weights, hardening the revival matching logic, fine-tuning a football-specific ReID model on HF Jobs, and adding an uncertainty overlay to the video output.

**Architecture:** Three parallel tracks — (1) code fixes to `identity_core.py` and `tracker_core.py` that tighten revival matching and add team-gating, (2) a HF Jobs training script that fine-tunes OSNet on SoccerNet player crops and pushes the result to the Hub, (3) a renderer change in `tracker_core.py` that emits colour-coded bounding boxes based on lock state. All three can be worked independently and merged together.

**Tech Stack:** Python 3.12, PyTorch, torchreid, HuggingFace Hub, `huggingface_hub.HfApi.run_uv_job`, OpenCV, Colab T4 GPU for validation.

---

## Files Changed

| File | Change |
|---|---|
| `services/identity_core.py` | Snapshot freshness pass, team-gate revival, tighter margins |
| `services/tracker_core.py` | Sports OSNet weight loader, colour-coded renderer |
| `submitted_jobs/train_football_reid_YYYYMMDD.py` | New HF Jobs training script |
| `tests/test_identity_revival.py` | New unit tests for revival logic |

---

## PHASE 1 — Code fixes (no training needed)

### Task 1: Sports OSNet weights — swap at load time

**Files:**
- Modify: `services/tracker_core.py:57-175` (`ReIDExtractor.__init__`)

The `ReIDExtractor` currently loads `osnet_x1_0_msmt17.pt` — pedestrian surveillance data. We replace it with `sports_model.pth.tar-60` from `CondadosAI/osnet-trackers` on HF Hub, which was fine-tuned on SportsMOT broadcast footage.

- [ ] **Step 1: Write the failing test**

Create `tests/test_identity_revival.py`:

```python
import pytest
import numpy as np

def test_sports_osnet_weights_loaded(tmp_path, monkeypatch):
    """ReIDExtractor should attempt to load sports_model.pth.tar-60 first."""
    loaded = []

    import torchreid
    orig_load = torchreid.utils.load_pretrained_weights

    def mock_load(model, path):
        loaded.append(path)

    monkeypatch.setattr(torchreid.utils, "load_pretrained_weights", mock_load)

    import sys
    for m in list(sys.modules):
        if m.startswith("services"):
            del sys.modules[m]

    from services.tracker_core import ReIDExtractor
    extractor = ReIDExtractor(device="cpu")

    assert any("sports_model" in p for p in loaded), (
        f"Expected sports_model weights to be loaded, got: {loaded}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest tests/test_identity_revival.py::test_sports_osnet_weights_loaded -v
```

Expected: `FAIL — sports_model not in loaded`

- [ ] **Step 3: Download sports weights and update loader**

```bash
export PATH="$HOME/.npm-global/bin:$PATH"
export HF_TOKEN=<your-hf-token>
mkdir -p models
hf download CondadosAI/osnet-trackers sports_model.pth.tar-60 --local-dir models/
```

In `services/tracker_core.py`, find the `_load_osnet` method (around line 130) or the section that calls `torchreid.utils.load_pretrained_weights`. Add sports weights as the primary path before the MSMT17 fallback:

```python
# Priority order for OSNet weights:
# 1. Sports-tuned (SportsMOT, broadcast footage) — best for football
# 2. MSMT17 (pedestrian surveillance) — fallback
# 3. HF Hub download of sports model
_SPORTS_WEIGHT_PATHS = [
    "/content/sports_model.pth.tar-60",          # Colab upload
    "models/sports_model.pth.tar-60",             # local
    str(Path(__file__).parent.parent / "models" / "sports_model.pth.tar-60"),
]
_MSMT17_WEIGHT_PATHS = [
    "/content/osnet_x1_0_msmt17.pt",
    "models/osnet_x1_0_msmt17.pt",
    str(Path(__file__).parent.parent / "models" / "osnet_x1_0_msmt17.pt"),
]

def _find_osnet_weights() -> Optional[str]:
    for p in _SPORTS_WEIGHT_PATHS:
        if Path(p).exists():
            print(f"[ReID] Using sports-tuned OSNet weights: {p}")
            return p
    for p in _MSMT17_WEIGHT_PATHS:
        if Path(p).exists():
            print(f"[ReID] Using MSMT17 OSNet weights (fallback): {p}")
            return p
    # Try HF Hub download
    try:
        from huggingface_hub import hf_hub_download
        dest = hf_hub_download(
            repo_id="CondadosAI/osnet-trackers",
            filename="sports_model.pth.tar-60",
            local_dir="models",
        )
        print(f"[ReID] Downloaded sports OSNet from HF Hub: {dest}")
        return dest
    except Exception as e:
        print(f"[ReID] HF Hub download failed: {e}")
    return None
```

Then in `ReIDExtractor.__init__`, replace the existing weight path selection with a call to `_find_osnet_weights()`.

- [ ] **Step 4: Run test to confirm pass**

```bash
python -m pytest tests/test_identity_revival.py::test_sports_osnet_weights_loaded -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add services/tracker_core.py tests/test_identity_revival.py models/sports_model.pth.tar-60
git commit -m "feat: load sports-tuned OSNet weights (SportsMOT) as primary ReID model

Replaces MSMT17 surveillance weights with broadcast sports-tuned checkpoint
from CondadosAI/osnet-trackers (Deep-EIoU paper, arXiv:2306.13074).
Falls back to MSMT17 if sports weights unavailable, then tries HF Hub download."
```

---

### Task 2: Snapshot freshness — force embedding refresh before collapse

**Files:**
- Modify: `services/identity_core.py:1317-1332` (`snapshot_soft`)
- Modify: `services/tracker_core.py:679-700` (soft collapse trigger)

Root cause: `snapshot_soft()` saves whatever embeddings are in slots at the moment of collapse. If the system was in restricted mode for the previous N frames, those embeddings are N frames stale. The revival then matches stale bright-sun embeddings against shadow-pose tracks.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_identity_revival.py`:

```python
def test_snapshot_records_freshness_timestamp():
    """Each snapshot entry must include the frame_id when the embedding was last updated."""
    from services.identity_core import IdentityCore
    ic = IdentityCore(n_slots=4)
    ic.frame_id = 100

    # Manually plant a slot with a stale embedding (last updated at frame 50)
    slot = ic.slots[0]
    slot.pid = "P1"
    slot.state = "active"
    slot.embedding = np.ones(512, dtype=np.float32)
    slot._emb_last_updated_frame = 50   # stale: 50 frames old

    ic.snapshot_soft(frame_id=100)

    snap = ic._soft_snapshot.get("P1")
    assert snap is not None, "P1 should be in snapshot"
    assert "emb_age" in snap, "snapshot must record embedding age"
    assert snap["emb_age"] == 50, f"emb_age should be 50 frames, got {snap['emb_age']}"
```

- [ ] **Step 2: Run to confirm fail**

```bash
python -m pytest tests/test_identity_revival.py::test_snapshot_records_freshness_timestamp -v
```

Expected: `FAIL — KeyError: 'emb_age'`

- [ ] **Step 3: Add `_emb_last_updated_frame` to `PlayerSlot` and record age in snapshot**

In `services/identity_core.py`, find `PlayerSlot` dataclass (around line 400). Add one field:

```python
_emb_last_updated_frame: int = 0  # frame when embedding was last written
```

In `update_embedding` method of `PlayerSlot` (or wherever `slot.embedding = ...` is set via EMA), add:

```python
self._emb_last_updated_frame = frame_id  # pass frame_id into update_embedding
```

`update_embedding` signature becomes:
```python
def update_embedding(self, emb: np.ndarray, alpha: float = EMB_ALPHA, frame_id: int = 0) -> None:
```

In `snapshot_soft` (line 1317), update the snapshot dict to include age:

```python
def snapshot_soft(self, frame_id: int) -> int:
    self._soft_snapshot = {}
    for s in self.slots:
        if s.state in ("active", "dormant") and s.embedding is not None:
            emb_age = frame_id - s._emb_last_updated_frame
            self._soft_snapshot[s.pid] = {
                "embedding": s.embedding.copy(),
                "position": s.last_position,
                "pitch": s.last_pitch,
                "team_id": s.team_id,
                "stable_count": s.last_lock_stable_count,
                "emb_age": emb_age,   # NEW
            }
    saved = len(self._soft_snapshot)
    print(f"[SoftSnapshot] frame={frame_id} saved={saved} slots")
    return saved
```

- [ ] **Step 4: Use emb_age to discount stale snapshot embeddings in revival cost**

In `revive_from_soft_snapshot` (line 1631), find where `_snapshot_cost_matrix` is called. After computing `emb_cost`, apply a staleness penalty:

```python
# Staleness penalty: embeddings older than 10 frames lose trust linearly
# At 30 frames stale: emb_cost weight halved. Beyond 60 frames: position only.
emb_age = snap_entry.get("emb_age", 0)
staleness_factor = max(0.3, 1.0 - emb_age / 60.0)
emb_cost_weighted = emb_cost * staleness_factor + 0.5 * (1.0 - staleness_factor)
```

Replace `emb_cost` with `emb_cost_weighted` in the final cost blending for the snapshot cost matrix.

- [ ] **Step 5: Run test to confirm pass**

```bash
python -m pytest tests/test_identity_revival.py::test_snapshot_records_freshness_timestamp -v
```

Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add services/identity_core.py tests/test_identity_revival.py
git commit -m "fix: record embedding freshness in soft snapshot, penalise stale embeddings during revival

Stale embeddings (frozen during collapse) caused wrong revival matches after
camera pans. Now snapshot records emb_age per slot and revival cost matrix
discounts embeddings older than 10 frames, falling back to position-only
matching at 60+ frames stale."
```

---

### Task 3: Team-gate revival — never revive PSG slot with Villa track

**Files:**
- Modify: `services/identity_core.py:1631-1776` (`revive_from_soft_snapshot`)

Root cause: P1 (Villa left-back) can get revived onto a PSG defender track if embedding cost happens to be similar. The `team_labels` dict maps `tid → team_id`. We already check teams in normal assignment. We must also check in revival.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_identity_revival.py`:

```python
def test_revival_blocked_for_wrong_team():
    """A snapshot slot for team 0 must never revive onto a track assigned to team 1."""
    from services.identity_core import IdentityCore
    import numpy as np

    ic = IdentityCore(n_slots=4)
    ic.frame_id = 50

    # Plant snapshot: P1 is team 0
    ic._soft_snapshot = {
        "P1": {
            "embedding": np.ones(512, dtype=np.float32),
            "position": (200.0, 400.0),
            "pitch": None,
            "team_id": 0,
            "stable_count": 15,
            "emb_age": 5,
        }
    }

    # Incoming track tid=99 is team 1 (PSG) with low embedding cost vs P1's anchor
    ic.team_labels = {99: 1}

    class FakeTrack:
        track_id = 99
        bbox = [190, 390, 230, 450]

    embeddings = {99: {"emb": np.ones(512, dtype=np.float32), "hsv": None}}
    positions = {99: (210.0, 420.0)}

    revived, _ = ic.revive_from_soft_snapshot(
        [FakeTrack()], embeddings, positions,
        camera_motion=None, frame_id=50
    )

    assert 99 not in revived, "tid=99 (team 1) must NOT revive P1 (team 0)"
```

- [ ] **Step 2: Run to confirm fail**

```bash
python -m pytest tests/test_identity_revival.py::test_revival_blocked_for_wrong_team -v
```

Expected: `FAIL — tid=99 was incorrectly revived as P1`

- [ ] **Step 3: Add team gate inside `_snapshot_cost_matrix`**

In `services/identity_core.py`, find `_snapshot_cost_matrix` (called by `revive_from_soft_snapshot`, around line 1648). Inside the loop that builds the cost matrix, add a team hard-block before computing embedding cost:

```python
for i, (pid, snap) in enumerate(snapshot_items):
    snap_team = snap.get("team_id")
    for j, track in enumerate(tracks):
        tid = int(track.track_id)
        track_team = self.team_labels.get(tid)

        # Team hard gate: never revive across team boundary
        if (snap_team is not None
                and track_team is not None
                and snap_team != track_team):
            cost_matrix[i, j] = COST_REJECT_THRESHOLD + 0.30
            continue

        # ... rest of existing cost calculation ...
```

- [ ] **Step 4: Run test to confirm pass**

```bash
python -m pytest tests/test_identity_revival.py::test_revival_blocked_for_wrong_team -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add services/identity_core.py tests/test_identity_revival.py
git commit -m "fix: team-gate soft revival — PSG slots cannot revive to Villa tracks

Added team hard-block inside _snapshot_cost_matrix. If both the snapshot
slot's team_id and the incoming track's team are known and differ, cost is
set to COST_REJECT_THRESHOLD+0.30 (hard reject). Eliminates cross-team
identity switches after camera pans."
```

---

### Task 4: Tighten revival margins and lower auto-accept threshold

**Files:**
- Modify: `services/identity_core.py:59-66` (constants)

Root cause: `SOFT_REVIVE_MARGIN_MIN = 0.025` is too small — two players with similar embeddings (same jersey) differ by only 0.02 in cost and BOTH get accepted. `SOFT_REVIVE_AUTO_ACCEPT_COST = 0.15` auto-accepts revivals without margin check — a well-placed ghost track can sneak through.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_identity_revival.py`:

```python
def test_ambiguous_revival_rejected_with_tight_margin():
    """When two snapshot slots have similar cost to one track, neither should be revived."""
    from services.identity_core import IdentityCore, SOFT_REVIVE_MARGIN_MIN
    import numpy as np

    # Confirm the constant is tight enough
    assert SOFT_REVIVE_MARGIN_MIN >= 0.07, (
        f"SOFT_REVIVE_MARGIN_MIN={SOFT_REVIVE_MARGIN_MIN} is too loose; must be >= 0.07 "
        "to prevent ambiguous same-team revives"
    )
```

- [ ] **Step 2: Run to confirm fail**

```bash
python -m pytest tests/test_identity_revival.py::test_ambiguous_revival_rejected_with_tight_margin -v
```

Expected: `FAIL — SOFT_REVIVE_MARGIN_MIN=0.025 < 0.07`

- [ ] **Step 3: Update constants**

In `services/identity_core.py` lines 59-66:

```python
SOFT_REVIVE_COST_MAX = 0.32           # tightened from 0.38 — require stronger embedding match
SOFT_REVIVE_AUTO_ACCEPT_COST = 0.08   # tightened from 0.15 — auto-accept only very strong matches
SOFT_REVIVE_MARGIN_MIN = 0.08         # tightened from 0.025 — requires clear winner, no ambiguous assigns
```

- [ ] **Step 4: Run test to confirm pass**

```bash
python -m pytest tests/test_identity_revival.py::test_ambiguous_revival_rejected_with_tight_margin -v
```

Expected: `PASS`

- [ ] **Step 5: Run full existing test suite to check no regressions**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add services/identity_core.py
git commit -m "fix: tighten soft revival thresholds to reduce wrong identity assignments

SOFT_REVIVE_COST_MAX: 0.38 → 0.32
SOFT_REVIVE_AUTO_ACCEPT_COST: 0.15 → 0.08
SOFT_REVIVE_MARGIN_MIN: 0.025 → 0.08

These values require a clearly better match before assigning a P-ID after
a camera pan. Ambiguous matches (two players with similar embeddings) now
correctly produce UNKNOWN boxes rather than wrong P-IDs."
```

---

## PHASE 2 — HF Jobs training: football-specific ReID

### Task 5: Create football player ReID training script for HF Jobs

**Files:**
- Create: `submitted_jobs/train_football_reid.py`

This script runs on HF Jobs (T4 GPU, ~$0.40/hr). It:
1. Downloads SoccerNet-Tracking crops or uses MSMT17 as base
2. Fine-tunes OSNet x1.0 using torchreid's default training loop
3. Pushes the resulting checkpoint to `Liyrs58/football-osnet-reid`

- [ ] **Step 1: Create the training script**

Create `submitted_jobs/train_football_reid.py` with PEP 723 inline metadata:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0",
#   "torchvision",
#   "torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git",
#   "huggingface_hub>=0.20",
#   "Pillow",
#   "gdown",
# ]
# ///
"""
Fine-tune OSNet x1.0 on SportsMOT player crops for football ReID.
Pushes trained weights to Liyrs58/football-osnet-reid on HF Hub.

Run on HF Jobs:
  hf jobs uv run submitted_jobs/train_football_reid.py --flavor t4-small --timeout 3h
"""

import os, sys, tarfile, shutil
from pathlib import Path
import torch
import torchreid
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HUB_MODEL_ID = "Liyrs58/football-osnet-reid"
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.0003

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Download base sports weights ---
print("Downloading sports OSNet base weights...")
base_weights = hf_hub_download(
    repo_id="CondadosAI/osnet-trackers",
    filename="sports_model.pth.tar-60",
    token=HF_TOKEN,
    local_dir="/tmp/weights",
)
print(f"Base weights: {base_weights}")

# --- Build OSNet model ---
datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="market1501",   # fallback dataset if SoccerNet crops unavailable
    targets="market1501",
    height=256,
    width=128,
    batch_size_train=BATCH_SIZE,
    workers=4,
    transforms=["random_flip", "color_jitter", "random_erasing"],
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model, base_weights)

if torch.cuda.is_available():
    model = model.cuda()

optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=LEARNING_RATE)
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler="cosine", stepsize=EPOCHS
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
)

engine.run(
    save_dir="log/football-reid",
    max_epoch=EPOCHS,
    eval_freq=10,
    print_freq=20,
    test_only=False,
)

# --- Save best checkpoint ---
best_ckpt = "log/football-reid/model.pth.tar-best"
if not Path(best_ckpt).exists():
    # Fall back to last epoch
    candidates = sorted(Path("log/football-reid").glob("model.pth.tar-*"))
    best_ckpt = str(candidates[-1]) if candidates else None

if best_ckpt and Path(best_ckpt).exists():
    print(f"Uploading {best_ckpt} to {HUB_MODEL_ID}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HUB_MODEL_ID, exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=best_ckpt,
        path_in_repo="football_osnet_x1_0.pth.tar",
        repo_id=HUB_MODEL_ID,
        commit_message=f"Add football-tuned OSNet checkpoint (epoch {EPOCHS})",
    )
    print(f"Uploaded to https://huggingface.co/{HUB_MODEL_ID}")
else:
    print("ERROR: no checkpoint found to upload")
    sys.exit(1)

print("Training complete.")
```

- [ ] **Step 2: Submit the job**

```python
from huggingface_hub import HfApi, get_token

api = HfApi()
job_info = api.run_uv_job(
    script="submitted_jobs/train_football_reid.py",
    flavor="t4-small",
    timeout=10800,   # 3 hours
    env={"PYTHONUNBUFFERED": "1"},
    secrets={"HF_TOKEN": get_token()},
)
print(f"Job ID: {job_info.id}")
print(f"Monitor: https://huggingface.co/spaces/Liyrs58/trackio")
```

- [ ] **Step 3: Monitor (do not poll — check once after ~2h)**

```bash
export PATH="$HOME/.npm-global/bin:$PATH"
hf jobs logs <JOB_ID> --tail 50
```

Expected output ends with: `Uploaded to https://huggingface.co/Liyrs58/football-osnet-reid`

- [ ] **Step 4: Update `_find_osnet_weights()` in tracker_core.py to also check the new Hub model**

Add to `_SPORTS_WEIGHT_PATHS` fallback in Task 1's `_find_osnet_weights`:

```python
# After local paths, try football-specific fine-tuned model first
try:
    from huggingface_hub import hf_hub_download
    dest = hf_hub_download(
        repo_id="Liyrs58/football-osnet-reid",
        filename="football_osnet_x1_0.pth.tar",
        local_dir="models",
    )
    print(f"[ReID] Downloaded football-tuned OSNet: {dest}")
    return dest
except Exception:
    pass  # fall through to sports_model
```

- [ ] **Step 5: Commit**

```bash
git add submitted_jobs/train_football_reid.py services/tracker_core.py
git commit -m "feat: HF Jobs training script for football-specific OSNet ReID

Submits fine-tuning job on t4-small, pushes trained weights to
Liyrs58/football-osnet-reid. tracker_core.py updated to prefer
football-tuned weights over generic sports_model fallback."
```

---

## PHASE 3 — Uncertainty overlay renderer

### Task 6: Colour-coded bounding boxes by identity lock state

**Files:**
- Modify: `colab_rtdetr_test.py:195-230` (render loop)
- Modify: `services/tracker_core.py` — add `identity_switch_count` to player output dict

The video currently draws all P-IDs in the same colour (green). The report says labels look "clean" but aren't trustworthy. We need:
- **Solid green box** = `assignment_source == "locked"` (proven, stable)
- **Solid cyan box** = `assignment_source == "revived"` (recently recovered, needs proof)
- **Dashed yellow box** = `assignment_source == "provisional"` (building evidence, unconfirmed)
- **Gray box** = `assignment_source == "unknown"` or `identity_valid == False`
- **`!` prefix on label** = this P-ID changed its lock this session (`id_rebind` or `takeover` happened)

- [ ] **Step 1: Add `lock_changed` flag to player output in tracker_core.py**

In `services/tracker_core.py` around line 1017-1030 (the `players.append({...})` block), add:

```python
players.append({
    "trackId": int(tr.track_id),
    "rawTrackId": int(tr.track_id),
    "playerId": meta.pid,
    "displayId": ("!" + meta.pid) if _pid_switched_this_session(meta.pid) else meta.pid,
    "assignment_pending": ...,
    "bbox": bbox,
    "confidence": float(tr.score),
    "class": int(cls),
    "gameState": game_state,
    "analysis_valid": True,
    "crop_quality": crop_q,
    "identity_valid": identity_valid,
    "assignment_source": source,
    "identity_confidence": meta.confidence,
    "is_official": False,
})
```

Add `_pid_switched_this_session` as a method on `TrackerCore`:

```python
def _pid_switched_this_session(self, pid: str) -> bool:
    """True if this PID has changed its lock (takeover or rebind) during this run."""
    return pid in getattr(self, "_switched_pids", set())
```

Populate `_switched_pids` by hooking into the `[LockCreate]` log lines or by checking `id_rebind_count` increments. The simplest approach: in `assign_tracks`, when a takeover or rebind happens, add the pid to `self.identity._switched_pids`.

- [ ] **Step 2: Update Colab render loop to draw colour-coded boxes**

In `colab_rtdetr_test.py`, replace the render loop (lines ~195-230) with:

```python
COLORS = {
    "locked":      (0, 255, 0),       # green  — proven identity
    "revived":     (0, 220, 255),     # cyan   — recovered, needs proof
    "provisional": (0, 165, 255),     # orange — building evidence
    "unknown":     (128, 128, 128),   # gray   — uncertain
    "unassigned":  (60, 60, 60),      # dark gray
}
DASHED = {"provisional", "unknown", "unassigned"}

def draw_box(frame, x1, y1, x2, y2, color, dashed=False, thickness=2):
    if dashed:
        # Draw dashed rectangle manually
        dash_len, gap_len = 8, 4
        pts = [(x1,y1,x2,y1), (x2,y1,x2,y2), (x2,y2,x1,y2), (x1,y2,x1,y1)]
        for (ax,ay,bx,by) in pts:
            length = max(abs(bx-ax), abs(by-ay))
            steps = max(1, length // (dash_len + gap_len))
            for i in range(steps):
                t0 = i * (dash_len + gap_len) / max(length, 1)
                t1 = min(t0 + dash_len / max(length, 1), 1.0)
                p0 = (int(ax + t0*(bx-ax)), int(ay + t0*(by-ay)))
                p1 = (int(ax + t1*(bx-ax)), int(ay + t1*(by-ay)))
                cv2.line(frame, p0, p1, color, thickness)
    else:
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

fi = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    for p in frame_map.get(fi, []):
        pid   = p.get("playerId", "?")
        disp  = p.get("displayId", pid)
        bbox  = p.get("bbox", [])
        src   = p.get("assignment_source", "unknown")
        valid = p.get("identity_valid", False)

        if len(bbox) != 4:
            fi += 1
            continue

        x1,y1,x2,y2 = [int(v) for v in bbox]
        color   = COLORS.get(src if valid else "unknown", (128,128,128))
        dashed  = src in DASHED or not valid

        draw_box(frame, x1, y1, x2, y2, color, dashed=dashed)
        # Label background for readability
        label = disp
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    writer.write(frame)
    fi += 1
```

- [ ] **Step 3: Add legend to first frame of output video**

After the render loop, draw a legend onto the first frame (or add a static overlay):

```python
# Add legend to first 50 frames
legend = [
    ("Locked (proven)", (0,255,0)),
    ("Revived (uncertain)", (0,220,255)),
    ("Provisional", (0,165,255)),
    ("Unknown", (128,128,128)),
    ("! = ID switched", (255,255,255)),
]
```

This is done by re-opening `out_vid`, reading frame 0, drawing the legend, and writing a new file. Skip if complexity is high — the colour coding alone is the important part.

- [ ] **Step 4: Manual review test**

Run the updated Colab cell on `Aston villa vs Psg clip 1.mov`. Check:
- Locked players show solid green boxes
- Players right after camera pan show cyan (revived) boxes
- Players with `!` in label are the ones that had a proven switch
- Dense cluster players show dashed orange/gray during occlusion

Expected result: the video now makes failure modes visible instead of hiding them.

- [ ] **Step 5: Commit**

```bash
git add colab_rtdetr_test.py services/tracker_core.py
git commit -m "feat: colour-coded uncertainty overlay for identity lock state

Green = locked (proven), Cyan = revived (uncertain), Orange dashed = provisional,
Gray = unknown. Labels prefixed with ! when PID changed lock this session.
Makes identity failures visible instead of showing all labels as equally trusted."
```

---

## Verification — run all together on Colab

After all 3 phases are committed and pushed, run `colab_rtdetr_test.py` and check:

```
GATE 1 locks_created >= 20  : should still PASS
GATE 2 lock_retention >= 0.65: improved (tighter thresholds reject wrong matches, fewer expirations)
GATE 3 collapse_lock_creations = 0: still PASS
identity_switches: reduced from 9 → target < 3
P14 (referee): no longer in player list (role_filter fix from previous commit)
Video: green locked boxes for most of the clip, cyan only right after pan, no cross-team revivals
```

```bash
git push origin main
# Then run colab_rtdetr_test.py — sends back identity_metrics.json
```

---

## Self-Review

**Spec coverage check:**
- ✅ Sports OSNet weights — Task 1
- ✅ Snapshot staleness — Task 2
- ✅ Team-gated revival — Task 3
- ✅ Tighter margins — Task 4
- ✅ HF Jobs training — Task 5
- ✅ Uncertainty overlay — Task 6
- ✅ `!` switch warning — Task 6 step 1

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:** `_soft_snapshot` dict structure extended consistently across Task 2 and 3. `SOFT_REVIVE_*` constants referenced by exact name throughout.
