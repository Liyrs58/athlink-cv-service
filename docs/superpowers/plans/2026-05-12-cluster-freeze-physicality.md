# Cluster Freeze + Physicality Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Block wrong P-ID assignments inside dense player clusters and reject physically-impossible relinks before they are accepted anywhere in the identity pipeline.

**Architecture:** Two orthogonal guards added to `services/identity_core.py`. Guard 1 (`CongestionDetector`): per-frame IoU + center-distance scan that tags each TID as "in cluster" or not; any tagged TID is frozen from new lock creation, takeover, or revival — existing same-tid locks are preserved. Guard 2 (`PhysicalityValidator`): a standalone function called at every point a PID would be granted (Hungarian relink, shadow relink, scene/soft/force revival, patch apply); it rejects impossible pixel jumps, team flips, official-PID assignment, and double occupancy.

**Tech Stack:** Python 3.11, NumPy, existing `identity_core.py` / `identity_locks.py` / `identity_patch_service.py`. No new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `services/identity_core.py` | Add `CongestionDetector` class; add `PhysicalityValidator` function; wire both into all assignment/revival paths; add metrics |
| `services/identity_patch_service.py` | Extend `PatchValidator` to call `PhysicalityValidator` for official-PID and team-flip checks |
| `tests/test_identity_core_regression.py` | 6 new regression tests (cluster freeze scenarios + physicality rejects) |

---

## Task 1: `CongestionDetector` — per-frame cluster detection

**Files:**
- Modify: `services/identity_core.py` — add `CongestionDetector` class and env constants

### Design
`CongestionDetector` takes a list of `(tid, bbox)` pairs per frame and returns the set of TIDs that are inside a dense cluster.

A TID is "in cluster" when ANY of these is true:
- Its bbox overlaps with ≥ 2 other boxes with IoU ≥ `iou_threshold` (default 0.10)
- Its center is within `radius_px` (default 80) of ≥ 2 other player centers

Env vars (read at module load):
- `ATHLINK_CLUSTER_IOU_THRESHOLD` = 0.10
- `ATHLINK_CLUSTER_RADIUS_PX` = 80
- `ATHLINK_CLUSTER_MIN_NEIGHBORS` = 2

- [ ] **Step 1: Write failing test**

Append to `tests/test_identity_core_regression.py`:

```python
class TestCongestionDetector:
    """CongestionDetector tags TIDs that are inside dense clusters."""

    def _make_bbox(self, cx, cy, w=50, h=120):
        return [cx - w//2, cy - h//2, cx + w//2, cy + h//2]

    def test_overlapping_group_tagged(self):
        """Three boxes heavily overlapping → all 3 tagged as in-cluster."""
        from services.identity_core import CongestionDetector
        det = CongestionDetector(iou_threshold=0.10, radius_px=80, min_neighbors=2)

        # Three boxes stacked nearly identically
        tid_bboxes = [
            (1, self._make_bbox(300, 400)),
            (2, self._make_bbox(310, 405)),
            (3, self._make_bbox(305, 402)),
        ]
        in_cluster = det.detect(tid_bboxes)
        assert 1 in in_cluster
        assert 2 in in_cluster
        assert 3 in in_cluster

    def test_isolated_player_not_tagged(self):
        """A player far from all others is not tagged."""
        from services.identity_core import CongestionDetector
        det = CongestionDetector(iou_threshold=0.10, radius_px=80, min_neighbors=2)

        tid_bboxes = [
            (1, self._make_bbox(300, 400)),
            (2, self._make_bbox(310, 405)),
            (3, self._make_bbox(305, 402)),
            (7, self._make_bbox(900, 200)),   # far away
        ]
        in_cluster = det.detect(tid_bboxes)
        assert 7 not in in_cluster, "isolated tid=7 must not be tagged"

    def test_pair_below_threshold_not_tagged(self):
        """Only 2 neighbors (< min_neighbors=2 extra) → not in cluster."""
        from services.identity_core import CongestionDetector
        det = CongestionDetector(iou_threshold=0.10, radius_px=80, min_neighbors=3)

        # Only 2 overlapping boxes, min_neighbors=3 → not enough
        tid_bboxes = [
            (1, self._make_bbox(300, 400)),
            (2, self._make_bbox(305, 402)),
        ]
        in_cluster = det.detect(tid_bboxes)
        assert len(in_cluster) == 0, "pair with min_neighbors=3 should not trigger"

    def test_empty_input(self):
        """Empty input returns empty set."""
        from services.identity_core import CongestionDetector
        det = CongestionDetector()
        assert det.detect([]) == set()
```

- [ ] **Step 2: Run test to verify ImportError**

```bash
cd /Users/rudra/athlink-cv-service/athlink-cv-service
python -m pytest tests/test_identity_core_regression.py::TestCongestionDetector -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'CongestionDetector'`

- [ ] **Step 3: Add `CongestionDetector` to `services/identity_core.py`**

Insert after the `ShadowBuffer` class and before `def _edge_for_center` (around line 200):

```python
# ── Congestion detector configuration ────────────────────────────────
_CLUSTER_IOU_THRESHOLD = float(_os.environ.get("ATHLINK_CLUSTER_IOU_THRESHOLD", "0.10"))
_CLUSTER_RADIUS_PX     = float(_os.environ.get("ATHLINK_CLUSTER_RADIUS_PX", "80"))
_CLUSTER_MIN_NEIGHBORS = int(_os.environ.get("ATHLINK_CLUSTER_MIN_NEIGHBORS", "2"))


def _bbox_iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CongestionDetector:
    """
    Per-frame cluster detection using bbox IoU and center-distance.

    A TID is tagged "in cluster" when it has >= min_neighbors other players
    within radius_px OR overlapping with IoU >= iou_threshold.
    """

    def __init__(
        self,
        iou_threshold: float = _CLUSTER_IOU_THRESHOLD,
        radius_px: float = _CLUSTER_RADIUS_PX,
        min_neighbors: int = _CLUSTER_MIN_NEIGHBORS,
    ):
        self._iou_thresh = iou_threshold
        self._radius = radius_px
        self._min_nb = min_neighbors

    def detect(self, tid_bboxes: List[Tuple[int, list]]) -> Set[int]:
        """
        Args:
            tid_bboxes: list of (tid, [x1,y1,x2,y2])
        Returns:
            set of TIDs that are inside a dense cluster
        """
        if len(tid_bboxes) < 2:
            return set()

        tids = [t for t, _ in tid_bboxes]
        boxes = [b for _, b in tid_bboxes]
        centers = [
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            for b in boxes
        ]

        neighbor_counts = [0] * len(tids)
        n = len(tids)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cx_diff = centers[i][0] - centers[j][0]
                cy_diff = centers[i][1] - centers[j][1]
                dist = (cx_diff ** 2 + cy_diff ** 2) ** 0.5
                iou = _bbox_iou(boxes[i], boxes[j])
                if dist < self._radius or iou >= self._iou_thresh:
                    neighbor_counts[i] += 1

        return {
            tids[i]
            for i in range(n)
            if neighbor_counts[i] >= self._min_nb
        }
```

- [ ] **Step 4: Run tests — expect 4/4 pass**

```bash
python -m pytest tests/test_identity_core_regression.py::TestCongestionDetector -v 2>&1 | tail -15
```

- [ ] **Step 5: Add `CongestionDetector` instance and `cluster_freeze_blocks` counter to `IdentityCore.__init__`**

In `__init__`, alongside other counters:

```python
        self.congestion_detector = CongestionDetector()
        self.cluster_freeze_blocks: int = 0
```

- [ ] **Step 6: Run broader regression to check no breakage**

```bash
python -m pytest tests/test_identity_core_regression.py tests/test_identity_stability.py -q 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add services/identity_core.py tests/test_identity_core_regression.py
git commit -m "feat: CongestionDetector — per-frame bbox IoU + center-distance cluster detection"
```

---

## Task 2: Wire cluster freeze into `assign_tracks`

**Files:**
- Modify: `services/identity_core.py` — `assign_tracks` signature + body (locked-pairs loop, Hungarian lock promotion path, revival paths)

### Design
`assign_tracks()` gains a new optional parameter `tid_bboxes: Optional[Dict[int, list]] = None` (maps tid → [x1,y1,x2,y2]).

At the start of the method body (after official gate, before locked-pairs loop), compute `_cluster_tids` once:

```python
_cluster_tids: Set[int] = set()
if tid_bboxes:
    _cluster_tids = self.congestion_detector.detect(
        [(tid, bbox) for tid, bbox in tid_bboxes.items()]
    )
```

Then at every point a **new** lock/revival/takeover would be created for a TID in `_cluster_tids`, block it and increment `cluster_freeze_blocks`. Existing locked same-tid pairs ALWAYS pass through regardless of cluster state.

- [ ] **Step 1: Write failing regression tests**

Append to `tests/test_identity_core_regression.py`:

```python
class TestClusterFreeze:
    """Cluster freeze blocks new locks/revivals inside dense groups."""

    def _make_track(self, tid):
        class T:
            track_id = tid
            time_since_update = 0
        return T()

    def _make_bbox(self, cx, cy, w=50, h=120):
        return [cx - w//2, cy - h//2, cx + w//2, cy + h//2]

    def test_new_lock_blocked_in_cluster(self):
        """New lock creation must be blocked for TID inside a dense cluster."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()
        identity.congestion_detector._min_nb = 2

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Seed slot with matching embedding so cost is low
        slot = identity.slots[0]
        slot.embedding = emb.copy()
        slot.state = "active"
        slot.pending_streak = 10  # ready to lock
        slot.pending_tid = 7
        slot.pending_seen_seq = 0

        # Three boxes in a tight cluster
        tight_bboxes = {
            7:  self._make_bbox(300, 400),
            8:  self._make_bbox(305, 402),
            9:  self._make_bbox(310, 405),
        }

        identity.begin_frame(1, present_tids={7, 8, 9})
        identity.identity_frame_seq = 1  # sync seq so streak counts
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (300.0, 400.0)},
            allow_new_assignments=True,
            tid_bboxes=tight_bboxes,
        )
        identity.end_frame()

        # Lock must NOT be created
        lk = identity.locks.get_lock(7)
        assert lk is None, "New lock must be blocked inside dense cluster"
        assert identity.cluster_freeze_blocks >= 1

    def test_existing_lock_preserved_in_cluster(self):
        """Existing stable lock survives cluster freeze — same-tid pair always passes."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Pre-lock tid=7 → P7
        identity.locks.try_create_lock(7, "P7", "hungarian", frame_id=0)

        tight_bboxes = {
            7:  [275, 340, 325, 460],
            8:  [280, 342, 330, 462],
            9:  [285, 345, 335, 465],
        }

        identity.begin_frame(5, present_tids={7, 8, 9})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (300.0, 400.0)},
            allow_new_assignments=True,
            tid_bboxes=tight_bboxes,
        )
        identity.end_frame()

        assert track_to_pid.get(7) == "P7", \
            "Existing stable lock P7 must pass through cluster freeze"

    def test_p6_cannot_revive_as_winger_through_cluster(self):
        """Regression: P6 centre-back must not revive as winger role through a cluster."""
        import numpy as np
        from services.identity_core import IdentityCore, ShadowEntry

        identity = IdentityCore()
        identity.shadow_buffer._require_edge = False
        identity.shadow_buffer._max_cost = 0.99

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Plant shadow for P6 (centre-back position ~centre-left)
        entry = ShadowEntry(
            pid="P6", last_tid=6, last_seen_frame=5,
            last_bbox=None, last_center=(400.0, 300.0),
            last_embedding=emb.copy(), team_id=0,
            exit_edge="interior", stable_count=20,
        )
        identity.shadow_buffer.add(entry, added_frame=5)

        # Seed slot P6 with embedding
        slot = identity._slot_by_pid("P6")
        slot.embedding = emb.copy()
        slot.state = "active"
        slot.pending_streak = 10
        slot.pending_tid = 21  # different tid — potential winger
        slot.pending_seen_seq = 0

        # Tight cluster at winger position
        tight_bboxes = {
            20: [830, 140, 880, 260],
            21: [835, 143, 885, 263],
            22: [840, 148, 890, 268],
        }

        identity.begin_frame(20, present_tids={20, 21, 22})
        identity.identity_frame_seq = 1
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(21)],
            embeddings={21: emb},
            positions={21: (855.0, 200.0)},
            allow_new_assignments=True,
            tid_bboxes=tight_bboxes,
        )
        identity.end_frame()

        assert track_to_pid.get(21) != "P6", \
            "P6 centre-back must not revive as winger through cluster"
        assert identity.cluster_freeze_blocks >= 1

    def test_p7_striker_exit_not_inherited_during_congestion(self):
        """Regression: P7 striker exits; different tid in cluster must not inherit P7."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()

        emb7 = np.random.randn(512).astype(np.float32)
        emb7 /= np.linalg.norm(emb7)

        # P7 was locked to tid=7 but tid=7 is absent now
        # tid=15 (different player) is nearby in cluster trying to inherit P7
        slot7 = identity._slot_by_pid("P7")
        slot7.embedding = emb7.copy()
        slot7.state = "active"
        slot7.pending_streak = 10
        slot7.pending_tid = 15
        slot7.pending_seen_seq = 0

        tight_bboxes = {
            15: [490, 380, 540, 500],
            16: [495, 382, 545, 502],
            17: [500, 385, 550, 505],
        }

        identity.begin_frame(30, present_tids={15, 16, 17})
        identity.identity_frame_seq = 1
        emb_new = emb7.copy()
        emb_new += 0.3 * np.random.randn(512).astype(np.float32)
        emb_new /= np.linalg.norm(emb_new)

        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(15)],
            embeddings={15: emb_new},
            positions={15: (515.0, 440.0)},
            allow_new_assignments=True,
            tid_bboxes=tight_bboxes,
        )
        identity.end_frame()

        # tid=15 should not get P7 inside the cluster
        assert track_to_pid.get(15) != "P7", \
            "P7 must not be inherited by tid=15 in dense cluster"
        assert identity.cluster_freeze_blocks >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_identity_core_regression.py::TestClusterFreeze -v 2>&1 | tail -15
```

Expected: `TypeError: assign_tracks() got unexpected keyword argument 'tid_bboxes'`

- [ ] **Step 3: Add `tid_bboxes` parameter to `assign_tracks` signature**

In `services/identity_core.py`, find `assign_tracks` (line ~601). Add `tid_bboxes: Optional[Dict[int, list]] = None` after `official_tids`:

```python
    def assign_tracks(
        self,
        tracks: Sequence[object],
        embeddings: Dict[int, np.ndarray],
        positions: Dict[int, Tuple[float, float]],
        memory_ok_tids: Optional[set] = None,
        allow_new_assignments: bool = True,
        camera_motion: Optional[Dict] = None,
        official_tids: Optional[Set[int]] = None,
        tid_bboxes: Optional[Dict[int, list]] = None,   # ← ADD
    ) -> Tuple[Dict[int, str], Dict[int, AssignmentMeta]]:
```

- [ ] **Step 4: Compute `_cluster_tids` at start of assign_tracks body**

Insert immediately after the official gate block (after `tracks = [tr for tr in tracks if ...]`):

```python
        # ── Congestion / cluster freeze ────────────────────────────────────
        _cluster_tids: Set[int] = set()
        if tid_bboxes:
            _cluster_tids = self.congestion_detector.detect(
                [(tid, bbox) for tid, bbox in tid_bboxes.items()]
            )
            if _cluster_tids and self.frame_id % self.debug_every == 0:
                print(f"[ClusterDetect] frame={self.frame_id} cluster_tids={sorted(_cluster_tids)}")
```

- [ ] **Step 5: Block new lock promotion inside cluster**

In the Hungarian path, find the block starting at `if lock_ready and cost_ok:` (around line 882). Insert a cluster check BEFORE the pan-gate check:

```python
                if lock_ready and cost_ok:
                    # Cluster freeze: block new lock creation inside dense cluster
                    if tid in _cluster_tids:
                        self.cluster_freeze_blocks += 1
                        print(f"[ClusterFreeze] frame={self.frame_id} tid={tid} pid={slot.pid} "
                              f"reason=dense_cluster lock_blocked")
                        # Fall through to normal provisional assignment (no lock)
                    else:
                        # [existing pan-gate + try_create_lock block here — leave untouched]
```

IMPORTANT: The cluster freeze only blocks lock *promotion*. The TID still gets its provisional assignment so existing pending tracking continues. Do NOT skip the rest of the assignment for this TID.

To implement this without moving large blocks of code: wrap the existing `if lock_ready and cost_ok:` body in a new `if tid not in _cluster_tids:` guard, keeping the outer `if lock_ready and cost_ok:` structure:

```python
                if lock_ready and cost_ok:
                    if tid in _cluster_tids:
                        self.cluster_freeze_blocks += 1
                        print(f"[ClusterFreeze] frame={self.frame_id} tid={tid} pid={slot.pid} "
                              f"reason=dense_cluster lock_blocked")
                    else:
                        # ── existing pan-gate + lock creation block (unchanged) ──
                        pan_restricted = self._camera_motion_restricted()
                        # ... rest of existing lock-promotion code unchanged ...
```

Read the current code from ~line 882 to ~line 930 before making this edit to get the exact indentation right.

- [ ] **Step 6: Block cluster TIDs in revival paths**

In `revive_cost_matrix` (line ~1068), after the shadow gate `continue`, add a cluster freeze check BEFORE `revived[tid] = pid`:

```python
            # ── Cluster freeze: block revival inside dense cluster ──
            if tid in _cluster_tids:
                self.cluster_freeze_blocks += 1
                print(f"[ClusterFreeze] frame={self.frame_id} tid={tid} pid={pid} "
                      f"reason=dense_cluster revival_blocked")
                meta[tid] = AssignmentMeta(
                    pid=None, source="cluster_frozen",
                    identity_state=IdentityState.UNKNOWN,
                    confidence=0.0, identity_valid=False,
                )
                continue
```

BUT: `revive_cost_matrix` does not receive `_cluster_tids` currently. It needs to receive it. Pass `_cluster_tids` through: add `cluster_tids: Optional[Set[int]] = None` parameter to `revive_cost_matrix`, `force_commit_remaining_scene_slots`, and `revive_from_soft_snapshot`. The callers in `tracker_core.py` don't pass it yet — that's fine; they'll get `None` which means no freeze (safe default).

Inside each revival method, at the acceptance point:
```python
            if cluster_tids and tid in cluster_tids:
                self.cluster_freeze_blocks += 1
                print(f"[ClusterFreeze] frame={self.frame_id} tid={tid} pid={pid} reason=dense_cluster")
                meta[tid] = AssignmentMeta(pid=None, source="cluster_frozen",
                    identity_state=IdentityState.UNKNOWN, confidence=0.0, identity_valid=False)
                continue
```

- [ ] **Step 7: Export `cluster_freeze_blocks` in `end_run_summary`**

Find `end_run_summary` and add:
```python
        print(f"  cluster_freeze_blocks        = {self.cluster_freeze_blocks}")
```
And in the returned dict:
```python
        "cluster_freeze_blocks": self.cluster_freeze_blocks,
```

- [ ] **Step 8: Run cluster freeze tests**

```bash
python -m pytest tests/test_identity_core_regression.py::TestClusterFreeze -v 2>&1 | tail -20
```

Expected: 4/4 PASS.

- [ ] **Step 9: Run full regression**

```bash
python -m pytest tests/test_identity_core_regression.py tests/test_identity_stability.py -q 2>&1 | tail -10
```

- [ ] **Step 10: Commit**

```bash
git add services/identity_core.py tests/test_identity_core_regression.py
git commit -m "feat: cluster freeze — block new locks/revivals inside dense player clusters"
```

---

## Task 3: `PhysicalityValidator` — relink/revival pre-check

**Files:**
- Modify: `services/identity_core.py` — add `PhysicalityValidator` + `_PhysRejectReason` enum; wire into all relink/revival acceptance points
- Modify: `services/identity_patch_service.py` — call `PhysicalityValidator` for official-PID + team-flip checks

### Design
A standalone function `check_physicality(...)` (not a class; stateless) validates a proposed PID assignment. It is called at every point a PID would be granted *before* the assignment is accepted.

```python
def check_physicality(
    pid: str,
    candidate_center: Tuple[float, float],
    candidate_team: Optional[int],
    current_frame: int,
    last_center: Optional[Tuple[float, float]],
    last_frame: Optional[int],
    last_team: Optional[int],
    is_official: bool = False,
    all_current_pids: Optional[Dict[str, Tuple[float, float]]] = None,
    fps: float = 30.0,
    frame_stride: int = 1,
    max_speed_px_per_sec: float = 650.0,
    max_relink_pixel_jump: float = 450.0,
    min_patch_distance_px: float = 24.0,
    reject_official_pid: bool = True,
) -> Tuple[bool, str, str]:
    """
    Returns (ok, reject_reason_code, detail_message).
    ok=True means assignment is physically plausible.
    """
```

Reject codes (str constants):
- `"OFFICIAL_PID"` — is_official=True + player PID
- `"IMPOSSIBLE_SPEED"` — implied speed > max_speed_px_per_sec
- `"IMPOSSIBLE_PIXEL_JUMP"` — relink pixel jump > max_relink_pixel_jump (for gaps > 1 frame)
- `"TEAM_FLIP"` — candidate_team != last_team when both are known
- `"DOUBLE_OCCUPANCY"` — another PID already at same location this frame

Env vars:
- `ATHLINK_MAX_PIXEL_JUMP_PER_SEC` = 650
- `ATHLINK_MAX_RELINK_PIXEL_JUMP` = 450
- `ATHLINK_MIN_PATCH_DISTANCE_PX` = 24
- `ATHLINK_REJECT_OFFICIAL_PID` = 1

- [ ] **Step 1: Write failing tests**

Append to `tests/test_identity_core_regression.py`:

```python
class TestPhysicalityValidator:
    """check_physicality rejects impossible assignments."""

    def test_official_pid_rejected(self):
        """is_official=True + player pid → OFFICIAL_PID rejection."""
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P13",
            candidate_center=(800.0, 400.0),
            candidate_team=None,
            current_frame=10,
            last_center=None,
            last_frame=None,
            last_team=None,
            is_official=True,
        )
        assert not ok
        assert code == "OFFICIAL_PID"

    def test_impossible_pixel_jump_rejected(self):
        """Teleport across screen in 1 frame → IMPOSSIBLE_SPEED rejection."""
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P6",
            candidate_center=(1800.0, 400.0),
            candidate_team=0,
            current_frame=11,
            last_center=(100.0, 400.0),     # 1700px jump in 1 frame
            last_frame=10,
            last_team=0,
            fps=30.0,
            frame_stride=1,
            max_speed_px_per_sec=650.0,
        )
        assert not ok, f"1700px jump in 1 frame must be rejected; got ok={ok}"
        assert code in ("IMPOSSIBLE_SPEED", "IMPOSSIBLE_PIXEL_JUMP")

    def test_team_flip_rejected(self):
        """Candidate team != last team → TEAM_FLIP rejection."""
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P10",
            candidate_center=(500.0, 300.0),
            candidate_team=1,           # changed from team 0
            current_frame=50,
            last_center=(510.0, 305.0),
            last_frame=49,
            last_team=0,
        )
        assert not ok, "Team flip must be rejected"
        assert code == "TEAM_FLIP"

    def test_plausible_assignment_accepted(self):
        """Normal motion, same team → accepted."""
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P5",
            candidate_center=(310.0, 405.0),
            candidate_team=0,
            current_frame=11,
            last_center=(300.0, 400.0),
            last_frame=10,
            last_team=0,
            fps=30.0,
            frame_stride=1,
        )
        assert ok, f"Plausible assignment must pass; code={code} detail={detail}"

    def test_double_occupancy_rejected(self):
        """Same PID already assigned elsewhere this frame → DOUBLE_OCCUPANCY."""
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P3",
            candidate_center=(500.0, 300.0),
            candidate_team=0,
            current_frame=20,
            last_center=None,
            last_frame=None,
            last_team=None,
            all_current_pids={"P3": (500.0, 300.0)},  # P3 already here
        )
        assert not ok
        assert code == "DOUBLE_OCCUPANCY"

    def test_p10_team_flip_in_cluster_rejected(self):
        """Regression: P10 PSG DM cannot flip to Aston Villa AM."""
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P10",
            candidate_center=(600.0, 350.0),
            candidate_team=1,   # Villa
            current_frame=90,
            last_center=(590.0, 340.0),
            last_frame=85,
            last_team=0,        # PSG
        )
        assert not ok
        assert code == "TEAM_FLIP"

    def test_p11_impossible_winger_jump_rejected(self):
        """Regression: P11 PSG DM cannot briefly switch to far-side winger (spatial jump)."""
        from services.identity_core import check_physicality
        # DM at centre (960, 540), winger at edge (100, 300) — 894px in 1 frame
        ok, code, detail = check_physicality(
            pid="P11",
            candidate_center=(100.0, 300.0),
            candidate_team=0,
            current_frame=341,
            last_center=(960.0, 540.0),
            last_frame=340,
            fps=30.0,
            frame_stride=1,
            max_speed_px_per_sec=650.0,
            last_team=0,
        )
        assert not ok
        assert code in ("IMPOSSIBLE_SPEED", "IMPOSSIBLE_PIXEL_JUMP")
```

- [ ] **Step 2: Run tests to verify ImportError**

```bash
python -m pytest tests/test_identity_core_regression.py::TestPhysicalityValidator -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'check_physicality'`

- [ ] **Step 3: Add `check_physicality` to `services/identity_core.py`**

Insert after `CongestionDetector` class (before `_edge_for_center`):

```python
# ── Physicality validator configuration ──────────────────────────────
_MAX_PIXEL_JUMP_PER_SEC  = float(_os.environ.get("ATHLINK_MAX_PIXEL_JUMP_PER_SEC", "650"))
_MAX_RELINK_PIXEL_JUMP   = float(_os.environ.get("ATHLINK_MAX_RELINK_PIXEL_JUMP", "450"))
_MIN_PATCH_DISTANCE_PX   = float(_os.environ.get("ATHLINK_MIN_PATCH_DISTANCE_PX", "24"))
_REJECT_OFFICIAL_PID     = int(_os.environ.get("ATHLINK_REJECT_OFFICIAL_PID", "1")) == 1


def check_physicality(
    pid: str,
    candidate_center: Tuple[float, float],
    candidate_team: Optional[int],
    current_frame: int,
    last_center: Optional[Tuple[float, float]],
    last_frame: Optional[int],
    last_team: Optional[int],
    is_official: bool = False,
    all_current_pids: Optional[Dict[str, Tuple[float, float]]] = None,
    fps: float = 30.0,
    frame_stride: int = 1,
    max_speed_px_per_sec: float = _MAX_PIXEL_JUMP_PER_SEC,
    max_relink_pixel_jump: float = _MAX_RELINK_PIXEL_JUMP,
    min_patch_distance_px: float = _MIN_PATCH_DISTANCE_PX,
    reject_official_pid: bool = _REJECT_OFFICIAL_PID,
) -> Tuple[bool, str, str]:
    """
    Validate a proposed PID assignment for physical plausibility.

    Returns (ok, reject_reason_code, detail_message).
    ok=True means the assignment is physically plausible.
    Reject codes: OFFICIAL_PID | IMPOSSIBLE_SPEED | IMPOSSIBLE_PIXEL_JUMP
                  | TEAM_FLIP | DOUBLE_OCCUPANCY
    """
    # 1. Official/referee must not receive player PID
    if reject_official_pid and is_official:
        return False, "OFFICIAL_PID", f"pid={pid} is_official=True"

    # 2. Team flip check
    if last_team is not None and candidate_team is not None:
        if last_team != candidate_team:
            return False, "TEAM_FLIP", (
                f"pid={pid} last_team={last_team} candidate_team={candidate_team}"
            )

    # 3. Speed / pixel jump check
    if last_center is not None and last_frame is not None:
        cx, cy = candidate_center
        lx, ly = last_center
        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
        frame_gap = max(1, current_frame - last_frame)
        elapsed_sec = (frame_gap * frame_stride) / max(fps, 1.0)

        if elapsed_sec > 0:
            speed_px_per_sec = dist / elapsed_sec
            if speed_px_per_sec > max_speed_px_per_sec:
                return False, "IMPOSSIBLE_SPEED", (
                    f"pid={pid} dist={dist:.1f}px gap={frame_gap}f "
                    f"speed={speed_px_per_sec:.1f}px/s > max={max_speed_px_per_sec}"
                )

        # Relink-specific: large absolute jump even if speed threshold would pass
        if frame_gap > 1 and dist > max_relink_pixel_jump:
            return False, "IMPOSSIBLE_PIXEL_JUMP", (
                f"pid={pid} dist={dist:.1f}px > max_relink={max_relink_pixel_jump} gap={frame_gap}f"
            )

    # 4. Double occupancy — same PID already assigned this frame
    if all_current_pids and pid in all_current_pids:
        existing_center = all_current_pids[pid]
        ex, ey = existing_center
        cx, cy = candidate_center
        dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
        if dist < min_patch_distance_px:
            return False, "DOUBLE_OCCUPANCY", (
                f"pid={pid} already at ({ex:.0f},{ey:.0f}) new=({cx:.0f},{cy:.0f}) dist={dist:.1f}"
            )

    return True, "", ""
```

- [ ] **Step 4: Run physicality tests — expect 7/7 pass**

```bash
python -m pytest tests/test_identity_core_regression.py::TestPhysicalityValidator -v 2>&1 | tail -15
```

- [ ] **Step 5: Add `physicality_rejects` counter and reason breakdown to `IdentityCore.__init__`**

```python
        self.physicality_rejects: int = 0
        self.physicality_reject_reasons: Dict[str, int] = {}
```

- [ ] **Step 6: Commit**

```bash
git add services/identity_core.py tests/test_identity_core_regression.py
git commit -m "feat: check_physicality — speed/team-flip/official/double-occupancy validator"
```

---

## Task 4: Wire `check_physicality` into all identity assignment paths

**Files:**
- Modify: `services/identity_core.py` — call `check_physicality` at Hungarian relink, shadow relink, scene revival, soft revival, force-commit revival

### Helper to call at each acceptance point

Add this small helper inside `IdentityCore` (private method):

```python
    def _physcheck(
        self,
        pid: str,
        tid: int,
        candidate_center: Optional[Tuple[float, float]],
        candidate_team: Optional[int],
        last_center: Optional[Tuple[float, float]],
        last_frame: Optional[int],
        last_team: Optional[int],
        is_official: bool = False,
        all_current_pids: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[bool, str]:
        """Wrapper that calls check_physicality and updates self metrics."""
        ok, code, detail = check_physicality(
            pid=pid,
            candidate_center=candidate_center or (0.0, 0.0),
            candidate_team=candidate_team,
            current_frame=self.frame_id,
            last_center=last_center,
            last_frame=last_frame,
            last_team=last_team,
            is_official=is_official,
            all_current_pids=all_current_pids,
        )
        if not ok:
            self.physicality_rejects += 1
            self.physicality_reject_reasons[code] = (
                self.physicality_reject_reasons.get(code, 0) + 1
            )
            print(f"[PhysReject] frame={self.frame_id} tid={tid} pid={pid} "
                  f"code={code} detail={detail}")
        return ok, code
```

- [ ] **Step 1: Write integration test for wired physicality rejection**

Append to `tests/test_identity_core_regression.py`:

```python
class TestPhysicalityWired:
    """check_physicality is wired into assign_tracks revival paths."""

    def _make_track(self, tid):
        class T:
            track_id = tid
            time_since_update = 0
        return T()

    def test_impossible_jump_blocks_hungarian_relink(self):
        """A relink with 1700px jump in 1 frame must be rejected by physicality check."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Lock tid=7 → P7 at left side of frame
        identity.locks.try_create_lock(7, "P7", "hungarian", frame_id=0)
        slot = identity._slot_by_pid("P7")
        slot.embedding = emb.copy()
        slot.last_position = (100.0, 400.0)   # was on left
        slot.last_seen_frame = 10

        # Now tid=7 re-appears at far right — teleport
        identity.begin_frame(11, present_tids={7})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (1820.0, 400.0)},   # 1720px jump in 1 frame
            allow_new_assignments=True,
        )
        identity.end_frame()

        assert identity.physicality_rejects >= 1, \
            "Physicality reject counter must increment"
        assert "IMPOSSIBLE_SPEED" in identity.physicality_reject_reasons or \
               "IMPOSSIBLE_PIXEL_JUMP" in identity.physicality_reject_reasons
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_identity_core_regression.py::TestPhysicalityWired -v 2>&1 | tail -15
```

Expected: FAIL (physicality_rejects == 0 because not wired yet)

- [ ] **Step 3: Add `_physcheck` helper method to `IdentityCore`**

Insert as a private method near `_relink_absent_existing_lock` (around line 360):

```python
    def _physcheck(
        self,
        pid: str,
        tid: int,
        candidate_center: Optional[Tuple[float, float]],
        candidate_team: Optional[int],
        last_center: Optional[Tuple[float, float]],
        last_frame: Optional[int],
        last_team: Optional[int],
        is_official: bool = False,
        all_current_pids: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[bool, str]:
        ok, code, detail = check_physicality(
            pid=pid,
            candidate_center=candidate_center or (0.0, 0.0),
            candidate_team=candidate_team,
            current_frame=self.frame_id,
            last_center=last_center,
            last_frame=last_frame,
            last_team=last_team,
            is_official=is_official,
            all_current_pids=all_current_pids,
        )
        if not ok:
            self.physicality_rejects += 1
            self.physicality_reject_reasons[code] = (
                self.physicality_reject_reasons.get(code, 0) + 1
            )
            print(f"[PhysReject] frame={self.frame_id} tid={tid} pid={pid} "
                  f"code={code} detail={detail}")
        return ok, code
```

- [ ] **Step 4: Wire `_physcheck` into the locked-pairs pass-through**

In `assign_tracks`, find the locked-pairs loop (line ~620 area). Right after getting `pid = lk.pid` and `slot`, and **before** emitting `track_to_pid[tid] = pid`, insert:

```python
            # Physicality check on locked pair: catch teleports/team-flips before emitting
            _pos_now = positions.get(tid)
            _team_now = getattr(slot, 'team_id', None)
            _phys_ok, _phys_code = self._physcheck(
                pid=pid, tid=tid,
                candidate_center=_pos_now,
                candidate_team=_team_now,
                last_center=slot.last_position,
                last_frame=slot.last_seen_frame,
                last_team=_team_now,  # same — team flip caught at revival not here
            )
            if not _phys_ok:
                # Don't emit PID but don't release lock either
                meta_map[tid] = AssignmentMeta(
                    pid=None, source="physicality_rejected",
                    identity_state=IdentityState.UNKNOWN,
                    confidence=0.0, identity_valid=False,
                )
                unlocked_tracks.append(tr)
                continue
```

- [ ] **Step 5: Wire into the `_relink_absent_existing_lock` path**

Find `_relink_absent_existing_lock` (around line 360). Before returning the result, add a physicality check using the slot's `last_position` vs the call-site position. Since this method doesn't receive position, pass it through: add `candidate_center: Optional[Tuple] = None` parameter and add:

```python
        if candidate_center is not None:
            phys_ok, phys_code = self._physcheck(
                pid=pid, tid=new_tid,
                candidate_center=candidate_center,
                candidate_team=None,
                last_center=slot.last_position if slot else None,
                last_frame=slot.last_seen_frame if slot else None,
                last_team=getattr(slot, 'team_id', None) if slot else None,
            )
            if not phys_ok:
                return False  # block the relink
```

Update all callers of `_relink_absent_existing_lock` to pass `candidate_center=positions.get(new_tid)`.

- [ ] **Step 6: Export physicality metrics in `end_run_summary`**

```python
        print(f"  physicality_rejects          = {self.physicality_rejects}")
        for code, cnt in self.physicality_reject_reasons.items():
            print(f"    {code}: {cnt}")
```

And in the returned dict:
```python
        "physicality_rejects": self.physicality_rejects,
        "physicality_reject_reasons": dict(self.physicality_reject_reasons),
```

- [ ] **Step 7: Run wired integration test**

```bash
python -m pytest tests/test_identity_core_regression.py::TestPhysicalityWired -v 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 8: Run full regression**

```bash
python -m pytest tests/test_identity_core_regression.py tests/test_identity_stability.py tests/test_identity_patch_service.py tests/test_fullfps_tracking_renderer.py -q 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add services/identity_core.py tests/test_identity_core_regression.py
git commit -m "feat: wire check_physicality into all identity assignment paths"
```

---

## Task 5: Extend `PatchValidator` with physicality checks

**Files:**
- Modify: `services/identity_patch_service.py` — call `check_physicality` for OFFICIAL_PID and TEAM_FLIP in `validate()`

- [ ] **Step 1: Write failing test**

Append to `tests/test_identity_patch_service.py`:

```python
class TestPatchValidatorPhysicality:
    """PatchValidator now rejects patches that assign player PID to official."""

    def _make_patch(self, pid_a="P13", is_official_pid=True, confidence=0.90):
        return {
            "action": "assign_pid_to_tracklet",
            "window_id": "w1",
            "pid": pid_a,
            "is_official_target": is_official_pid,
            "confidence": confidence,
            "reason_codes": [],
        }

    def test_official_pid_patch_rejected(self):
        """Patch that assigns a player PID to an official must be rejected."""
        from services.identity_patch_service import PatchValidator
        validator = PatchValidator()
        patch = self._make_patch(pid_a="P13", is_official_pid=True)

        result = validator.validate(patch, {"jobId": "t", "frames": [], "total_frames": 0})
        assert result is not None, "Official PID patch must be rejected"
        assert "OFFICIAL" in result.reason.upper() or "official" in result.reason.lower()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_identity_patch_service.py::TestPatchValidatorPhysicality -v 2>&1 | tail -10
```

Expected: FAIL (patch accepted when it shouldn't be)

- [ ] **Step 3: Extend `PatchValidator.validate` in `services/identity_patch_service.py`**

Import `check_physicality` at top of file:
```python
try:
    from services.identity_core import check_physicality
except ImportError:
    from identity_core import check_physicality  # type: ignore
```

In `PatchValidator.validate`, before the final `return None`, add:

```python
        # Official PID gate: reject patch that assigns player PID to an official
        if patch.get("is_official_target"):
            ok, code, detail = check_physicality(
                pid=patch.get("pid", ""),
                candidate_center=(0.0, 0.0),
                candidate_team=None,
                current_frame=0,
                last_center=None,
                last_frame=None,
                last_team=None,
                is_official=True,
            )
            if not ok:
                return PatchRejection(patch, "OFFICIAL_PID_ASSIGNMENT", detail)
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_identity_patch_service.py -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/test_identity_core_regression.py tests/test_identity_patch_service.py tests/test_identity_stability.py tests/test_fullfps_tracking_renderer.py -q 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add services/identity_patch_service.py tests/test_identity_patch_service.py
git commit -m "feat: PatchValidator rejects official-PID assignments via check_physicality"
```

---

## Task 6: Final test run + git diff summary + push

- [ ] **Step 1: Full suite**

```bash
cd /Users/rudra/athlink-cv-service/athlink-cv-service
python -m pytest tests/test_identity_core_regression.py tests/test_identity_patch_service.py tests/test_identity_stability.py tests/test_fullfps_tracking_renderer.py -q 2>&1
```

Expected: all green.

- [ ] **Step 2: Diff summary**

```bash
git log 561ad30..HEAD --oneline
git diff 561ad30 --stat
```

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-Review

- [x] **Spec coverage:** CongestionDetector (item 1), cluster freeze on all paths (item 2), existing locks preserved (item 3), PhysicalityValidator (item 4), impossible speed/jump/team-flip/official/double-occupancy (item 5), metrics cluster_freeze_blocks + physicality_rejects + reason breakdown (item 6), regression tests P6/P7/P10/P11/speed/official (item 7).
- [x] **No placeholders:** All code blocks complete.
- [x] **Type consistency:** `check_physicality` returns `Tuple[bool, str, str]`; `_physcheck` returns `Tuple[bool, str]` — consistent across Task 3 definition and Task 4 wire-up.
- [x] **`_relink_absent_existing_lock` signature change:** callers in `assign_tracks` must pass `candidate_center=positions.get(new_tid)`. There is exactly one caller. Verify the updated call site in Task 4 Step 5.
- [x] **Revival methods `cluster_tids` param:** `revive_cost_matrix`, `force_commit_remaining_scene_slots`, `revive_from_soft_snapshot` all get `cluster_tids: Optional[Set[int]] = None`. The callers in `tracker_core.py` don't pass it — safe because `None` disables freeze. Wiring tracker_core to pass bboxes is a follow-up.
