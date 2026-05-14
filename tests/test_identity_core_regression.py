import pytest
import numpy as np
from services.identity_core import IdentityCore
from types import SimpleNamespace

def test_seed_provisional_unwraps_dict_embedding():
    """
    Regression test for the dual-feature dictionary bug.
    Ensures that when seed_provisional_from_tracks is passed an embed_map
    containing dicts (e.g. {"emb": ndarray, "hsv": ndarray}), it correctly
    unwraps them and stores ONLY the ndarray in slot.embedding.
    """
    # Initialize IdentityCore
    identity = IdentityCore()
    
    # Create mock inputs for seed_provisional_from_tracks
    tid = 1
    mock_emb_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_hsv_array = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    
    # embed_map contains a dictionary as the value (simulating dual-feature)
    embed_map = {
        tid: {
            "emb": mock_emb_array,
            "hsv": mock_hsv_array
        }
    }
    
    positions = {tid: (100.0, 200.0)}
    pitch_positions = {tid: (0.5, 0.5)}
    team_labels = {tid: 1}
    frame_id = 10
    
    # Call the method
    seeded_count = identity.seed_provisional_from_tracks(
        embed_map,
        positions,
        pitch_positions,
        team_labels,
        frame_id
    )
    
    assert seeded_count == 1, "Should have seeded exactly one track"
    
    # Retrieve the slot that was seeded (it should be the first one)
    slot = identity.slots[0]
    
    # VERIFY THE FIX: The embedding must be an ndarray, NOT a dict!
    assert slot.embedding is not None, "Slot embedding should be populated"
    assert not isinstance(slot.embedding, dict), "FATAL: slot.embedding is still a dictionary!"
    assert isinstance(slot.embedding, np.ndarray), "slot.embedding must be an ndarray"
    
    # Verify the contents match the unwrapped array
    np.testing.assert_array_equal(slot.embedding, mock_emb_array)
    
    # Additionally, verify that if we run _slot_cost with a dual-embedding dict, it also unwraps it
    cost = identity._slot_cost(
        slot=slot,
        t_data=embed_map[tid],  # Pass the dict directly
        t_pos=(100.0, 200.0),
        tid=tid
    )
    
    # Should calculate successfully without throwing TypeError
    assert isinstance(cost, float), "Cost should compute successfully as a float"


def test_player_slot_cap_blocks_new_pid_creation(monkeypatch):
    """Extra raw tracks must stay UNKNOWN instead of creating PIDs above the match cap."""
    monkeypatch.setenv("ATHLINK_MAX_PLAYER_SLOTS", "2")
    identity = IdentityCore()
    identity.begin_frame(20, present_tids={99})

    # Simulate P1/P2 already occupied in this frame. The next unmatched track
    # must not be allowed to grab P3 just because that empty slot exists.
    for slot in identity.slots[:2]:
        slot.seen_this_frame = True
        slot.state = "active"
        slot.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    track = SimpleNamespace(track_id=99)
    _, meta = identity.assign_tracks(
        [track],
        {99: np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        {99: (100.0, 200.0)},
        allow_new_assignments=True,
    )

    assert meta[99].pid is None
    assert meta[99].identity_valid is False


def test_player_slot_cap_still_allows_registered_slot_recovery(monkeypatch):
    """The cap should block fresh PIDs, not prevent matching back to known slots."""
    monkeypatch.setenv("ATHLINK_MAX_PLAYER_SLOTS", "2")
    identity = IdentityCore()
    identity.begin_frame(21, present_tids={42})

    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    slot = identity.slots[0]
    slot.embedding = emb.copy()
    slot.state = "dormant"
    slot.last_position = (100.0, 200.0)
    slot.last_seen_frame = 20

    track = SimpleNamespace(track_id=42)
    _, meta = identity.assign_tracks(
        [track],
        {42: emb.copy()},
        {42: (102.0, 201.0)},
        allow_new_assignments=True,
    )

    assert meta[42].pid == "P1"


def test_absent_stable_lock_can_relink_to_tracker_fragment(monkeypatch):
    """A stable PID may follow a new raw tid when the old tid is absent."""
    monkeypatch.setenv("ATHLINK_MAX_PLAYER_SLOTS", "2")
    identity = IdentityCore(debug_every=9999)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    identity.locks.try_create_lock(
        tid=1, pid="P1", source="hungarian", frame_id=0, confidence=0.0
    )
    identity.locks.get_lock(1).stable_count = 20
    slot = identity.slots[0]
    slot.embedding = emb.copy()
    slot.state = "dormant"
    slot.last_position = (100.0, 200.0)
    slot.last_seen_frame = 0

    track = SimpleNamespace(track_id=99)
    saw_revived = False
    for frame_id in range(1, 7):
        identity.begin_frame(frame_id, present_tids={99})
        _, meta = identity.assign_tracks(
            [track],
            {99: emb.copy()},
            {99: (100.0 + frame_id, 200.0)},
            allow_new_assignments=True,
        )
        saw_revived = saw_revived or meta[99].source == "revived"
        identity.end_frame(frame_id)

    assert identity.locks.get_tid_for_pid("P1") == 99
    assert meta[99].pid == "P1"
    assert saw_revived
    assert identity.revived_count >= 1


class TestOfficialGateRegression:
    """Regression: referee/official must never receive a player P-ID."""

    def _make_track(self, tid: int):
        class T:
            track_id = tid
            time_since_update = 0
        return T()

    def test_suspected_official_cannot_receive_P13(self):
        """Exact regression for the observed P13 referee bug."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        for frame in range(10):
            identity.begin_frame(frame, present_tids={45})
            track_to_pid, meta = identity.assign_tracks(
                tracks=[self._make_track(45)],
                embeddings={45: emb},
                positions={45: (800.0, 400.0)},
                allow_new_assignments=True,
                official_tids={45},
            )
            identity.end_frame()

            assert track_to_pid.get(45) is None, \
                f"Frame {frame}: official tid=45 must not receive P-ID, got {track_to_pid.get(45)}"
            assert meta[45].source == "official_blocked", \
                f"Frame {frame}: source must be 'official_blocked', got {meta[45].source}"

        assert identity.official_pid_blocks == 10, \
            f"Expected 10 blocks over 10 frames, got {identity.official_pid_blocks}"

    def test_official_cannot_steal_pid_from_player(self):
        """If P5 was locked to tid=10 and tid=10 is re-classified as official, P5 must not emit."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()
        identity.locks.try_create_lock(10, "P5", "hungarian", frame_id=0)

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        identity.begin_frame(1, present_tids={10})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(10)],
            embeddings={10: emb},
            positions={10: (200.0, 300.0)},
            allow_new_assignments=True,
            official_tids={10},
        )
        identity.end_frame()

        assert track_to_pid.get(10) is None, "P5 must not emit when tid=10 is official"
        assert meta[10].source == "official_blocked"


class TestCongestionDetector:
    """CongestionDetector tags TIDs that are inside dense clusters."""

    def _make_bbox(self, cx, cy, w=50, h=120):
        return [cx - w//2, cy - h//2, cx + w//2, cy + h//2]

    def test_overlapping_group_tagged(self):
        """Three boxes heavily overlapping → all 3 tagged as in-cluster."""
        from services.identity_core import CongestionDetector
        det = CongestionDetector(iou_threshold=0.10, radius_px=80, min_neighbors=2)

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
            (7, self._make_bbox(900, 200)),
        ]
        in_cluster = det.detect(tid_bboxes)
        assert 7 not in in_cluster, "isolated tid=7 must not be tagged"

    def test_pair_below_threshold_not_tagged(self):
        """Only 2 players with min_neighbors=3 → not in cluster."""
        from services.identity_core import CongestionDetector
        det = CongestionDetector(iou_threshold=0.10, radius_px=80, min_neighbors=3)

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

        slot = identity.slots[0]
        slot.embedding = emb.copy()
        slot.state = "active"
        slot.pending_streak = 10
        slot.pending_tid = 7
        # identity_frame_seq starts at 0; assign_tracks increments to 1 first thing,
        # then checks seq_ok = pending_seen_seq == identity_frame_seq - 1 == 0.
        # Set pending_seen_seq = 0 so seq_ok=True and streak increments (→11 >= 5).
        slot.pending_seen_seq = 0
        slot.last_assigned_tid = 7

        tight_bboxes = {
            7:  self._make_bbox(300, 400),
            8:  self._make_bbox(305, 402),
            9:  self._make_bbox(310, 405),
        }

        identity.begin_frame(1, present_tids={7, 8, 9})
        # identity_frame_seq is 0 at init; assign_tracks will increment to 1
        # so seq_ok = (0 == 0) = True — streak increments
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (300.0, 400.0)},
            allow_new_assignments=True,
            tid_bboxes=tight_bboxes,
        )
        identity.end_frame()

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

    def test_p7_striker_exit_not_inherited_during_congestion(self):
        """Regression: different tid in cluster must not inherit P7."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()

        emb7 = np.random.randn(512).astype(np.float32)
        emb7 /= np.linalg.norm(emb7)

        slot7 = identity._slot_by_pid("P7")
        slot7.embedding = emb7.copy()
        slot7.state = "active"
        slot7.pending_streak = 10
        slot7.pending_tid = 15
        slot7.last_assigned_tid = 15
        # identity_frame_seq starts at 0; assign_tracks increments to 1,
        # so seq_ok = (pending_seen_seq == 0) = True → streak increments
        slot7.pending_seen_seq = 0

        tight_bboxes = {
            15: self._make_bbox(490, 380),
            16: self._make_bbox(495, 382),
            17: self._make_bbox(500, 385),
        }

        identity.begin_frame(30, present_tids={15, 16, 17})
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

        assert track_to_pid.get(15) != "P7", \
            "P7 must not be inherited by tid=15 in dense cluster"
        assert identity.cluster_freeze_blocks >= 1

    def test_p6_cannot_revive_through_cluster(self):
        """Regression: P6 revival must be blocked when target TID is in cluster."""
        import numpy as np
        from services.identity_core import IdentityCore, ShadowEntry

        identity = IdentityCore()
        identity.shadow_buffer._require_edge = False
        identity.shadow_buffer._max_cost = 0.99

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        entry = ShadowEntry(
            pid="P6", last_tid=6, last_seen_frame=5,
            last_bbox=None, last_center=(400.0, 300.0),
            last_embedding=emb.copy(), team_id=0,
            exit_edge="interior", stable_count=20,
        )
        identity.shadow_buffer.add(entry, added_frame=5)

        slot = identity._slot_by_pid("P6")
        slot.embedding = emb.copy()
        slot.state = "active"
        slot.pending_streak = 10
        slot.pending_tid = 21
        slot.pending_seen_seq = 0

        tight_bboxes = {
            20: self._make_bbox(830, 140),
            21: self._make_bbox(835, 143),
            22: self._make_bbox(840, 148),
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
            "P6 must not revive to cluster-congested winger TID"
        assert identity.cluster_freeze_blocks >= 1


class TestPhysicalityValidator:
    """check_physicality rejects impossible assignments."""

    def test_official_pid_rejected(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P13", candidate_center=(800.0, 400.0), candidate_team=None,
            current_frame=10, last_center=None, last_frame=None, last_team=None,
            is_official=True,
        )
        assert not ok
        assert code == "OFFICIAL_PID"

    def test_impossible_pixel_jump_rejected(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P6", candidate_center=(1800.0, 400.0), candidate_team=0,
            current_frame=11, last_center=(100.0, 400.0), last_frame=10,
            last_team=0, fps=30.0, frame_stride=1, max_speed_px_per_sec=650.0,
        )
        assert not ok, f"1700px jump in 1 frame must be rejected; got ok={ok}"
        assert code in ("IMPOSSIBLE_SPEED", "IMPOSSIBLE_PIXEL_JUMP")

    def test_team_flip_rejected(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P10", candidate_center=(500.0, 300.0), candidate_team=1,
            current_frame=50, last_center=(510.0, 305.0), last_frame=49, last_team=0,
        )
        assert not ok, "Team flip must be rejected"
        assert code == "TEAM_FLIP"

    def test_plausible_assignment_accepted(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P5", candidate_center=(310.0, 405.0), candidate_team=0,
            current_frame=11, last_center=(300.0, 400.0), last_frame=10,
            last_team=0, fps=30.0, frame_stride=1,
        )
        assert ok, f"Plausible assignment must pass; code={code}"

    def test_double_occupancy_rejected(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P3", candidate_center=(500.0, 300.0), candidate_team=0,
            current_frame=20, last_center=None, last_frame=None, last_team=None,
            all_current_pids={"P3": (500.0, 300.0)},
        )
        assert not ok
        assert code == "DOUBLE_OCCUPANCY"

    def test_p10_team_flip_rejected(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P10", candidate_center=(600.0, 350.0), candidate_team=1,
            current_frame=90, last_center=(590.0, 340.0), last_frame=85, last_team=0,
        )
        assert not ok
        assert code == "TEAM_FLIP"

    def test_p11_impossible_winger_jump_rejected(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P11", candidate_center=(100.0, 300.0), candidate_team=0,
            current_frame=341, last_center=(960.0, 540.0), last_frame=340,
            fps=30.0, frame_stride=1, max_speed_px_per_sec=650.0, last_team=0,
        )
        assert not ok
        assert code in ("IMPOSSIBLE_SPEED", "IMPOSSIBLE_PIXEL_JUMP")

    def test_pan_compensated_motion_not_rejected_as_player_speed(self):
        from services.identity_core import check_physicality
        ok, code, detail = check_physicality(
            pid="P5", candidate_center=(125.0, 400.0), candidate_team=0,
            current_frame=11, last_center=(100.0, 400.0), last_frame=10,
            last_team=0, fps=30.0, frame_stride=1, max_speed_px_per_sec=650.0,
            camera_motion={"dx": 25.0, "dy": 0.0, "motion_class": "pan"},
        )
        assert ok, f"Pan-compensated movement should pass physicality; code={code} detail={detail}"


class TestPhysicalityWired:
    """check_physicality is wired into assign_tracks."""

    def _make_track(self, tid):
        class T:
            track_id = tid
            time_since_update = 0
        return T()

    def test_impossible_jump_on_locked_pair_rejected(self):
        """Locked pair P7/tid=7: 1720px jump in 1 frame → physicality reject."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        identity.locks.try_create_lock(7, "P7", "hungarian", frame_id=0)
        slot = identity._slot_by_pid("P7")
        slot.embedding = emb.copy()
        slot.last_position = (100.0, 400.0)
        slot.last_seen_frame = 10

        identity.begin_frame(11, present_tids={7})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(7)],
            embeddings={7: emb},
            positions={7: (1820.0, 400.0)},   # 1720px jump in 1 frame at 30fps
            allow_new_assignments=True,
        )
        identity.end_frame()

        assert identity.physicality_rejects >= 1, \
            "Physicality reject counter must increment for impossible jump"
        assert (
            "IMPOSSIBLE_SPEED" in identity.physicality_reject_reasons or
            "IMPOSSIBLE_PIXEL_JUMP" in identity.physicality_reject_reasons
        )

    def test_locked_pair_uses_camera_motion_compensation(self):
        """Camera pan should not make a locked player fail raw pixel-speed checks."""
        import numpy as np
        from services.identity_core import IdentityCore

        identity = IdentityCore()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        identity.locks.try_create_lock(5, "P5", "hungarian", frame_id=0)
        slot = identity._slot_by_pid("P5")
        slot.embedding = emb.copy()
        slot.last_position = (100.0, 400.0)
        slot.last_seen_frame = 10

        identity.begin_frame(11, present_tids={5})
        track_to_pid, meta = identity.assign_tracks(
            tracks=[self._make_track(5)],
            embeddings={5: emb},
            positions={5: (125.0, 400.0)},
            allow_new_assignments=True,
            camera_motion={"dx": 25.0, "dy": 0.0, "motion_class": "pan"},
        )
        identity.end_frame()

        assert track_to_pid.get(5) == "P5"
        assert identity.physicality_rejects == 0


class TestShadowCaptureRegression:
    """Shadow capture should represent departures, not every active unlocked slot."""

    def test_recently_seen_active_slot_not_immediately_recaptured_as_shadow(self):
        from services.identity_core import IdentityCore

        identity = IdentityCore()
        slot = identity._slot_by_pid("P8")
        slot.state = "active"
        slot.last_position = (500.0, 600.0)
        slot.last_seen_frame = 20
        slot.last_lock_stable_count = 80
        slot.last_assigned_tid = 117

        identity.begin_frame(21, present_tids=set())

        assert not identity.shadow_buffer.has_shadow("P8")
