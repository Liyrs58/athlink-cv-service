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
