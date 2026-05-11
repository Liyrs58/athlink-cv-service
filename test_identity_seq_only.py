#!/usr/bin/env python3
"""Quick test: verify identity_frame_seq and pending_streak logic."""

from services.identity_core import IdentityCore, AssignmentMeta
import numpy as np

# Mock track object
class MockTrack:
    def __init__(self, track_id):
        self.track_id = track_id

# Test sequence
ic = IdentityCore(debug_every=1)
ic.begin_frame(0)

# Simulate slot seeding (from seed_provisional_from_tracks)
slots_to_seed = 5
embed_map = {i: np.ones(512, dtype=np.float32) / np.sqrt(512) for i in range(1, slots_to_seed + 1)}
positions = {i: (100 + i * 10, 200) for i in range(1, slots_to_seed + 1)}
pitch_pos = {i: (50 + i * 5, 60) for i in range(1, slots_to_seed + 1)}
team_labels = {i: 1 if i <= 2 else 2 for i in range(1, slots_to_seed + 1)}

seeded = ic.seed_provisional_from_tracks(embed_map, positions, pitch_pos, team_labels, 0)
print(f"[TestSeed] Seeded {seeded} slots")

# Check slot states after seed
for i in range(1, slots_to_seed + 1):
    slot = ic.slots[i - 1]
    print(f"  Slot P{i}: pending_tid={slot.pending_tid} streak={slot.pending_streak} seq={slot.pending_seen_seq} emb={slot.embedding is not None}")

ic.end_frame()

# Now simulate assign_tracks with the seeded slots
ic.begin_frame(1)
tracks = [MockTrack(i) for i in range(1, slots_to_seed + 1)]
embeddings = embed_map.copy()
positions_f1 = positions.copy()

print(f"\n[TestAssign] Frame 1, before assign_tracks: identity_frame_seq={ic.identity_frame_seq}")
track_to_pid, meta_map = ic.assign_tracks(
    tracks, embeddings, positions_f1,
    memory_ok_tids=None,
    allow_new_assignments=True
)
print(f"[TestAssign] Frame 1, after assign_tracks: identity_frame_seq={ic.identity_frame_seq}")
print(f"  Assigned {len(track_to_pid)} tracks")

# Check streaks after frame 1
for i in range(1, slots_to_seed + 1):
    slot = ic.slots[i - 1]
    print(f"  Slot P{i}: pending_tid={slot.pending_tid} streak={slot.pending_streak} seq={slot.pending_seen_seq}")

ic.end_frame()

# Frame 2 - should see streaks increment
ic.begin_frame(2)
tracks = [MockTrack(i) for i in range(1, slots_to_seed + 1)]
embeddings = embed_map.copy()
positions_f2 = positions.copy()

print(f"\n[TestAssign] Frame 2, before assign_tracks: identity_frame_seq={ic.identity_frame_seq}")
track_to_pid, meta_map = ic.assign_tracks(
    tracks, embeddings, positions_f2,
    memory_ok_tids=None,
    allow_new_assignments=True
)
print(f"[TestAssign] Frame 2, after assign_tracks: identity_frame_seq={ic.identity_frame_seq}")

# Check streaks after frame 2
for i in range(1, slots_to_seed + 1):
    slot = ic.slots[i - 1]
    if slot.pending_streak > 1:
        print(f"  Slot P{i}: pending_tid={slot.pending_tid} streak={slot.pending_streak} seq={slot.pending_seen_seq} ← STREAK BUILT!")
    else:
        print(f"  Slot P{i}: pending_tid={slot.pending_tid} streak={slot.pending_streak} seq={slot.pending_seen_seq} ← RESET (BAD)")

ic.end_frame()

# Check final result
locks_created = ic.locks.locks_created
print(f"\n[TestResult]")
print(f"  locks_created = {locks_created}")
print(f"  Expected: >= 1 (if streaks built to 5 over 5+ frames)")

# Expected: If LOCK_PROMOTE_FRAMES=5, we need 5 consecutive assign_tracks calls
# So we'd see locks at frame 5+
print(f"\n  Will need ~5+ processed frames for locks to fire")
print(f"  (LOCK_PROMOTE_FRAMES=5, so streak must reach 5)")
