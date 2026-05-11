#!/usr/bin/env python3
"""Test: verify pending_streak reaches 5 and locks are created."""

from services.identity_core import IdentityCore
import numpy as np

class MockTrack:
    def __init__(self, track_id):
        self.track_id = track_id

ic = IdentityCore(debug_every=1)

# Seed 3 slots with embeddings
embed_map = {i: np.ones(512, dtype=np.float32) / np.sqrt(512) for i in range(1, 4)}
positions = {i: (100 + i * 10, 200) for i in range(1, 4)}
pitch_pos = {i: (50 + i * 5, 60) for i in range(1, 4)}
team_labels = {i: 1 if i <= 2 else 2 for i in range(1, 4)}

ic.begin_frame(0)
ic.seed_provisional_from_tracks(embed_map, positions, pitch_pos, team_labels, 0)
ic.end_frame()

# Run 5 frames of matching
for frame in range(1, 6):
    ic.begin_frame(frame)
    tracks = [MockTrack(i) for i in range(1, 4)]
    track_to_pid, meta_map = ic.assign_tracks(
        tracks, embed_map, positions,
        allow_new_assignments=True
    )

    # Report streaks
    print(f"\n[Frame {frame}]")
    for i in range(1, 4):
        slot = ic.slots[i - 1]
        print(f"  P{i}: tid={slot.pending_tid} streak={slot.pending_streak}")

    locks = ic.locks.locks_created
    print(f"  locks_created={locks}")

    ic.end_frame()

print(f"\n[FINAL] locks_created={ic.locks.locks_created}")
print(f"Expected: >= 3 (one lock per seeded track)")
