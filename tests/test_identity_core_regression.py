import pytest
import numpy as np
from services.identity_core import IdentityCore

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

