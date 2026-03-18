
import json
from services.game_brain import detect_situation, get_situation_history

with open("temp/game_brain_test/tracking/track_results.json") as f:
    r = json.load(f)

tracks = r.get("tracks", [])
frame_metadata = r.get("frameMetadata", [])
situation_history = []
situation_counts = {}

for meta in frame_metadata:
    frame_idx = meta.get("frameIndex", 0)
    active = [t for t in tracks if t.get("firstSeen",0) <= frame_idx <= t.get("lastSeen",0)]
    result = detect_situation(tracks=active, ball=None, frame_idx=frame_idx)
    situation_history.append(result)
    s = result["situation"]
    situation_counts[s] = situation_counts.get(s, 0) + 1

total = sum(situation_counts.values())
for sit, count in sorted(situation_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100 if total > 0 else 0
    print(f"{sit}: {count} frames ({pct:.1f}%)")
