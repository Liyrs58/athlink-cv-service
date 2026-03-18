from services.tracking_service import run_tracking
from services.game_brain import detect_situation, get_situation_history

r = run_tracking(
    job_id='game_brain_test',
    video_path='/Users/rudra/Desktop/villa_psg_40s.mp4',
    frame_stride=2,
    max_frames=500
)

tracks = r.get('tracks', [])
frame_metadata = r.get('frameMetadata', [])

print(f'Total tracks: {len(tracks)}')
print(f'Tracks with 5+ detections: {sum(1 for t in tracks if t.get("confirmed_detections",0) >= 5)}')
print()

situation_history = []
situation_counts = {}

for meta in frame_metadata:
    frame_idx = meta.get('frameIndex', 0)
    active = [t for t in tracks if t.get('firstSeen',0) <= frame_idx <= t.get('lastSeen',0)]
    result = detect_situation(tracks=active, ball=None, frame_idx=frame_idx)
    situation_history.append(result)
    s = result['situation']
    situation_counts[s] = situation_counts.get(s, 0) + 1

print('--- Situation breakdown ---')
total = sum(situation_counts.values())
for sit, count in sorted(situation_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100 if total > 0 else 0
    print(f'{sit:15s}: {count:4d} frames ({pct:.1f}%)')

print()
print(f'Dominant: {get_situation_history(situation_history, window=len(situation_history))}')
