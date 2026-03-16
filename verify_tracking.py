import json
import sys
import argparse
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Verify tracking results")
    parser.add_argument("jobId", help="Job ID to verify")
    parser.add_argument("--render", action="store_true", help="Render tracked frames")
    args = parser.parse_args()
    
    job_id = args.jobId
    tracking_path = Path(f"temp/{job_id}/tracking/track_results.json")
    
    if not tracking_path.exists():
        print(f"Error: Tracking file not found at {tracking_path}")
        print("Run tracking analysis first.")
        sys.exit(1)
    
    # Load tracking results
    with open(tracking_path) as f:
        data = json.load(f)
    
    # Print summary table
    print(f"\nTracking Summary for Job: {job_id}")
    print("=" * 80)
    print(f"Video Path:     {data['videoPath']}")
    print(f"Frame Stride:   {data['frameStride']}")
    print(f"Frames Processed: {data['framesProcessed']}")
    print(f"Track Count:    {data['trackCount']}")
    print("=" * 80)
    
    if data['trackCount'] == 0:
        print("\nWARNING: No tracks found!")
        print("This likely means detections are empty. Check Brick 5 detection results first.")
        return
    
    # Print per-track table
    print(f"\nTrack Details:")
    print(f"{'ID':<4} {'Hits':<6} {'FirstSeen':<10} {'LastSeen':<10} {'Duration(s)':<12}")
    print("-" * 50)
    
    for track in sorted(data['tracks'], key=lambda x: x['trackId']):
        duration = 0.0
        if track['trajectory']:
            first_time = track['trajectory'][0]['timestampSeconds']
            last_time = track['trajectory'][-1]['timestampSeconds']
            duration = last_time - first_time
        
        print(f"{track['trackId']:<4} {track['hits']:<6} {track['firstSeen']:<10} {track['lastSeen']:<10} {duration:<12.1f}")
    
    # Render frames if requested
    if args.render:
        print(f"\nRendering tracked frames...")

        # Build frame lookup from track trajectories
        frame_lookup = {}
        for track in data['tracks']:
            for point in track['trajectory']:
                fi = point['frameIndex']
                if fi not in frame_lookup:
                    frame_lookup[fi] = []
                frame_lookup[fi].append((track['trackId'], point['bbox']))
        
        if not frame_lookup:
            print("No frames with tracks found to render.")
            return
        
        # Color palette (9 BGR colors)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        
        # Open video
        cap = cv2.VideoCapture(data['videoPath'])
        if not cap.isOpened():
            print(f"Error: Cannot open video {data['videoPath']}")
            return
        raw_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        raw_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        needs_rotation = raw_h > raw_w
        
        # Create output directory
        output_dir = Path(f"temp/{job_id}/verify_tracking")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rendered_count = 0
        
        for frame_idx in sorted(frame_lookup.keys()):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            if needs_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Get timestamp from fps
            fps = data.get('fps', 30.0)
            timestamp = frame_idx / fps if fps > 0 else 0.0
            
            # Draw tracks
            for track_id, bbox in frame_lookup[frame_idx]:
                color = colors[(track_id - 1) % len(colors)]
                
                # Convert bbox [x1, y1, x2, y2] to rectangle
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"T{track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y1 = max(y1 - 25, 0)
                label_y2 = max(y1 - 5, 0)
                cv2.rectangle(frame, (x1, label_y1), (x1 + label_size[0], label_y2), color, -1)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw frame info
            info_text = f"frame {frame_idx} | t={timestamp:.2f}s"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Save frame
            output_filename = f"track_{frame_idx:06d}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), frame)
            rendered_count += 1
        
        cap.release()
        
        print(f"Rendered {rendered_count} frames to {output_dir}")
        print(f"Open directory to review: open {output_dir}")

if __name__ == "__main__":
    main()
