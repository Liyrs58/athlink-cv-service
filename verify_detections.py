import json
import sys
import cv2
import numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_detections.py <jobId>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    detections_path = Path(f"temp/{job_id}/detections/sample_detections.json")
    
    if not detections_path.exists():
        print(f"Error: Detection file not found at {detections_path}")
        print("Run POST /api/v1/detect/players first.")
        sys.exit(1)
    
    # Load detection results
    with open(detections_path) as f:
        frames = json.load(f)
    
    # Create output directory
    output_dir = Path(f"temp/{job_id}/verify")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_detections = 0
    
    print(f"\nVerifying detections for job: {job_id}")
    print("=" * 60)
    
    for frame in frames:
        frame_index = frame["frameIndex"]
        timestamp = frame.get("timestampSeconds", 0.0)
        image_path = Path(frame["imagePath"])
        detections = frame.get("detections", [])
        
        # Get brightness from frame metadata if available
        brightness = "N/A"
        if "brightness" in frame:
            brightness = f"{frame['brightness']:.1f}"
        
        # Load image
        if not image_path.exists():
            print(f"  [skip] Frame {frame_index}: Image not found at {image_path}")
            continue
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [skip] Frame {frame_index}: Failed to load image")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Draw detections
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["className"]
            
            # Convert bbox [x, y, width, height] to [x1, y1, x2, y2]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = x1 + int(bbox[2])
            y2 = y1 + int(bbox[3])
            
            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # Draw green rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label text
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y1 = max(y1 - 25, 0)
            label_y2 = max(y1 - 5, 0)
            cv2.rectangle(img, (x1, label_y1), (x1 + label_size[0], label_y2), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw info bar at bottom
        info_text = f"frame {frame_index} | t={timestamp}s | brightness={brightness} | {len(detections)} detections"
        
        # Create semi-transparent overlay for info bar
        overlay = img.copy()
        bar_height = 30
        cv2.rectangle(overlay, (0, img_height - bar_height), (img_width, img_height), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Add info text
        cv2.putText(img, info_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save annotated image
        output_filename = f"verified_{frame_index:06d}.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), img)
        
        # Print frame info
        print(f"  Frame {frame_index:4d} | t={timestamp:6.2f}s | brightness={brightness:>6} | {len(detections):2d} detections | -> {output_filename}")
        
        total_detections += len(detections)
    
    # Summary
    print("=" * 60)
    print(f"Total frames processed: {len(frames)}")
    print(f"Total detections: {total_detections}")
    
    if total_detections == 0:
        print("\n" + "!" * 60)
        print("WARNING: No detections found!")
        print("!" * 60)
        print("\nPossible causes:")
        print("1. Frames are dark/blank — check brightness values in detection JSON")
        print("2. Wrong YOLO model — check YOLO_MODEL_PATH environment variable")
        print("3. Confidence threshold too high — try YOLO_CONF=0.25")
        print("4. No people visible in sampled frames")
        print("\nSuggestions:")
        print("- Check the original frame images in temp/{job_id}/frames/")
        print("- Try lowering confidence threshold: export YOLO_CONF=0.25")
        print("- Verify model path: echo $YOLO_MODEL_PATH")
    
    print(f"\nAnnotated images saved to: {output_dir}")
    print(f"Open the directory to review: open {output_dir}")

if __name__ == "__main__":
    main()
