import cv2
import os
from pathlib import Path

def compile_video(image_folder, output_path):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        print("No images found.")
        return
    
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    
    # Use mp4v or h264 for Mac compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 5.0, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    video.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    compile_video("/Users/rudra/athlink-cv-service/athlink-cv-service/temp/e2e_v2/verify_tracking", "v2_tracking_overlay.mp4")
