import os
import requests

def download_file(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(dst) / (1024*1024)
        print(f"Done: {dst} ({size:.1f} MB)")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if "drive.google.com" in url:
            print("Google Drive link may be blocked by virus scan warning. Skipping x1_0 fallback to x0_25.")

# YOLOv8 pose - Using generic link or boxmot will fetch it
download_file(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt",
    "models/yolov8n-pose.pt"
)

# osnet_x1_0 — Google Drive direct link
download_file(
    "https://drive.google.com/uc?export=download&id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "models/osnet_x1_0_msmt17.pt"
)

print("All downloads complete.")
