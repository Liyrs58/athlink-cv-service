"""
Submit both training jobs to HF Jobs.

Run locally:
  export HF_TOKEN=<your-token>
  python submitted_jobs/submit_jobs.py

Jobs:
  1. D-FINE-small football detector  (t4-small, 2h, ~$0.80)
  2. OSNet football ReID              (t4-small, 3h, ~$1.20)
"""
import os
from huggingface_hub import HfApi, get_token

api = HfApi()

# ── Job 1: Football detector ──────────────────────────────────────────────
print("Submitting detector training job...")
detector_job = api.run_uv_job(
    script="submitted_jobs/train_football_detector.py",
    flavor="t4-small",
    timeout=7200,
    env={"PYTHONUNBUFFERED": "1"},
    secrets={"HF_TOKEN": get_token()},
)
print(f"  Detector job ID : {detector_job.id}")
print(f"  Logs            : hf jobs logs {detector_job.id}")
print(f"  Output          : https://huggingface.co/Liyrs58/dfine-football-detector")

# ── Job 2: Football ReID ──────────────────────────────────────────────────
print("\nSubmitting ReID training job...")
reid_job = api.run_uv_job(
    script="submitted_jobs/train_football_reid.py",
    flavor="t4-small",
    timeout=10800,
    env={"PYTHONUNBUFFERED": "1"},
    secrets={"HF_TOKEN": get_token()},
)
print(f"  ReID job ID     : {reid_job.id}")
print(f"  Logs            : hf jobs logs {reid_job.id}")
print(f"  Output          : https://huggingface.co/Liyrs58/football-osnet-reid")

print(f"\nBoth jobs running. Monitor with:")
print(f"  hf jobs ps")
