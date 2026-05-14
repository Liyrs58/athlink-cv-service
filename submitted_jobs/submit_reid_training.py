"""
Submit the football OSNet ReID training job to Hugging Face Jobs.

Run manually:
    python submitted_jobs/submit_reid_training.py

DO NOT import or call from application code — human-triggered only.
"""

from huggingface_hub import HfApi, get_token

api = HfApi()
job_info = api.run_uv_job(
    script="submitted_jobs/train_football_reid.py",
    flavor="t4-small",
    timeout=10800,
    env={"PYTHONUNBUFFERED": "1"},
    secrets={"HF_TOKEN": get_token()},
)
print(f"Job ID: {job_info.id}")
print(f"Logs: hf jobs logs {job_info.id}")
