"""Supabase cloud storage upload and file management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

BUCKET_VIDEOS = "match-videos"
BUCKET_RESULTS = "match-results"
BUCKET_RENDERS = "match-renders"

_client = None  # type: Any


def _get_client():
    # type: () -> Any
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("SUPABASE_URL")
    # Prefer service-role key (bypasses RLS); fall back to anon key
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — storage disabled")
        return None
    from supabase import create_client
    _client = create_client(url, key)
    return _client


_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def upload_file(bucket, path, data, content_type="application/octet-stream"):
    # type: (str, str, bytes, str) -> Optional[str]
    """Upload bytes to Supabase storage. Returns public URL or None."""
    client = _get_client()
    if client is None:
        return None
    try:
        client.storage.from_(bucket).upload(
            path, data,
            {"content-type": content_type, "upsert": "true"}
        )
        result = client.storage.from_(bucket).get_public_url(path)
        return result
    except Exception as e:
        logger.error("[storage] upload failed %s/%s: %s", bucket, path, e)
        return None


def upload_file_from_path(bucket, remote_path, local_path):
    # type: (str, str, str) -> Optional[str]
    """Read a local file and upload it. Returns public URL or None."""
    p = Path(local_path)
    if not p.exists():
        return None
    try:
        data = p.read_bytes()
    except Exception as e:
        logger.error("[storage] cannot read %s: %s", local_path, e)
        return None
    ext = p.suffix.lower()
    content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
    return upload_file(bucket, remote_path, data, content_type)


def get_public_url(bucket, path):
    # type: (str, str) -> Optional[str]
    """Return the public URL for an already-uploaded file."""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.storage.from_(bucket).get_public_url(path)
    except Exception as e:
        logger.error("[storage] get_public_url failed %s/%s: %s", bucket, path, e)
        return None


def upload_job_results(job_id):
    # type: (str) -> Dict[str, Any]
    """Upload all outputs for a completed job to Supabase storage."""
    base = Path("temp") / job_id
    uploaded = {}   # type: Dict[str, str]
    failed = []     # type: list
    skipped = []    # type: list

    # Fixed file mappings: (local_relative, bucket, remote_path, logical_name)
    file_map = [
        ("tracking/track_results.json", BUCKET_RESULTS,
         "{}/track_results.json".format(job_id), "track_results"),
        ("tracking/team_results.json", BUCKET_RESULTS,
         "{}/team_results.json".format(job_id), "team_results"),
        ("pitch/pitch_map.json", BUCKET_RESULTS,
         "{}/pitch_map.json".format(job_id), "pitch_map"),
        ("tactics/tactics_results.json", BUCKET_RESULTS,
         "{}/tactics_results.json".format(job_id), "tactics_results"),
        ("render/output.mp4", BUCKET_RENDERS,
         "{}/output.mp4".format(job_id), "render_output"),
        ("highlights/analytics_highlight.mp4", BUCKET_RENDERS,
         "{}/analytics_highlight.mp4".format(job_id), "analytics_highlight"),
    ]

    for local_rel, bucket, remote, name in file_map:
        local_path = str(base / local_rel)
        if not Path(local_path).exists():
            skipped.append(name)
            continue
        url = upload_file_from_path(bucket, remote, local_path)
        if url is not None:
            uploaded[name] = url
        else:
            failed.append(name)

    # Dynamic directories: clips/, spotlight/, and reports/
    for dirname, label_prefix in [("clips", "clips"), ("spotlight", "spotlight"), ("reports", "reports")]:
        local_dir = base / dirname
        if not local_dir.is_dir():
            continue
        for file_path in sorted(local_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(base / dirname)
            remote = "{}/{}/{}".format(job_id, dirname, rel)
            name = "{}/{}".format(label_prefix, rel)
            url = upload_file_from_path(BUCKET_RENDERS, remote, str(file_path))
            if url is not None:
                uploaded[name] = url
            else:
                failed.append(name)

    return {
        "job_id": job_id,
        "uploaded": uploaded,
        "failed": failed,
        "skipped": skipped,
    }
