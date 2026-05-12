import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
    from PIL import Image
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def main():
    parser = argparse.ArgumentParser(description="VLM Supervisor (Multi-Backend)")
    parser.add_argument("--job-id", required=True, help="Job identifier")
    parser.add_argument("--all", action="store_true", help="Review all severities (default is high severity only)")
    parser.add_argument("--vlm-backend", choices=["gemini", "qwen_local", "gemma_local", "none"], default="gemini",
                        help="VLM backend to use")
    parser.add_argument("--severity-routing", action="store_true", 
                        help="Route severities to different backends (high->gemini/qwen_7b, medium->qwen_3b)")
    args = parser.parse_args()

    if args.vlm_backend == "gemini":
        if not HAS_GENAI:
            print("[VLM] Error: google-genai or Pillow not installed.")
            print("      Run: pip install google-genai Pillow")
            sys.exit(1)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[VLM] Error: GEMINI_API_KEY environment variable is not set.")
            print("      Get a free key from Google AI Studio and set it.")
            sys.exit(1)

        client = genai.Client(api_key=api_key)
    elif args.vlm_backend == "none":
        print("[VLM] Backend set to 'none'. Exiting.")
        sys.exit(0)
    else:
        print(f"[VLM] Backend {args.vlm_backend} not fully implemented yet in this script stub.")
        client = None

    casefiles_dir = Path(f"temp/{args.job_id}/vlm_casefiles")
    if not casefiles_dir.exists():
        print(f"[VLM] No casefiles found at {casefiles_dir}")
        sys.exit(0)

    patch_plan_path = Path(f"temp/{args.job_id}/identity_patch_plan.json")
    if not patch_plan_path.exists():
        print(f"[VLM] No patch plan found at {patch_plan_path}")
        sys.exit(1)

    with open(patch_plan_path) as f:
        patch_plan = json.load(f)

    # Convert corrections list to dict for easier updating
    corrections_by_window = {c.get("window_id"): c for c in patch_plan.get("corrections", [])}

    print(f"[VLM] Starting review of casefiles in {casefiles_dir}")
    reviewed_count = 0

    for case_dir in sorted(casefiles_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_json_path = case_dir / "case.json"
        contact_sheet_path = case_dir / "contact_sheet.jpg"

        if not case_json_path.exists() or not contact_sheet_path.exists():
            continue

        with open(case_json_path) as f:
            case_data = json.load(f)

        window_id = case_data.get("window_id")
        severity = case_data.get("severity", "low").lower()
        if not window_id:
            continue
            
        if not args.all and severity != "high":
            continue

        print(f"\n[VLM] Reviewing {window_id} (severity: {severity})...")

        prompt = case_data.get("vlm_prompt", "")
        img = Image.open(contact_sheet_path)

        if args.severity_routing:
            if severity == "low":
                print("  -> Low severity: deterministic only, skipping VLM")
                continue
            elif severity == "medium":
                active_backend = "qwen_local" # conceptual routing to 3B
                model_name = "qwen2.5-vl-3b"
            elif severity == "high":
                active_backend = args.vlm_backend # user default or gemini
                model_name = "gemini-2.5-flash" if active_backend == "gemini" else "qwen2.5-vl-7b"
            else:
                active_backend = args.vlm_backend
                model_name = "gemini-2.5-flash"
        else:
            active_backend = args.vlm_backend
            model_name = "gemini-2.5-flash"

        # Caching logic
        import hashlib
        # include model name in cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cache_key = f"{window_id}_{prompt_hash}_{model_name}_{active_backend}"
        cache_path = case_dir / f"vlm_cache_{cache_key}.json"
        
        if cache_path.exists():
            print(f"  -> Cache hit for {window_id}")
            with open(cache_path) as f:
                result = json.load(f)
        else:
            try:
                if active_backend == "gemini" and client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            prompt + "\n\nProvide your answer ONLY as a JSON object matching this schema: {\"decision\": \"accept_patch | reject_patch | needs_human | no_change\", \"confidence\": 0.0, \"reason_code\": \"string\", \"pid\": \"P13\", \"frame_start\": 0, \"frame_end\": 0, \"recommended_action\": \"reject_revival | split_tracklet | swap_pid_after_frame | no_change\", \"human_review_required\": false}", 
                            img
                        ]
                    )
                    text = response.text
                else:
                    # Mock response for local models not yet implemented
                    text = '{"decision": "needs_human", "confidence": 0.0, "reason_code": "NOT_IMPLEMENTED", "human_review_required": true}'

                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = {"decision": "needs_human", "reason_code": "JSON_PARSE_ERROR"}
                    
                # Save to cache
                with open(cache_path, "w") as f:
                    json.dump(result, f)
                    
                # Rate limit for free tier
                if args.vlm_backend == "gemini":
                    time.sleep(4)

            except Exception as e:
                print(f"  -> API Error: {e}")
                result = {"decision": "VLM_UNREVIEWED", "reason_code": "API_FAILURE"}
                
        decision = result.get("decision", "VLM_UNREVIEWED")
        print(f"  -> Decision: {decision}")
        
        # Update patch plan if accepted
        if decision == "accept_patch" and window_id in corrections_by_window:
            corrections_by_window[window_id]["vlm_approved"] = True
        elif decision == "reject_patch" and window_id in corrections_by_window:
            corrections_by_window[window_id]["vlm_approved"] = False
        elif decision == "VLM_UNREVIEWED" and window_id in corrections_by_window:
            # Leave it as unreviewed, don't crash
            corrections_by_window[window_id]["vlm_approved"] = None

        reviewed_count += 1

    # Save updated patch plan
    patch_plan["corrections"] = list(corrections_by_window.values())
    with open(patch_plan_path, "w") as f:
        json.dump(patch_plan, f, indent=2)

    print(f"\n[VLM] Finished reviewing {reviewed_count} cases.")
    print(f"[VLM] Updated {patch_plan_path}")

if __name__ == "__main__":
    main()
