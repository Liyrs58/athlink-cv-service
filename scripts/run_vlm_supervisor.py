import argparse
import json
import os
import sys
from pathlib import Path

try:
    from google import genai
    from PIL import Image
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def main():
    parser = argparse.ArgumentParser(description="VLM Supervisor (Gemini Free Tier)")
    parser.add_argument("--job-id", required=True, help="Job identifier")
    args = parser.parse_args()

    if not HAS_GENAI:
        print("[VLM] Error: google-generativeai or Pillow not installed.")
        print("      Run: pip install google-generativeai Pillow")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[VLM] Error: GEMINI_API_KEY environment variable is not set.")
        print("      Get a free key from Google AI Studio and set it.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

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
        if not window_id:
            continue

        print(f"\n[VLM] Reviewing {window_id}...")

        prompt = case_data.get("vlm_prompt", "")
        img = Image.open(contact_sheet_path)

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, img]
            )
            text = response.text

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                decision = result.get("decision", "needs_human")
                print(f"  -> Decision: {decision}")
                print(f"  -> Summary: {result.get('analyst_summary', '')}")

                # Update patch plan if accepted
                if decision == "accept_patch" and window_id in corrections_by_window:
                    corrections_by_window[window_id]["vlm_approved"] = True
                elif decision == "reject_patch" and window_id in corrections_by_window:
                    corrections_by_window[window_id]["vlm_approved"] = False

            else:
                print(f"  -> Failed to parse JSON from VLM response: {text[:100]}...")

        except Exception as e:
            print(f"  -> API Error: {e}")
            
        reviewed_count += 1

    # Save updated patch plan
    patch_plan["corrections"] = list(corrections_by_window.values())
    with open(patch_plan_path, "w") as f:
        json.dump(patch_plan, f, indent=2)

    print(f"\n[VLM] Finished reviewing {reviewed_count} cases.")
    print(f"[VLM] Updated {patch_plan_path}")

if __name__ == "__main__":
    main()
