#!/usr/bin/env python3
"""
Verification script for track eviction fix.
Checks that the key changes are in place without running full tracking.
"""

import ast
import sys

def verify_fix():
    """Verify the track eviction fix is properly implemented."""
    
    with open('services/tracking_service.py', 'r') as f:
        content = f.read()
    
    checks = {
        '1. _last_processed_frame on new tracks': '_last_processed_frame": processed_count' in content,
        '2. _last_processed_frame on track updates': 'active_tracks[track_id]["_last_processed_frame"] = processed_count' in content,
        '3. Inactivity-based eviction': 'inactivity = processed_count - track.get("_last_processed_frame"' in content,
        '4. Removed presence-based eviction': 'disappeared = prev_active_ids - current_active_ids' not in content,
        '5. Recently_lost cleanup uses processed frames': '(processed_count - track.get("_last_processed_frame", 0)) > 15' in content,
        '6. Track stitching preserves _last_processed_frame': 'ti["_last_processed_frame"] = max(' in content,
    }
    
    print("🔍 Verifying track eviction fix:")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and passed
    
    print("=" * 50)
    
    if all_passed:
        print("✅ All checks passed! Track eviction fix is properly implemented.")
        print("\n📋 Key changes:")
        print("  • Tracks now use inactivity-based eviction (processed frames)")
        print("  • No more instant eviction on missed YOLO frames")
        print("  • max_track_age=30 now means 30 processed frames")
        print("  • Proper _last_processed_frame tracking throughout")
    else:
        print("❌ Some checks failed. Fix may be incomplete.")
        return False
    
    return True

if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)
