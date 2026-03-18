import sys

print("Running startup checks...")
try:
    import cv2
    print(f"cv2 OK: {cv2.__version__}")
except ImportError as e:
    print(f"FATAL: cv2 import failed: {e}")
    sys.exit(1)
try:
    import numpy
    print(f"numpy OK: {numpy.__version__}")
except ImportError as e:
    print(f"FATAL: numpy import failed: {e}")
    sys.exit(1)
try:
    import fastapi
    print(f"fastapi OK: {fastapi.__version__}")
except ImportError as e:
    print(f"FATAL: fastapi import failed: {e}")
    sys.exit(1)
print("All startup checks passed.")
