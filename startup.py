#!/usr/bin/env python3
"""
Startup script for Railway deployment with better error handling
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting AthLink CV Service...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current directory: {os.getcwd()}")
        
        # Try importing main app
        try:
            from main import app
            logger.info("✓ Main app imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import main app: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Get port from environment
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting on port: {port}")
        
        # Start uvicorn
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
