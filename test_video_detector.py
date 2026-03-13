import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_detector():
    print("--- Testing VideoDetector C++ Extension ---")
    
    try:
        import video_detector_cpp
        print("✓ Successfully imported video_detector_cpp")
    except ImportError as e:
        print(f"✗ Failed to import video_detector_cpp: {e}")
        return

    # Check for a sample video or create a dummy one if possible
    # For now, we'll just test the constructor
    model_path = "models/ai_detector_model_pytorch_script.ts"
    if not os.path.exists(model_path):
        print(f"! Model not found at {model_path}. Please ensure it exists.")
        # Create a dummy model for testing if it doesn't exist? 
        # Better to wait for user or use an existing one.
        return

    try:
        print(f"Loading model from {model_path}...")
        detector = video_detector_cpp.VideoDetector(model_path)
        print("✓ VideoDetector initialized successfully")
        
        # We need a video file to test detect_video
        # The user likely has some in data/real_videos/
    except Exception as e:
        print(f"✗ Error during VideoDetector initialization: {e}")

if __name__ == "__main__":
    test_detector()
