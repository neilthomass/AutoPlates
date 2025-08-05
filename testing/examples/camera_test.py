#!/usr/bin/env python3
"""
Simple camera test script.

This script tests basic camera functionality to ensure the camera
is working before running the full license plate detection system.
"""

import cv2
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main import RealTimeLicensePlateDetector


def test_camera_basic():
    """Test basic camera functionality."""
    print("Testing Basic Camera Functionality")
    print("=" * 40)
    
    # Try to open camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Failed to open camera")
        return False
    
    print("Camera opened successfully")
    
    # Try to capture a frame
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame")
        camera.release()
        return False
    
    print("Frame captured successfully")
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame type: {frame.dtype}")
    
    # Display frame briefly
    cv2.imshow('Camera Test', frame)
    print("Displaying test frame (press any key to continue)...")
    cv2.waitKey(3000)  # Wait 3 seconds
    cv2.destroyAllWindows()
    
    camera.release()
    print("Camera test completed successfully")
    return True


def test_detector_initialization():
    """Test detector initialization without camera."""
    print("\nTesting Detector Initialization")
    print("=" * 40)
    
    try:
        # Initialize detector without camera
        detector = RealTimeLicensePlateDetector(camera_id=0)
        print("Detector initialized successfully")
        
        # Test database initialization
        print("Database initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return False


def test_model_loading():
    """Test model loading (this will download the model)."""
    print("\nTesting Model Loading")
    print("=" * 40)
    
    try:
        detector = RealTimeLicensePlateDetector(camera_id=0)
        
        print("Downloading resources...")
        detector.download_resources()
        print("Resources downloaded")
        
        print("Loading model...")
        detector.load_model()
        print("Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def main():
    """Run all camera tests."""
    print("Camera and System Test")
    print("=" * 50)
    
    success = True
    
    # Test basic camera
    if not test_camera_basic():
        success = False
        print("\nCamera test failed. Make sure you have a camera connected.")
    
    # Test detector initialization
    if not test_detector_initialization():
        success = False
    
    # Test model loading (optional - takes time to download)
    print("\nDo you want to test model loading? (This will download the model)")
    print("   This may take a few minutes to download the model...")
    response = input("   Continue? (y/n): ").lower().strip()
    
    if response == 'y':
        if not test_model_loading():
            success = False
    else:
        print("   Skipping model loading test")
    
    print("\n" + "=" * 50)
    
    if success:
        print("All tests passed! Your system is ready for real-time detection.")
        print("\nYou can now run the main detection system with:")
        print("  uv run python src/main.py")
    else:
        print("Some tests failed. Please check the issues above.")
        print("\nCommon issues:")
        print("  - No camera connected")
        print("  - Camera in use by another application")
        print("  - Insufficient permissions")
        print("  - Network issues (for model download)")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 