#!/usr/bin/env python3
"""
Basic usage example for the License Plate Detector.

This example demonstrates the core functionality:
1. Initialize the detector
2. Download resources
3. Load the model
4. Run inference on sample images
5. Display results
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main import LicensePlateDetector


def main():
    """Run basic usage example."""
    print("License Plate Detector - Basic Usage Example")
    print("=" * 60)
    
    # Initialize detector
    print("1. Initializing detector...")
    detector = LicensePlateDetector()
    print("   Detector initialized")
    
    try:
        # Download resources
        print("\n2. Downloading model and dataset...")
        detector.download_resources()
        print("   Resources downloaded")
        
        # Load model
        print("\n3. Loading YOLOv8 model...")
        detector.load_model()
        print("   Model loaded")
        
        # Get sample images
        print("\n4. Getting sample images...")
        sample_images = detector.get_sample_images(num_samples=2)
        print(f"   Found {len(sample_images)} sample images")
        
        # Run inference demo
        print("\n5. Running inference demo...")
        detector.run_inference_demo(sample_images)
        print("   Inference demo completed")
        
        print("\nBasic usage example completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 