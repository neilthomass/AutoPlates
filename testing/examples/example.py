#!/usr/bin/env python3
"""
Example usage of the LicensePlateDetector class.

This script demonstrates how to use the refactored license plate detection
pipeline programmatically with SQL database storage.
"""

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))

from main import LicensePlateDetector
from db_utils import PlateDatabase


def example_basic_usage():
    """Example of basic usage of the LicensePlateDetector."""
    print("License Plate Detector - Basic Example")
    print("=" * 50)
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    try:
        # Download resources (this may take a while on first run)
        print("Downloading model and dataset...")
        detector.download_resources()
        
        # Load the model
        print("🔧 Loading YOLOv8 model...")
        detector.load_model()
        
        # Get sample images and run demo
        print("🎯 Running inference on sample images...")
        sample_images = detector.get_sample_images(num_samples=2)
        detector.run_inference_demo(sample_images)
        
        print("Basic example completed successfully!")
        
    except Exception as e:
        print(f"Error in basic example: {e}")


def example_database_operations():
    """Example of database operations."""
    print("\nLicense Plate Detector - Database Operations Example")
    print("=" * 50)
    
    # Initialize database utility
    db = PlateDatabase()
    
    try:
        # Get database statistics
        stats = db.get_stats()
        if stats:
            print(f"Database Statistics:")
            print(f"   - Total detections: {stats['total_detections']}")
            print(f"   - Unique images: {stats['unique_images']}")
            print(f"   - Average confidence: {stats['avg_confidence']}")
            print(f"   - High confidence detections: {stats['high_confidence_count']}")
        else:
            print("No data in database yet.")
        
        # Get recent detections
        recent = db.get_recent_detections(5)
        if recent:
            print(f"\n Recent detections ({len(recent)}):")
            for detection in recent:
                print(f"   - {detection['timestamp']}: {detection['plate_text']} "
                      f"(conf: {detection['confidence']:.3f})")
        
        # Get high confidence detections
        high_conf = db.get_high_confidence_detections(0.8)
        if high_conf:
            print(f"\n🎯 High confidence detections ({len(high_conf)}):")
            for detection in high_conf:
                print(f"   - {detection['plate_text']} (conf: {detection['confidence']:.3f})")
        
        print("Database operations example completed successfully!")
        
    except Exception as e:
        print(f" Error in database operations example: {e}")


def example_full_pipeline():
    """Example of the full detection and storage pipeline."""
    print("\nLicense Plate Detector - Full Pipeline Example")
    print("=" * 50)
    
    detector = LicensePlateDetector()
    
    try:
        # Setup
        detector.download_resources()
        detector.load_model()
        
        # Process dataset and store in database
        print("Processing dataset and storing in database...")
        detection_count, skipped_images = detector.process_dataset()
        
        print(f"Results:")
        print(f"   - Detections saved to database: {detection_count}")
        print(f"   - Skipped images: {len(skipped_images)}")
        
        # Display database results
        detector.display_database_results()
        
        print("Full pipeline example completed successfully!")
        
    except Exception as e:
        print(f" Error in full pipeline example: {e}")

def example_custom_detection():
    """Example of custom detection on specific images."""
    print("\nLicense Plate Detector - Custom Detection Example")
    print("=" * 50)
    
    detector = LicensePlateDetector()
    
    try:
        # Setup
        detector.download_resources()
        detector.load_model()
        
        # Get sample images
        sample_images = detector.get_sample_images(num_samples=3)
        
        # Process each image individually
        for img_path in sample_images:
            print(f"\nProcessing: {os.path.basename(img_path)}")
            detections = detector.detect_plates_in_image(img_path)
            
            if detections:
                print(f"   Found {len(detections)} license plate(s):")
                for i, detection in enumerate(detections):
                    print(f"     {i+1}. {detection['plate_text']} "
                          f"(conf: {detection['confidence']:.3f})")
            else:
                print("   No license plates detected.")
        
        print("Custom detection example completed successfully!")
        
    except Exception as e:
        print(f" Error in custom detection example: {e}")


def main():
    """Run all examples."""
    print("🎯 Running License Plate Detector Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_database_operations()
    example_full_pipeline()
    example_custom_detection()
    
    print("\nAll examples completed!")
    print("\nTips:")
    print("   - The first run will download the model and dataset")
    print("   - Subsequent runs will reuse downloaded resources")
    print("   - Database is stored in database/plates.db")
    print("   - You can query the database using database/db_utils.py")
    print("   - Check the README.md for more information")


if __name__ == "__main__":
    main() 