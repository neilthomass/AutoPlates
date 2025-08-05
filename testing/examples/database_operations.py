#!/usr/bin/env python3
"""
Database operations example for the License Plate Detector.

This example demonstrates:
1. Database initialization and setup
2. Querying database statistics
3. Retrieving recent detections
4. Filtering by confidence
5. Exporting data to CSV
"""

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))

from main import LicensePlateDetector
from db_utils import PlateDatabase


def main():
    """Run database operations example."""
    print("License Plate Detector - Database Operations Example")
    print("=" * 60)
    
    # Initialize detector and database
    print("1. Initializing detector and database...")
    detector = LicensePlateDetector()
    db = PlateDatabase()
    print("   Detector and database initialized")
    
    try:
        # Download resources and load model
        print("\n2. Setting up model...")
        detector.download_resources()
        detector.load_model()
        print("   Model ready")
        
        # Process some images to populate database
        print("\n3. Processing sample images...")
        sample_images = detector.get_sample_images(num_samples=3)
        
        # Process each image individually
        for i, img_path in enumerate(sample_images):
            print(f"   Processing image {i+1}/{len(sample_images)}: {os.path.basename(img_path)}")
            detections = detector.detect_plates_in_image(img_path)
            
            if detections:
                print(f"     Found {len(detections)} license plate(s)")
                for j, detection in enumerate(detections):
                    print(f"       {j+1}. {detection['plate_text']} (conf: {detection['confidence']:.3f})")
            else:
                print("     No license plates detected")
        
        print("   Sample processing completed")
        
        # Query database statistics
        print("\n4. Querying database statistics...")
        stats = db.get_stats()
        
        if stats:
            print(f"   Database Statistics:")
            print(f"      - Total detections: {stats['total_detections']}")
            print(f"      - Unique images: {stats['unique_images']}")
            print(f"      - Average confidence: {stats['avg_confidence']:.3f}")
            print(f"      - High confidence detections: {stats['high_confidence_count']}")
        else:
            print("   No data in database yet")
        
        # Get recent detections
        print("\n5. Retrieving recent detections...")
        recent = db.get_recent_detections(limit=5)
        
        if recent:
            print(f"   Recent detections ({len(recent)}):")
            for i, detection in enumerate(recent):
                print(f"      {i+1}. {detection['timestamp']}: {detection['plate_text']} "
                      f"(conf: {detection['confidence']:.3f})")
        else:
            print("   No recent detections")
        
        # Get high confidence detections
        print("\n6. Finding high confidence detections...")
        high_conf = db.get_high_confidence_detections(min_confidence=0.8)
        
        if high_conf:
            print(f"   High confidence detections ({len(high_conf)}):")
            for i, detection in enumerate(high_conf):
                print(f"      {i+1}. {detection['plate_text']} (conf: {detection['confidence']:.3f})")
        else:
            print("   No high confidence detections")
        
        # Export to CSV (if there's data)
        if stats and stats['total_detections'] > 0:
            print("\n7. Exporting data to CSV...")
            csv_path = "detections_export.csv"
            success = db.export_to_csv(csv_path)
            
            if success:
                print(f"   Data exported to {csv_path}")
            else:
                print("   Failed to export data")
        
        print("\nDatabase operations example completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 