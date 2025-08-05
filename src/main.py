#!/usr/bin/env python3
"""
YOLOv8 License Plate Detector - Real-time Camera System

This script provides real-time license plate detection using a camera feed.
It detects license plates continuously and stores results in SQL database
with duplicate prevention for plates detected within a few seconds.

Author: Your Name
Date: 2024
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import sqlite3
import json
import time
import threading
from collections import defaultdict

# YOLO and KaggleHub
import kagglehub
from ultralytics import YOLO


class RealTimeLicensePlateDetector:
    """Real-time license plate detector with camera feed and duplicate prevention."""
    
    def __init__(self, model_name: str = "harshitsingh09/yolov8-license-plate-detector/pyTorch/default",
                 dataset_name: str = "harshitsingh09/license-plate-detection-dataset-anpr-yolo-format",
                 db_path: str = "database/plates.db",
                 camera_id: int = 0,
                 duplicate_window: int = 5):
        """
        Initialize the real-time license plate detector.
        
        Args:
            model_name: Kaggle model identifier
            dataset_name: Kaggle dataset identifier
            db_path: Path to SQLite database
            camera_id: Camera device ID (usually 0 for default camera)
            duplicate_window: Time window in seconds to prevent duplicates
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.db_path = db_path
        self.camera_id = camera_id
        self.duplicate_window = duplicate_window
        self.model = None
        self.model_path = None
        self.dataset_path = None
        self.camera = None
        self.is_running = False
        self.recent_plates = defaultdict(list)  # Track recent plates by text
        
        # Initialize database
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize the SQLite database with the required table."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create the plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                plate_text TEXT,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at: {self.db_path}")
        
    def download_resources(self) -> None:
        """Download the model and dataset from Kaggle."""
        print("Downloading resources from Kaggle...")
        
        try:
            # Download model
            self.model_path = kagglehub.model_download(self.model_name)
            model_file = os.path.join(self.model_path, "best.pt")
            assert os.path.exists(model_file), "Model file not found after download."
            print(f"Model loaded from: {model_file}")
            
            # Download dataset (for validation only)
            self.dataset_path = kagglehub.dataset_download(self.dataset_name)
            val_glob_path = os.path.join(self.dataset_path, "YOLO_dataset/images/val", "*.png")
            val_images = glob(val_glob_path)
            assert len(val_images) > 0, "No validation images found in dataset."
            print(f"Dataset loaded from: {self.dataset_path}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download/load resources: {e}")
    
    def load_model(self) -> None:
        """Load the YOLOv8 model."""
        if not self.model_path:
            raise RuntimeError("Model path not set. Call download_resources() first.")
        
        model_file = os.path.join(self.model_path, "best.pt")
        self.model = YOLO(model_file)
        print("YOLOv8 model loaded successfully")
    
    def initialize_camera(self) -> bool:
        """Initialize the camera feed."""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                print(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def is_duplicate_plate(self, plate_text: str) -> bool:
        """
        Check if a plate was recently detected to prevent duplicates.
        
        Args:
            plate_text: The license plate text to check
            
        Returns:
            True if this is a duplicate within the time window
        """
        current_time = time.time()
        
        # Clean up old entries
        cutoff_time = current_time - self.duplicate_window
        self.recent_plates[plate_text] = [
            t for t in self.recent_plates[plate_text] if t > cutoff_time
        ]
        
        # Check if this plate was recently detected
        if self.recent_plates[plate_text]:
            return True
        
        # Add current detection time
        self.recent_plates[plate_text].append(current_time)
        return False
    
    def detect_plates_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in a camera frame.
        
        Args:
            frame: Camera frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if not self.model:
            return []
        
        results = []
        
        try:
            # Run inference on the frame
            model_results = self.model(frame)
            boxes = getattr(model_results[0], "boxes", None)

            if not boxes or len(boxes) == 0:
                return results

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                
                # For now, we'll use a placeholder for plate text
                # In a real application, you'd use OCR here
                plate_text = f"PLATE_{i+1}_{int(time.time())}"
                
                # Check for duplicates
                if self.is_duplicate_plate(plate_text):
                    continue
                
                detection = {
                    'plate_text': plate_text,
                    'confidence': confidence,
                    'bbox_x1': x1,
                    'bbox_y1': y1,
                    'bbox_x2': x2,
                    'bbox_y2': y2,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(detection)

        except Exception as e:
            print(f"Error detecting plates in frame: {e}")
        
        return results
    
    def save_detection_to_db(self, detection: Dict[str, Any]) -> None:
        """
        Save a detection to the database.
        
        Args:
            detection: Detection dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO plates 
                (timestamp, plate_text, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection['timestamp'],
                detection['plate_text'],
                detection['confidence'],
                detection['bbox_x1'],
                detection['bbox_y1'],
                detection['bbox_x2'],
                detection['bbox_y2']
            ))
            
            conn.commit()
            conn.close()
            
            print(f"Saved detection: {detection['plate_text']} (conf: {detection['confidence']:.3f})")
            
        except Exception as e:
            print(f"Error saving detection to database: {e}")
    
    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection boxes and labels on the frame.
        
        Args:
            frame: Camera frame
            detections: List of detections
            
        Returns:
            Frame with detections drawn
        """
        for detection in detections:
            x1, y1, x2, y2 = (
                detection['bbox_x1'], detection['bbox_y1'],
                detection['bbox_x2'], detection['bbox_y2']
            )
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['plate_text']} ({detection['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def run_camera_detection(self) -> None:
        """Run continuous camera detection."""
        if not self.camera:
            print("Camera not initialized")
            return
        
        print("Starting real-time license plate detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    continue
                
                # Detect plates in frame
                detections = self.detect_plates_in_frame(frame)
                
                # Save new detections to database
                for detection in detections:
                    self.save_detection_to_db(detection)
                
                # Draw detections on frame
                frame_with_detections = self.draw_detections_on_frame(frame, detections)
                
                # Add status text
                status_text = f"Detections: {len(detections)} | Press 'q' to quit"
                cv2.putText(frame_with_detections, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('License Plate Detection', frame_with_detections)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopping detection...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in camera detection: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the database.
        
        Returns:
            Dictionary with database statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM plates")
        total_detections = cursor.fetchone()[0]
        
        # Get unique plates
        cursor.execute("SELECT COUNT(DISTINCT plate_text) FROM plates")
        unique_plates = cursor.fetchone()[0]
        
        # Get average confidence
        cursor.execute("SELECT AVG(confidence) FROM plates")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Get recent detections
        cursor.execute("""
            SELECT timestamp, plate_text, confidence 
            FROM plates 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_detections = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_detections': total_detections,
            'unique_plates': unique_plates,
            'average_confidence': round(avg_confidence, 3),
            'recent_detections': recent_detections
        }
    
    def display_database_results(self) -> None:
        """Display database statistics and recent results."""
        stats = self.get_database_stats()
        
        print("\nDatabase Statistics:")
        print(f"   - Total detections: {stats['total_detections']}")
        print(f"   - Unique plates: {stats['unique_plates']}")
        print(f"   - Average confidence: {stats['average_confidence']}")
        
        if stats['recent_detections']:
            print("\nRecent Detections:")
            for timestamp, plate_text, confidence in stats['recent_detections']:
                print(f"   - {timestamp}: {plate_text} (conf: {confidence:.3f})")


def main():
    """Main function to run the real-time license plate detection."""
    print("YOLOv8 Real-time License Plate Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = RealTimeLicensePlateDetector(
        camera_id=0,  # Use default camera
        duplicate_window=5  # Prevent duplicates within 5 seconds
    )
    
    try:
        # Download resources
        detector.download_resources()
        
        # Load model
        detector.load_model()
        
        # Initialize camera
        if not detector.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return 1
        
        # Run camera detection
        detector.run_camera_detection()
        
        # Display final statistics
        detector.display_database_results()
        
        print("\nReal-time detection completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        detector.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())
