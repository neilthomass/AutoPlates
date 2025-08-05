#!/usr/bin/env python3
"""
Unit tests for the LicensePlateDetector class.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main import LicensePlateDetector


class TestLicensePlateDetector(unittest.TestCase):
    """Test cases for LicensePlateDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_plates.db')
        self.detector = LicensePlateDetector(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.db_path, self.db_path)
        self.assertIsNone(self.detector.model)
        self.assertIsNone(self.detector.model_path)
        self.assertIsNone(self.detector.dataset_path)
    
    def test_database_initialization(self):
        """Test that database is properly initialized."""
        # Check that database file was created
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that database has the correct table structure
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if plates table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='plates'
        """)
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        # Check table structure
        cursor.execute("PRAGMA table_info(plates)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        expected_columns = [
            'id', 'timestamp', 'image_path', 'plate_text', 
            'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 
            'bbox_y2', 'created_at'
        ]
        
        for col in expected_columns:
            self.assertIn(col, column_names)
        
        conn.close()
    
    @patch('kagglehub.model_download')
    @patch('kagglehub.dataset_download')
    def test_download_resources(self, mock_dataset_download, mock_model_download):
        """Test resource downloading."""
        # Mock the download functions
        mock_model_download.return_value = '/fake/model/path'
        mock_dataset_download.return_value = '/fake/dataset/path'
        
        # Mock os.path.exists and glob
        with patch('os.path.exists', return_value=True), \
             patch('glob.glob', return_value=['fake_image.png']):
            
            self.detector.download_resources()
            
            # Check that paths were set
            self.assertEqual(self.detector.model_path, '/fake/model/path')
            self.assertEqual(self.detector.dataset_path, '/fake/dataset/path')
    
    @patch('ultralytics.YOLO')
    def test_load_model(self, mock_yolo):
        """Test model loading."""
        # Set up model path
        self.detector.model_path = '/fake/model/path'
        
        # Mock YOLO
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        self.detector.load_model()
        
        # Check that model was loaded
        self.assertEqual(self.detector.model, mock_model)
        mock_yolo.assert_called_once_with('/fake/model/path/best.pt')
    
    def test_load_model_without_path(self):
        """Test that loading model without path raises error."""
        with self.assertRaises(RuntimeError):
            self.detector.load_model()
    
    @patch('glob.glob')
    def test_get_sample_images(self, mock_glob):
        """Test getting sample images."""
        # Mock dataset path and glob
        self.detector.dataset_path = '/fake/dataset/path'
        mock_glob.return_value = ['image1.png', 'image2.png', 'image3.png']
        
        # Mock random.sample
        with patch('random.sample', return_value=['image1.png', 'image2.png']):
            images = self.detector.get_sample_images(num_samples=2, seed=42)
            
            self.assertEqual(len(images), 2)
            self.assertEqual(images, ['image1.png', 'image2.png'])
    
    def test_get_sample_images_without_dataset(self):
        """Test that getting samples without dataset raises error."""
        with self.assertRaises(RuntimeError):
            self.detector.get_sample_images()
    
    @patch('cv2.imread')
    @patch('ultralytics.YOLO')
    def test_detect_plates_in_image(self, mock_yolo, mock_imread):
        """Test plate detection in a single image."""
        # Mock image reading
        mock_imread.return_value = Mock()
        
        # Mock YOLO model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        mock_box = Mock()
        
        # Set up mock chain
        mock_yolo.return_value = mock_model
        mock_model.return_value = [mock_result]
        mock_result.boxes = mock_boxes
        mock_boxes.__len__ = Mock(return_value=1)
        mock_boxes.__getitem__ = Mock(return_value=mock_box)
        mock_box.xyxy = [[100, 200, 300, 400]]
        mock_box.conf = [0.95]
        
        self.detector.model = mock_model
        
        # Test detection
        detections = self.detector.detect_plates_in_image('fake_image.jpg')
        
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection['image_path'], 'fake_image.jpg')
        self.assertEqual(detection['plate_text'], 'PLATE_1')
        self.assertEqual(detection['confidence'], 0.95)
        self.assertEqual(detection['bbox_x1'], 100)
        self.assertEqual(detection['bbox_y1'], 200)
        self.assertEqual(detection['bbox_x2'], 300)
        self.assertEqual(detection['bbox_y2'], 400)
    
    def test_detect_plates_without_model(self):
        """Test that detection without model raises error."""
        with self.assertRaises(RuntimeError):
            self.detector.detect_plates_in_image('fake_image.jpg')


class TestDatabaseOperations(unittest.TestCase):
    """Test cases for database operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_plates.db')
        self.detector = LicensePlateDetector(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_stats_empty(self):
        """Test database statistics when empty."""
        stats = self.detector.get_database_stats()
        
        self.assertEqual(stats['total_detections'], 0)
        self.assertEqual(stats['unique_images'], 0)
        self.assertEqual(stats['average_confidence'], 0.0)
    
    def test_insert_detection(self):
        """Test inserting a detection into database."""
        # Create a mock detection
        detection = {
            'timestamp': '2024-01-01T12:00:00',
            'image_path': '/fake/image.jpg',
            'plate_text': 'ABC123',
            'confidence': 0.95,
            'bbox_x1': 100,
            'bbox_y1': 200,
            'bbox_x2': 300,
            'bbox_y2': 400
        }
        
        # Insert detection
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO plates 
            (timestamp, image_path, plate_text, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection['timestamp'],
            detection['image_path'],
            detection['plate_text'],
            detection['confidence'],
            detection['bbox_x1'],
            detection['bbox_y1'],
            detection['bbox_x2'],
            detection['bbox_y2']
        ))
        
        conn.commit()
        conn.close()
        
        # Check statistics
        stats = self.detector.get_database_stats()
        self.assertEqual(stats['total_detections'], 1)
        self.assertEqual(stats['unique_images'], 1)
        self.assertEqual(stats['average_confidence'], 0.95)


if __name__ == '__main__':
    unittest.main() 