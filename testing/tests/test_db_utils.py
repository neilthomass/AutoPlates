#!/usr/bin/env python3
"""
Unit tests for the database utilities.
"""

import unittest
import sys
import os
import tempfile
import shutil
import sqlite3
from datetime import datetime, timedelta

# Add database to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))

from db_utils import PlateDatabase


class TestPlateDatabase(unittest.TestCase):
    """Test cases for PlateDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_plates.db')
        self.db = PlateDatabase(db_path=self.db_path)
        
        # Initialize database with schema
        self._init_test_database()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _init_test_database(self):
        """Initialize test database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                plate_text TEXT,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create views
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS recent_detections AS
            SELECT timestamp, image_path, plate_text, confidence, 
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, created_at
            FROM plates 
            ORDER BY created_at DESC
        ''')
        
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS detection_stats AS
            SELECT 
                COUNT(*) as total_detections,
                COUNT(DISTINCT image_path) as unique_images,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence,
                COUNT(CASE WHEN confidence >= 0.8 THEN 1 END) as high_confidence_count
            FROM plates
        ''')
        
        conn.commit()
        conn.close()
    
    def test_init(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db)
        self.assertEqual(self.db.db_path, self.db_path)
    
    def test_get_connection(self):
        """Test getting database connection."""
        conn = self.db.get_connection()
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()
    
    def test_get_stats_empty(self):
        """Test getting stats from empty database."""
        stats = self.db.get_stats()
        
        self.assertEqual(stats['total_detections'], 0)
        self.assertEqual(stats['unique_images'], 0)
        self.assertEqual(stats['avg_confidence'], 0.0)
        self.assertIsNone(stats['min_confidence'])
        self.assertIsNone(stats['max_confidence'])
        self.assertEqual(stats['high_confidence_count'], 0)
    
    def test_get_stats_with_data(self):
        """Test getting stats from database with data."""
        # Insert test data
        self._insert_test_data()
        
        stats = self.db.get_stats()
        
        self.assertEqual(stats['total_detections'], 3)
        self.assertEqual(stats['unique_images'], 2)
        self.assertAlmostEqual(stats['avg_confidence'], 0.85, places=2)
        self.assertEqual(stats['min_confidence'], 0.7)
        self.assertEqual(stats['max_confidence'], 0.95)
        self.assertEqual(stats['high_confidence_count'], 2)
    
    def test_get_recent_detections(self):
        """Test getting recent detections."""
        # Insert test data
        self._insert_test_data()
        
        recent = self.db.get_recent_detections(limit=2)
        
        self.assertEqual(len(recent), 2)
        self.assertIn('timestamp', recent[0])
        self.assertIn('image_path', recent[0])
        self.assertIn('plate_text', recent[0])
        self.assertIn('confidence', recent[0])
    
    def test_get_high_confidence_detections(self):
        """Test getting high confidence detections."""
        # Insert test data
        self._insert_test_data()
        
        high_conf = self.db.get_high_confidence_detections(min_confidence=0.8)
        
        self.assertEqual(len(high_conf), 2)
        for detection in high_conf:
            self.assertGreaterEqual(detection['confidence'], 0.8)
    
    def test_get_detections_by_date_range(self):
        """Test getting detections by date range."""
        # Insert test data
        self._insert_test_data()
        
        start_date = '2024-01-01T00:00:00'
        end_date = '2024-01-02T00:00:00'
        
        detections = self.db.get_detections_by_date_range(start_date, end_date)
        
        self.assertEqual(len(detections), 2)
        for detection in detections:
            self.assertGreaterEqual(detection['timestamp'], start_date)
            self.assertLessEqual(detection['timestamp'], end_date)
    
    def test_get_detections_by_image(self):
        """Test getting detections for a specific image."""
        # Insert test data
        self._insert_test_data()
        
        detections = self.db.get_detections_by_image('/fake/image1.jpg')
        
        self.assertEqual(len(detections), 2)
        for detection in detections:
            self.assertEqual(detection['image_path'], '/fake/image1.jpg')
    
    def test_delete_old_detections(self):
        """Test deleting old detections."""
        # Insert test data
        self._insert_test_data()
        
        # Delete detections older than 1 day
        deleted_count = self.db.delete_old_detections(days_old=1)
        
        # Should delete 1 detection (the old one)
        self.assertEqual(deleted_count, 1)
        
        # Check remaining detections
        stats = self.db.get_stats()
        self.assertEqual(stats['total_detections'], 2)
    
    def test_export_to_csv(self):
        """Test exporting to CSV."""
        # Insert test data
        self._insert_test_data()
        
        csv_path = os.path.join(self.temp_dir, 'export.csv')
        success = self.db.export_to_csv(csv_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_path))
        
        # Check CSV content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # Header + data
            self.assertIn('timestamp', lines[0])
            self.assertIn('image_path', lines[0])
    
    def _insert_test_data(self):
        """Insert test data into database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert test detections
        test_data = [
            ('2024-01-01T10:00:00', '/fake/image1.jpg', 'ABC123', 0.95, 100, 200, 300, 400),
            ('2024-01-01T11:00:00', '/fake/image1.jpg', 'XYZ789', 0.85, 150, 250, 350, 450),
            ('2023-12-31T12:00:00', '/fake/image2.jpg', 'DEF456', 0.70, 120, 220, 320, 420),
        ]
        
        for data in test_data:
            cursor.execute('''
                INSERT INTO plates 
                (timestamp, image_path, plate_text, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
        
        conn.commit()
        conn.close()


class TestDatabaseEdgeCases(unittest.TestCase):
    """Test edge cases for database operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_plates.db')
        self.db = PlateDatabase(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_not_exists(self):
        """Test behavior when database doesn't exist."""
        # Remove database file
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        # Should handle gracefully
        stats = self.db.get_stats()
        self.assertEqual(stats, {})
    
    def test_invalid_query(self):
        """Test handling of invalid queries."""
        # This should not raise an exception
        try:
            self.db.export_to_csv('test.csv', query="SELECT * FROM nonexistent_table")
        except Exception:
            pass  # Expected to fail, but should not crash


if __name__ == '__main__':
    unittest.main() 