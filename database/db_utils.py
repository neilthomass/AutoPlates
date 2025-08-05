#!/usr/bin/env python3
"""
Database utilities for license plate detection.

This module provides utility functions for querying and managing
the license plate detection database.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class PlateDatabase:
    """Utility class for managing the license plate detection database."""
    
    def __init__(self, db_path: str = "database/plates.db"):
        """
        Initialize the database utility.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute("SELECT * FROM detection_stats")
        stats = cursor.fetchone()
        
        if stats:
            return {
                'total_detections': stats[0],
                'unique_images': stats[1],
                'avg_confidence': round(stats[2], 3) if stats[2] else 0.0,
                'min_confidence': stats[3],
                'max_confidence': stats[4],
                'high_confidence_count': stats[5]
            }
        
        conn.close()
        return {}
    
    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent detections.
        
        Args:
            limit: Maximum number of detections to return
            
        Returns:
            List of recent detection dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, image_path, plate_text, confidence, 
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, created_at
            FROM recent_detections 
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        detections = []
        for row in results:
            detections.append({
                'timestamp': row[0],
                'image_path': row[1],
                'plate_text': row[2],
                'confidence': row[3],
                'bbox_x1': row[4],
                'bbox_y1': row[5],
                'bbox_x2': row[6],
                'bbox_y2': row[7],
                'created_at': row[8]
            })
        
        return detections
    
    def get_high_confidence_detections(self, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """
        Get detections with high confidence scores.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of high-confidence detection dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, image_path, plate_text, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM plates 
            WHERE confidence >= ?
            ORDER BY confidence DESC
        """, (min_confidence,))
        
        results = cursor.fetchall()
        conn.close()
        
        detections = []
        for row in results:
            detections.append({
                'timestamp': row[0],
                'image_path': row[1],
                'plate_text': row[2],
                'confidence': row[3],
                'bbox_x1': row[4],
                'bbox_y1': row[5],
                'bbox_x2': row[6],
                'bbox_y2': row[7]
            })
        
        return detections
    
    def get_detections_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get detections within a date range.
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            List of detection dictionaries within the date range
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, image_path, plate_text, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM plates 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        detections = []
        for row in results:
            detections.append({
                'timestamp': row[0],
                'image_path': row[1],
                'plate_text': row[2],
                'confidence': row[3],
                'bbox_x1': row[4],
                'bbox_y1': row[5],
                'bbox_x2': row[6],
                'bbox_y2': row[7]
            })
        
        return detections
    
    def get_detections_by_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Get all detections for a specific image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of detection dictionaries for the image
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, image_path, plate_text, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM plates 
            WHERE image_path = ?
            ORDER BY confidence DESC
        """, (image_path,))
        
        results = cursor.fetchall()
        conn.close()
        
        detections = []
        for row in results:
            detections.append({
                'timestamp': row[0],
                'image_path': row[1],
                'plate_text': row[2],
                'confidence': row[3],
                'bbox_x1': row[4],
                'bbox_y1': row[5],
                'bbox_x2': row[6],
                'bbox_y2': row[7]
            })
        
        return detections
    
    def delete_old_detections(self, days_old: int = 30) -> int:
        """
        Delete detections older than specified days.
        
        Args:
            days_old: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM plates WHERE timestamp < ?", (cutoff_date,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def export_to_csv(self, output_path: str, query: str = "SELECT * FROM plates") -> bool:
        """
        Export database results to CSV.
        
        Args:
            output_path: Path to output CSV file
            query: SQL query to execute
            
        Returns:
            True if export was successful
        """
        try:
            import csv
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Write to CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                writer.writerows(results)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False


def main():
    """Example usage of the database utilities."""
    db = PlateDatabase()
    
    print("License Plate Detection Database Utilities")
    print("=" * 50)
    
    # Get overall stats
    stats = db.get_stats()
    if stats:
        print(f"Total detections: {stats['total_detections']}")
        print(f"Unique images: {stats['unique_images']}")
        print(f"Average confidence: {stats['avg_confidence']}")
        print(f"High confidence detections: {stats['high_confidence_count']}")
    else:
        print("No data found in database.")
    
    # Get recent detections
    recent = db.get_recent_detections(5)
    if recent:
        print(f"\nRecent detections ({len(recent)}):")
        for detection in recent:
            print(f"  - {detection['timestamp']}: {os.path.basename(detection['image_path'])} "
                  f"-> {detection['plate_text']} (conf: {detection['confidence']:.3f})")
    
    # Get high confidence detections
    high_conf = db.get_high_confidence_detections(0.9)
    if high_conf:
        print(f"\nHigh confidence detections ({len(high_conf)}):")
        for detection in high_conf:
            print(f"  - {detection['plate_text']} (conf: {detection['confidence']:.3f})")


if __name__ == "__main__":
    main() 