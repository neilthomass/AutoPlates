-- License Plate Detection Database Schema
-- This file contains the SQL schema for storing license plate detection results

-- Create the plates table to store detection results
CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO format timestamp of detection
    image_path TEXT NOT NULL,          -- Path to the source image
    plate_text TEXT,                   -- Detected license plate text (placeholder for now)
    confidence REAL,                   -- Detection confidence score (0.0 to 1.0)
    bbox_x1 INTEGER,                   -- Bounding box top-left x coordinate
    bbox_y1 INTEGER,                   -- Bounding box top-left y coordinate
    bbox_x2 INTEGER,                   -- Bounding box bottom-right x coordinate
    bbox_y2 INTEGER,                   -- Bounding box bottom-right y coordinate
    created_at TEXT DEFAULT CURRENT_TIMESTAMP  -- When this record was inserted
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_plates_timestamp ON plates(timestamp);
CREATE INDEX IF NOT EXISTS idx_plates_image_path ON plates(image_path);
CREATE INDEX IF NOT EXISTS idx_plates_confidence ON plates(confidence);
CREATE INDEX IF NOT EXISTS idx_plates_created_at ON plates(created_at);

-- Create a view for recent detections
CREATE VIEW IF NOT EXISTS recent_detections AS
SELECT 
    timestamp,
    image_path,
    plate_text,
    confidence,
    bbox_x1,
    bbox_y1,
    bbox_x2,
    bbox_y2,
    created_at
FROM plates 
ORDER BY created_at DESC;

-- Create a view for high-confidence detections
CREATE VIEW IF NOT EXISTS high_confidence_detections AS
SELECT 
    timestamp,
    image_path,
    plate_text,
    confidence,
    bbox_x1,
    bbox_y1,
    bbox_x2,
    bbox_y2
FROM plates 
WHERE confidence >= 0.8
ORDER BY confidence DESC;

-- Create a view for detection statistics
CREATE VIEW IF NOT EXISTS detection_stats AS
SELECT 
    COUNT(*) as total_detections,
    COUNT(DISTINCT image_path) as unique_images,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence,
    COUNT(CASE WHEN confidence >= 0.8 THEN 1 END) as high_confidence_count
FROM plates; 