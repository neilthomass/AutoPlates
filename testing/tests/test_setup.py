#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import cv2
        print("OpenCV imported successfully")
    except ImportError as e:
        print(f"OpenCV import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("Matplotlib imported successfully")
    except ImportError as e:
        print(f"Matplotlib import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("NumPy imported successfully")
    except ImportError as e:
        print(f"NumPy import failed: {e}")
        return False
    
    try:
        import kagglehub
        print("KaggleHub imported successfully")
    except ImportError as e:
        print(f"KaggleHub import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"Ultralytics import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("tqdm imported successfully")
    except ImportError as e:
        print(f"tqdm import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("Pillow imported successfully")
    except ImportError as e:
        print(f"Pillow import failed: {e}")
        return False
    
    try:
        import sqlite3
        print("SQLite3 imported successfully")
    except ImportError as e:
        print(f"SQLite3 import failed: {e}")
        return False
    
    return True

def test_main_module():
    """Test that the main module can be imported."""
    try:
        import sys
        import os
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
        sys.path.append(src_path)
        from main import LicensePlateDetector
        print("LicensePlateDetector class imported successfully")
        return True
    except ImportError as e:
        print(f"Main module import failed: {e}")
        return False

def test_database_utils():
    """Test that the database utilities can be imported."""
    try:
        import sys
        import os
        # Add database to path
        database_path = os.path.join(os.path.dirname(__file__), '..', '..', 'database')
        sys.path.append(database_path)
        from db_utils import PlateDatabase
        print("PlateDatabase class imported successfully")
        return True
    except ImportError as e:
        print(f"Database utils import failed: {e}")
        return False

def test_folder_structure():
    """Test that the required folders exist."""
    import os
    
    required_folders = ['src', 'database', 'testing']
    missing_folders = []
    
    for folder in required_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print(f" Missing folders: {missing_folders}")
        return False
    else:
        print("All required folders exist")
        return True

def test_testing_structure():
    """Test that the testing folder structure is correct."""
    import os
    
    testing_subfolders = ['testing/tests', 'testing/examples']
    missing_subfolders = []
    
    for subfolder in testing_subfolders:
        if not os.path.exists(subfolder):
            missing_subfolders.append(subfolder)
    
    if missing_subfolders:
        print(f"Missing testing subfolders: {missing_subfolders}")
        return False
    else:
        print("Testing folder structure is correct")
        return True

def main():
    """Run all tests."""
    print("Testing setup...")
    print("=" * 40)
    
    success = True
    
    # Test folder structure
    if not test_folder_structure():
        success = False
    
    # Test testing structure
    if not test_testing_structure():
        success = False
    
    print()
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test main module
    if not test_main_module():
        success = False
    
    print()
    
    # Test database utils
    if not test_database_utils():
        success = False
    
    print()
    print("=" * 40)
    
    if success:
        print(" All tests passed! Setup is working correctly.")
        print("\nYou can now run the main script with:")
        print("  uv run python src/main.py")
        print("\nOr run the test suite with:")
        print("  uv run python testing/run_tests.py")
        print("\nOr test the database utilities with:")
        print("  uv run python database/db_utils.py")
    else:
        print("Some tests failed. Please check the setup.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 