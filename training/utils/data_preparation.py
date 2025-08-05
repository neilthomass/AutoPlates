#!/usr/bin/env python3
"""
Data Preparation Utilities for License Plate Detection Training

This module provides utilities for preparing and organizing the dataset
for training the license plate detection model.
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm


class DatasetPreparator:
    """Utility class for preparing and organizing training datasets."""
    
    def __init__(self, source_dataset_path: str, output_path: str = "training/data"):
        """
        Initialize the dataset preparator.
        
        Args:
            source_dataset_path: Path to the source dataset
            output_path: Path where prepared data will be saved
        """
        self.source_path = Path(source_dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def analyze_dataset(self) -> Dict:
        """
        Analyze the source dataset structure.
        
        Returns:
            Dictionary with dataset statistics
        """
        print("Analyzing dataset structure...")
        
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'image_formats': set(),
            'annotation_formats': set(),
            'missing_annotations': 0
        }
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.source_path.rglob(f"*{ext}"))
            image_files.extend(self.source_path.rglob(f"*{ext.upper()}"))
        
        stats['total_images'] = len(image_files)
        
        # Analyze image formats
        for img_file in image_files:
            stats['image_formats'].add(img_file.suffix.lower())
        
        # Find annotation files
        annotation_extensions = {'.txt', '.xml', '.json'}
        annotation_files = []
        
        for ext in annotation_extensions:
            annotation_files.extend(self.source_path.rglob(f"*{ext}"))
        
        stats['total_annotations'] = len(annotation_files)
        
        # Check for missing annotations
        for img_file in image_files:
            annotation_file = img_file.with_suffix('.txt')
            if not annotation_file.exists():
                stats['missing_annotations'] += 1
        
        print(f"Dataset Analysis:")
        print(f"  - Total images: {stats['total_images']}")
        print(f"  - Total annotations: {stats['total_annotations']}")
        print(f"  - Missing annotations: {stats['missing_annotations']}")
        print(f"  - Image formats: {list(stats['image_formats'])}")
        
        return stats
    
    def prepare_yolo_dataset(self, train_split: float = 0.8, val_split: float = 0.1, 
                           test_split: float = 0.1, seed: int = 42) -> bool:
        """
        Prepare the dataset for YOLO training.
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            seed: Random seed for reproducibility
            
        Returns:
            True if successful, False otherwise
        """
        print("Preparing YOLO dataset...")
        
        # Set random seed
        random.seed(seed)
        
        # Create output directories
        train_dir = self.output_path / "train"
        val_dir = self.output_path / "val"
        test_dir = self.output_path / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / "images").mkdir(exist_ok=True)
            (dir_path / "labels").mkdir(exist_ok=True)
        
        # Find all image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for ext in image_extensions:
            image_files.extend(self.source_path.rglob(f"*{ext}"))
            image_files.extend(self.source_path.rglob(f"*{ext.upper()}"))
        
        # Filter images that have annotations
        valid_images = []
        for img_file in image_files:
            annotation_file = img_file.with_suffix('.txt')
            if annotation_file.exists():
                valid_images.append(img_file)
        
        print(f"Found {len(valid_images)} valid image-annotation pairs")
        
        if len(valid_images) == 0:
            print("No valid image-annotation pairs found!")
            return False
        
        # Shuffle and split data
        random.shuffle(valid_images)
        
        total = len(valid_images)
        train_count = int(total * train_split)
        val_count = int(total * val_split)
        
        train_images = valid_images[:train_count]
        val_images = valid_images[train_count:train_count + val_count]
        test_images = valid_images[train_count + val_count:]
        
        print(f"Data split:")
        print(f"  - Training: {len(train_images)} images")
        print(f"  - Validation: {len(val_images)} images")
        print(f"  - Testing: {len(test_images)} images")
        
        # Copy files to respective directories
        self._copy_split(train_images, train_dir, "train")
        self._copy_split(val_images, val_dir, "validation")
        self._copy_split(test_images, test_dir, "test")
        
        # Create dataset.yaml
        self._create_dataset_yaml()
        
        print("Dataset preparation completed successfully!")
        return True
    
    def _copy_split(self, image_files: List[Path], output_dir: Path, split_name: str):
        """Copy image and annotation files for a specific split."""
        print(f"Copying {split_name} files...")
        
        for img_file in tqdm(image_files, desc=f"Copying {split_name}"):
            # Copy image
            img_dest = output_dir / "images" / img_file.name
            shutil.copy2(img_file, img_dest)
            
            # Copy annotation
            annotation_file = img_file.with_suffix('.txt')
            if annotation_file.exists():
                annotation_dest = output_dir / "labels" / annotation_file.name
                shutil.copy2(annotation_file, annotation_dest)
    
    def _create_dataset_yaml(self):
        """Create the dataset.yaml file for YOLO training."""
        dataset_config = {
            'path': str(self.output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['license_plate']
        }
        
        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Created dataset configuration: {yaml_path}")
    
    def validate_annotations(self) -> Dict:
        """
        Validate annotation files for YOLO format.
        
        Returns:
            Dictionary with validation results
        """
        print("Validating annotations...")
        
        validation_results = {
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'empty_annotations': 0,
            'format_errors': []
        }
        
        # Find all annotation files
        annotation_files = list(self.output_path.rglob("*.txt"))
        
        for annotation_file in tqdm(annotation_files, desc="Validating annotations"):
            try:
                with open(annotation_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    validation_results['empty_annotations'] += 1
                    continue
                
                valid_lines = 0
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        validation_results['format_errors'].append(
                            f"{annotation_file}:{line_num} - Expected 5 values, got {len(parts)}"
                        )
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= class_id <= 0):  # Only class 0 for license plate
                            validation_results['format_errors'].append(
                                f"{annotation_file}:{line_num} - Invalid class ID: {class_id}"
                            )
                            continue
                        
                        if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                            validation_results['format_errors'].append(
                                f"{annotation_file}:{line_num} - Values out of range [0,1]"
                            )
                            continue
                        
                        valid_lines += 1
                        
                    except ValueError:
                        validation_results['format_errors'].append(
                            f"{annotation_file}:{line_num} - Non-numeric values"
                        )
                        continue
                
                if valid_lines > 0:
                    validation_results['valid_annotations'] += 1
                else:
                    validation_results['invalid_annotations'] += 1
                    
            except Exception as e:
                validation_results['format_errors'].append(
                    f"{annotation_file} - Error reading file: {e}"
                )
                validation_results['invalid_annotations'] += 1
        
        print(f"Validation Results:")
        print(f"  - Valid annotations: {validation_results['valid_annotations']}")
        print(f"  - Invalid annotations: {validation_results['invalid_annotations']}")
        print(f"  - Empty annotations: {validation_results['empty_annotations']}")
        print(f"  - Format errors: {len(validation_results['format_errors'])}")
        
        if validation_results['format_errors']:
            print("Sample format errors:")
            for error in validation_results['format_errors'][:5]:
                print(f"    {error}")
        
        return validation_results


def main():
    """Main function for data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    parser.add_argument("--source", required=True, help="Path to source dataset")
    parser.add_argument("--output", default="training/data", help="Output directory")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate", action="store_true", help="Validate annotations after preparation")
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = DatasetPreparator(args.source, args.output)
    
    # Analyze dataset
    stats = preparator.analyze_dataset()
    
    # Prepare dataset
    success = preparator.prepare_yolo_dataset(
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    if success and args.validate:
        preparator.validate_annotations()
    
    if success:
        print("Dataset preparation completed successfully!")
        return 0
    else:
        print("Dataset preparation failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 