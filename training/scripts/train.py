#!/usr/bin/env python3
"""
License Plate Detection Model Training Script

This script trains a YOLOv8 model for license plate detection using
the prepared dataset and configuration files.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO
import kagglehub


class LicensePlateTrainer:
    """Trainer class for license plate detection model."""
    
    def __init__(self, config_path: str = "training/configs/training_config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.dataset_path = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from: {self.config_path}")
        return config
    
    def download_dataset(self) -> bool:
        """Download the license plate dataset from Kaggle."""
        print("Downloading license plate dataset from Kaggle...")
        
        try:
            # Download the dataset
            dataset_path = kagglehub.dataset_download(
                "harshitsingh09/license-plate-detection-dataset-anpr-yolo-format"
            )
            self.dataset_path = Path(dataset_path)
            
            print(f"Dataset downloaded to: {self.dataset_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def prepare_dataset(self, output_path: str = "training/data") -> bool:
        """Prepare the dataset for training."""
        print("Preparing dataset for training...")
        
        try:
            # Import the data preparation utility
            sys.path.append(str(Path(__file__).parent.parent / "utils"))
            from data_preparation import DatasetPreparator
            
            # Initialize preparator
            preparator = DatasetPreparator(str(self.dataset_path), output_path)
            
            # Analyze dataset
            stats = preparator.analyze_dataset()
            
            # Prepare YOLO dataset
            success = preparator.prepare_yolo_dataset(
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                seed=42
            )
            
            if success:
                # Validate annotations
                validation_results = preparator.validate_annotations()
                
                if validation_results['invalid_annotations'] > 0:
                    print(f"Warning: {validation_results['invalid_annotations']} invalid annotations found")
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return False
    
    def initialize_model(self) -> bool:
        """Initialize the YOLOv8 model for training."""
        print("Initializing YOLOv8 model...")
        
        try:
            # Get model configuration
            model_config = self.config['model']
            architecture = model_config['architecture']
            
            # Create model
            self.model = YOLO(f"{architecture}.pt")
            
            print(f"Model initialized: {architecture}")
            return True
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
    
    def train_model(self, dataset_yaml: str = "training/data/dataset.yaml", 
                   epochs: Optional[int] = None, batch_size: Optional[int] = None) -> bool:
        """
        Train the license plate detection model.
        
        Args:
            dataset_yaml: Path to dataset configuration file
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size (overrides config)
            
        Returns:
            True if training completed successfully
        """
        if not self.model:
            print("Model not initialized. Call initialize_model() first.")
            return False
        
        print("Starting model training...")
        
        try:
            # Get training configuration
            train_config = self.config['training']
            save_config = self.config['save']
            
            # Override config with arguments if provided
            if epochs is not None:
                train_config['epochs'] = epochs
            if batch_size is not None:
                train_config['batch_size'] = batch_size
            
            # Prepare training arguments
            train_args = {
                'data': dataset_yaml,
                'epochs': train_config['epochs'],
                'batch': train_config['batch_size'],
                'imgsz': self.config['model']['input_size'],
                'device': train_config['device'],
                'workers': train_config['workers'],
                'project': save_config['project'],
                'name': save_config['name'],
                'exist_ok': save_config['exist_ok'],
                'pretrained': True,
                'optimizer': train_config['optimizer'],
                'verbose': train_config['verbose'],
                'seed': train_config['seed'],
                'deterministic': True,
                'single_cls': True,  # Single class for license plates
                'amp': True,  # Automatic Mixed Precision
                'lr0': train_config['lr0'],
                'lrf': train_config['lrf'],
                'momentum': train_config['momentum'],
                'weight_decay': train_config['weight_decay'],
                'warmup_epochs': train_config['warmup_epochs'],
                'warmup_momentum': train_config['warmup_momentum'],
                'warmup_bias_lr': train_config['warmup_bias_lr'],
                'box': train_config['box'],
                'cls': train_config['cls'],
                'dfl': train_config['dfl'],
                'hsv_h': self.config['augmentation']['hsv_h'],
                'hsv_s': self.config['augmentation']['hsv_s'],
                'hsv_v': self.config['augmentation']['hsv_v'],
                'degrees': self.config['augmentation']['degrees'],
                'translate': self.config['augmentation']['translate'],
                'scale': self.config['augmentation']['scale'],
                'shear': self.config['augmentation']['shear'],
                'perspective': self.config['augmentation']['perspective'],
                'flipud': self.config['augmentation']['flipud'],
                'fliplr': self.config['augmentation']['fliplr'],
                'mosaic': self.config['augmentation']['mosaic'],
                'mixup': self.config['augmentation']['mixup'],
                'copy_paste': self.config['augmentation']['copy_paste'],
                'save_period': self.config['validation']['save_period'],
                'plots': self.config['validation']['plots'],
                'save_metrics': self.config['validation']['save_metrics']
            }
            
            print("Training configuration:")
            print(f"  - Epochs: {train_args['epochs']}")
            print(f"  - Batch size: {train_args['batch']}")
            print(f"  - Image size: {train_args['imgsz']}")
            print(f"  - Device: {train_args['device']}")
            print(f"  - Learning rate: {train_args['lr0']}")
            print(f"  - Optimizer: {train_args['optimizer']}")
            
            # Start training
            results = self.model.train(**train_args)
            
            print("Training completed successfully!")
            print(f"Best model saved to: {results.save_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def validate_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate the trained model on the test set.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Dictionary with validation metrics
        """
        print(f"Validating model: {model_path}")
        
        try:
            # Load the trained model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val()
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1': results.box.map50 * 2 / (results.box.map50 + 1)
            }
            
            print("Validation Results:")
            print(f"  - mAP@0.5: {metrics['mAP50']:.4f}")
            print(f"  - mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error validating model: {e}")
            return {}
    
    def export_model(self, model_path: str, export_formats: list = ['onnx', 'torchscript']) -> bool:
        """
        Export the trained model to different formats.
        
        Args:
            model_path: Path to the trained model
            export_formats: List of formats to export to
            
        Returns:
            True if export successful
        """
        print(f"Exporting model: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            for format_name in export_formats:
                print(f"Exporting to {format_name.upper()}...")
                model.export(format=format_name)
            
            print("Model export completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error exporting model: {e}")
            return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train license plate detection model")
    parser.add_argument("--config", default="training/configs/training_config.yaml", 
                       help="Path to training configuration file")
    parser.add_argument("--dataset-yaml", default="training/data/dataset.yaml",
                       help="Path to dataset configuration file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--download-dataset", action="store_true", 
                       help="Download dataset from Kaggle")
    parser.add_argument("--prepare-dataset", action="store_true",
                       help="Prepare dataset for training")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--validate", help="Path to model for validation")
    parser.add_argument("--export", help="Path to model for export")
    parser.add_argument("--export-formats", nargs="+", default=['onnx', 'torchscript'],
                       help="Formats to export model to")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LicensePlateTrainer(args.config)
    
    try:
        # Download dataset if requested
        if args.download_dataset:
            if not trainer.download_dataset():
                return 1
        
        # Prepare dataset if requested
        if args.prepare_dataset:
            if not trainer.prepare_dataset():
                return 1
        
        # Initialize model
        if not trainer.initialize_model():
            return 1
        
        # Train model if requested
        if args.train:
            if not trainer.train_model(args.dataset_yaml, args.epochs, args.batch_size):
                return 1
        
        # Validate model if requested
        if args.validate:
            trainer.validate_model(args.validate)
        
        # Export model if requested
        if args.export:
            trainer.export_model(args.export, args.export_formats)
        
        print("Training pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 