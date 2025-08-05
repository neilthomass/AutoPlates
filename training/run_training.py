#!/usr/bin/env python3
"""
Training Runner Script

This script provides a simple interface to run the complete training pipeline
for license plate detection model training.
"""

import os
import sys
import argparse
from pathlib import Path


def run_data_preparation():
    """Run the data preparation step."""
    print("Step 1: Preparing dataset...")
    
    # Add utils to path
    utils_path = Path(__file__).parent / "utils"
    sys.path.append(str(utils_path))
    
    try:
        from data_preparation import DatasetPreparator
        import kagglehub
        
        # Download dataset
        print("Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download(
            "harshitsingh09/license-plate-detection-dataset-anpr-yolo-format"
        )
        
        # Prepare dataset
        preparator = DatasetPreparator(dataset_path, "training/data")
        
        # Analyze dataset
        stats = preparator.analyze_dataset()
        
        # Prepare for training
        success = preparator.prepare_yolo_dataset(
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            seed=42
        )
        
        if success:
            # Validate annotations
            validation_results = preparator.validate_annotations()
            print("Dataset preparation completed successfully!")
            return True
        else:
            print("Dataset preparation failed!")
            return False
            
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return False


def run_training(epochs=None, batch_size=None):
    """Run the model training step."""
    print("Step 2: Training model...")
    
    try:
        # Import training script
        train_script = Path(__file__).parent / "scripts" / "train.py"
        
        # Build command
        cmd = [sys.executable, str(train_script), "--train"]
        
        if epochs:
            cmd.extend(["--epochs", str(epochs)])
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])
        
        # Run training
        import subprocess
        result = subprocess.run(cmd, check=True)
        
        print("Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error in training: {e}")
        return False


def run_evaluation():
    """Run the model evaluation step."""
    print("Step 3: Evaluating model...")
    
    try:
        # Import evaluation script
        eval_script = Path(__file__).parent / "scripts" / "evaluate.py"
        
        # Find the best model
        model_path = Path("training/models/license_plate_detector/weights/best.pt")
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return False
        
        # Build command
        cmd = [
            sys.executable, str(eval_script),
            "--trained-model", str(model_path),
            "--save-visualization", "training/comparison_results.png",
            "--save-results", "training/evaluation_results.yaml"
        ]
        
        # Run evaluation
        import subprocess
        result = subprocess.run(cmd, check=True)
        
        print("Evaluation completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return False


def main():
    """Main training runner function."""
    parser = argparse.ArgumentParser(description="Run license plate detection training pipeline")
    parser.add_argument("--skip-data-prep", action="store_true", 
                       help="Skip data preparation step")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training step")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--config", default="training/configs/training_config.yaml",
                       help="Path to training configuration file")
    
    args = parser.parse_args()
    
    print("License Plate Detection Training Pipeline")
    print("=" * 50)
    
    success = True
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        if not run_data_preparation():
            success = False
            print("Data preparation failed. Stopping pipeline.")
            return 1
    else:
        print("Skipping data preparation...")
    
    # Step 2: Training
    if not args.skip_training and success:
        if not run_training(args.epochs, args.batch_size):
            success = False
            print("Training failed. Stopping pipeline.")
            return 1
    else:
        print("Skipping training...")
    
    # Step 3: Evaluation
    if not args.skip_evaluation and success:
        if not run_evaluation():
            success = False
            print("Evaluation failed.")
            return 1
    else:
        print("Skipping evaluation...")
    
    if success:
        print("\nTraining pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Check training results in training/models/license_plate_detector/")
        print("2. View comparison results in training/comparison_results.png")
        print("3. Review evaluation metrics in training/evaluation_results.yaml")
        print("4. Use the trained model in your detection system")
        return 0
    else:
        print("\nTraining pipeline failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 