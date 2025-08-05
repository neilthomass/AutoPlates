#!/usr/bin/env python3
"""
Model Evaluation Script

This script evaluates and compares the trained license plate detection model
against the original Kaggle model to measure improvements.
"""

import os
import sys
import yaml
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import kagglehub


class ModelEvaluator:
    """Evaluator class for comparing license plate detection models."""
    
    def __init__(self, test_images_path: str = "training/data/test/images"):
        """
        Initialize the evaluator.
        
        Args:
            test_images_path: Path to test images
        """
        self.test_images_path = Path(test_images_path)
        self.results = {}
        
    def load_models(self, trained_model_path: str, kaggle_model_path: str = None) -> bool:
        """
        Load both the trained model and the original Kaggle model.
        
        Args:
            trained_model_path: Path to the trained model
            kaggle_model_path: Path to Kaggle model (will download if None)
            
        Returns:
            True if models loaded successfully
        """
        print("Loading models...")
        
        try:
            # Load trained model
            self.trained_model = YOLO(trained_model_path)
            print(f"Trained model loaded: {trained_model_path}")
            
            # Load or download Kaggle model
            if kaggle_model_path is None:
                print("Downloading Kaggle model...")
                kaggle_model_dir = kagglehub.model_download(
                    "harshitsingh09/yolov8-license-plate-detector/pyTorch/default"
                )
                kaggle_model_path = Path(kaggle_model_dir) / "best.pt"
            
            self.kaggle_model = YOLO(kaggle_model_path)
            print(f"Kaggle model loaded: {kaggle_model_path}")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_test_images(self, max_images: int = 50) -> List[Path]:
        """Get list of test images."""
        if not self.test_images_path.exists():
            print(f"Test images path not found: {self.test_images_path}")
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        test_images = []
        
        for ext in image_extensions:
            test_images.extend(self.test_images_path.glob(f"*{ext}"))
            test_images.extend(self.test_images_path.glob(f"*{ext.upper()}"))
        
        # Limit number of images for evaluation
        return test_images[:max_images]
    
    def evaluate_model(self, model: YOLO, model_name: str, test_images: List[Path]) -> Dict[str, Any]:
        """
        Evaluate a single model on test images.
        
        Args:
            model: YOLO model to evaluate
            model_name: Name of the model for results
            test_images: List of test image paths
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {model_name}...")
        
        results = {
            'model_name': model_name,
            'total_images': len(test_images),
            'detections': 0,
            'avg_confidence': 0.0,
            'avg_inference_time': 0.0,
            'detection_details': []
        }
        
        total_confidence = 0.0
        total_time = 0.0
        
        for img_path in test_images:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Run inference
            start_time = time.time()
            predictions = model(img)
            inference_time = time.time() - start_time
            
            # Process results
            if predictions and len(predictions) > 0:
                prediction = predictions[0]
                boxes = prediction.boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Get detections
                    confidences = boxes.conf.cpu().numpy()
                    num_detections = len(confidences)
                    
                    results['detections'] += num_detections
                    total_confidence += np.sum(confidences)
                    
                    # Store detection details
                    for i, conf in enumerate(confidences):
                        results['detection_details'].append({
                            'image': img_path.name,
                            'confidence': float(conf),
                            'inference_time': inference_time
                        })
            
            total_time += inference_time
        
        # Calculate averages
        if results['detections'] > 0:
            results['avg_confidence'] = total_confidence / results['detections']
        
        if len(test_images) > 0:
            results['avg_inference_time'] = total_time / len(test_images)
        
        print(f"  - Total detections: {results['detections']}")
        print(f"  - Average confidence: {results['avg_confidence']:.4f}")
        print(f"  - Average inference time: {results['avg_inference_time']:.4f}s")
        
        return results
    
    def compare_models(self, trained_model_path: str, kaggle_model_path: str = None) -> Dict[str, Any]:
        """
        Compare the trained model against the Kaggle model.
        
        Args:
            trained_model_path: Path to the trained model
            kaggle_model_path: Path to Kaggle model (will download if None)
            
        Returns:
            Dictionary with comparison results
        """
        print("Starting model comparison...")
        
        # Load models
        if not self.load_models(trained_model_path, kaggle_model_path):
            return {}
        
        # Get test images
        test_images = self.get_test_images()
        if not test_images:
            print("No test images found!")
            return {}
        
        print(f"Found {len(test_images)} test images")
        
        # Evaluate both models
        trained_results = self.evaluate_model(self.trained_model, "Trained Model", test_images)
        kaggle_results = self.evaluate_model(self.kaggle_model, "Kaggle Model", test_images)
        
        # Compare results
        comparison = {
            'trained_model': trained_results,
            'kaggle_model': kaggle_results,
            'improvements': {}
        }
        
        # Calculate improvements
        if kaggle_results['detections'] > 0:
            detection_improvement = ((trained_results['detections'] - kaggle_results['detections']) 
                                   / kaggle_results['detections'] * 100)
            comparison['improvements']['detection_rate'] = detection_improvement
        
        if kaggle_results['avg_confidence'] > 0:
            confidence_improvement = ((trained_results['avg_confidence'] - kaggle_results['avg_confidence']) 
                                    / kaggle_results['avg_confidence'] * 100)
            comparison['improvements']['confidence'] = confidence_improvement
        
        if kaggle_results['avg_inference_time'] > 0:
            speed_improvement = ((kaggle_results['avg_inference_time'] - trained_results['avg_inference_time']) 
                               / kaggle_results['avg_inference_time'] * 100)
            comparison['improvements']['speed'] = speed_improvement
        
        # Print comparison summary
        print("\nModel Comparison Summary:")
        print("=" * 50)
        print(f"Trained Model:")
        print(f"  - Detections: {trained_results['detections']}")
        print(f"  - Avg Confidence: {trained_results['avg_confidence']:.4f}")
        print(f"  - Avg Inference Time: {trained_results['avg_inference_time']:.4f}s")
        
        print(f"\nKaggle Model:")
        print(f"  - Detections: {kaggle_results['detections']}")
        print(f"  - Avg Confidence: {kaggle_results['avg_confidence']:.4f}")
        print(f"  - Avg Inference Time: {kaggle_results['avg_inference_time']:.4f}s")
        
        print(f"\nImprovements:")
        for metric, improvement in comparison['improvements'].items():
            print(f"  - {metric.replace('_', ' ').title()}: {improvement:+.2f}%")
        
        return comparison
    
    def visualize_comparison(self, comparison: Dict[str, Any], save_path: str = "training/evaluation_results.png"):
        """
        Create visualization of model comparison.
        
        Args:
            comparison: Comparison results dictionary
            save_path: Path to save the visualization
        """
        print(f"Creating visualization: {save_path}")
        
        trained = comparison['trained_model']
        kaggle = comparison['kaggle_model']
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Detection count comparison
        models = ['Trained', 'Kaggle']
        detections = [trained['detections'], kaggle['detections']]
        ax1.bar(models, detections, color=['blue', 'orange'])
        ax1.set_title('Total Detections')
        ax1.set_ylabel('Number of Detections')
        
        # Average confidence comparison
        confidences = [trained['avg_confidence'], kaggle['avg_confidence']]
        ax2.bar(models, confidences, color=['blue', 'orange'])
        ax2.set_title('Average Confidence')
        ax2.set_ylabel('Confidence Score')
        
        # Inference time comparison
        times = [trained['avg_inference_time'], kaggle['avg_inference_time']]
        ax3.bar(models, times, color=['blue', 'orange'])
        ax3.set_title('Average Inference Time')
        ax3.set_ylabel('Time (seconds)')
        
        # Confidence distribution
        trained_confs = [d['confidence'] for d in trained['detection_details']]
        kaggle_confs = [d['confidence'] for d in kaggle['detection_details']]
        
        if trained_confs and kaggle_confs:
            ax4.hist(trained_confs, alpha=0.7, label='Trained', bins=20, color='blue')
            ax4.hist(kaggle_confs, alpha=0.7, label='Kaggle', bins=20, color='orange')
            ax4.set_title('Confidence Distribution')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {save_path}")
    
    def save_results(self, comparison: Dict[str, Any], save_path: str = "training/evaluation_results.yaml"):
        """
        Save evaluation results to YAML file.
        
        Args:
            comparison: Comparison results dictionary
            save_path: Path to save the results
        """
        print(f"Saving results to: {save_path}")
        
        # Convert numpy types to native Python types for YAML serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def convert_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    convert_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            convert_dict(item)
                else:
                    d[key] = convert_numpy(value)
        
        # Create a copy to avoid modifying the original
        results_copy = comparison.copy()
        convert_dict(results_copy)
        
        with open(save_path, 'w') as f:
            yaml.dump(results_copy, f, default_flow_style=False, indent=2)
        
        print(f"Results saved to: {save_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate license plate detection models")
    parser.add_argument("--trained-model", required=True, help="Path to trained model")
    parser.add_argument("--kaggle-model", help="Path to Kaggle model (will download if not provided)")
    parser.add_argument("--test-images", default="training/data/test/images", 
                       help="Path to test images")
    parser.add_argument("--max-images", type=int, default=50, 
                       help="Maximum number of test images to use")
    parser.add_argument("--save-visualization", default="training/evaluation_results.png",
                       help="Path to save visualization")
    parser.add_argument("--save-results", default="training/evaluation_results.yaml",
                       help="Path to save results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.test_images)
    
    # Run comparison
    comparison = evaluator.compare_models(args.trained_model, args.kaggle_model)
    
    if comparison:
        # Create visualization
        evaluator.visualize_comparison(comparison, args.save_visualization)
        
        # Save results
        evaluator.save_results(comparison, args.save_results)
        
        print("Evaluation completed successfully!")
        return 0
    else:
        print("Evaluation failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 