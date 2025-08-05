# License Plate Detection Model Training

This folder contains the complete training pipeline for creating a custom license plate detection model that aims to beat or replicate the performance of the original Kaggle model.

## Overview

The training pipeline includes:
- **Data Preparation**: Utilities to organize and prepare the dataset
- **Model Training**: YOLOv8-based training with customizable configurations
- **Model Evaluation**: Comparison tools to measure improvements over the original model
- **Configuration Management**: YAML-based configuration for easy customization

## Folder Structure

```
training/
├── configs/
│   └── training_config.yaml    # Training configuration
├── data/
│   ├── dataset.yaml           # Dataset configuration
│   ├── train/                 # Training data (auto-generated)
│   ├── val/                   # Validation data (auto-generated)
│   └── test/                  # Test data (auto-generated)
├── models/                    # Trained models will be saved here
├── scripts/
│   ├── train.py              # Main training script
│   └── evaluate.py           # Model evaluation script
├── utils/
│   └── data_preparation.py   # Dataset preparation utilities
└── README.md                 # This file
```

## Quick Start

### 1. Prepare the Dataset

First, download and prepare the dataset:

```bash
# Download dataset and prepare for training
uv run python training/scripts/train.py --download-dataset --prepare-dataset
```

### 2. Train the Model

Train a custom model with default settings:

```bash
# Train with default configuration
uv run python training/scripts/train.py --train
```

Or customize training parameters:

```bash
# Train with custom parameters
uv run python training/scripts/train.py --train --epochs 150 --batch-size 32
```

### 3. Evaluate and Compare

Compare your trained model against the original Kaggle model:

```bash
# Evaluate the trained model
uv run python training/scripts/evaluate.py --trained-model training/models/license_plate_detector/weights/best.pt
```

## Detailed Usage

### Data Preparation

The data preparation process:
1. Downloads the license plate dataset from Kaggle
2. Organizes images and annotations into train/val/test splits
3. Validates annotation format and quality
4. Creates YAML configuration for YOLO training

```bash
# Manual data preparation
uv run python training/utils/data_preparation.py --source /path/to/dataset --output training/data --validate
```

### Training Configuration

The training configuration is defined in `training/configs/training_config.yaml`. Key settings:

```yaml
# Model Configuration
model:
  architecture: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  input_size: [640, 640]
  num_classes: 1

# Training Configuration
training:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  optimizer: "SGD"

# Augmentation Configuration
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  fliplr: 0.5
  mosaic: 1.0
```

### Advanced Training

#### Custom Training Parameters

```bash
# Train with specific configuration
uv run python training/scripts/train.py \
  --config training/configs/training_config.yaml \
  --dataset-yaml training/data/dataset.yaml \
  --epochs 200 \
  --batch-size 8 \
  --train
```

#### Resume Training

```bash
# Resume from checkpoint
uv run python training/scripts/train.py \
  --train \
  --resume training/models/license_plate_detector/weights/last.pt
```

#### Multi-GPU Training

```bash
# Train on multiple GPUs
uv run python training/scripts/train.py \
  --train \
  --device 0,1,2,3
```

### Model Evaluation

The evaluation script provides comprehensive comparison metrics:

```bash
# Basic evaluation
uv run python training/scripts/evaluate.py \
  --trained-model training/models/license_plate_detector/weights/best.pt

# Custom evaluation
uv run python training/scripts/evaluate.py \
  --trained-model training/models/license_plate_detector/weights/best.pt \
  --test-images training/data/test/images \
  --max-images 100 \
  --save-visualization training/comparison_results.png \
  --save-results training/evaluation_results.yaml
```

## Training Pipeline

### Step 1: Dataset Preparation

```python
from training.utils.data_preparation import DatasetPreparator

# Initialize preparator
preparator = DatasetPreparator("path/to/source/dataset", "training/data")

# Analyze dataset
stats = preparator.analyze_dataset()

# Prepare for YOLO training
success = preparator.prepare_yolo_dataset(
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    seed=42
)

# Validate annotations
validation_results = preparator.validate_annotations()
```

### Step 2: Model Training

```python
from training.scripts.train import LicensePlateTrainer

# Initialize trainer
trainer = LicensePlateTrainer("training/configs/training_config.yaml")

# Download dataset
trainer.download_dataset()

# Prepare dataset
trainer.prepare_dataset()

# Initialize model
trainer.initialize_model()

# Train model
success = trainer.train_model(
    dataset_yaml="training/data/dataset.yaml",
    epochs=100,
    batch_size=16
)
```

### Step 3: Model Evaluation

```python
from training.scripts.evaluate import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator("training/data/test/images")

# Compare models
comparison = evaluator.compare_models(
    trained_model_path="training/models/license_plate_detector/weights/best.pt"
)

# Create visualization
evaluator.visualize_comparison(comparison, "training/comparison.png")

# Save results
evaluator.save_results(comparison, "training/results.yaml")
```

## Configuration Options

### Model Architecture

Choose from different YOLOv8 variants:
- `yolov8n`: Nano (fastest, smallest)
- `yolov8s`: Small
- `yolov8m`: Medium
- `yolov8l`: Large
- `yolov8x`: Extra Large (slowest, most accurate)

### Training Parameters

Key parameters to tune:
- **Learning Rate**: Start with 0.01, reduce if training is unstable
- **Batch Size**: Increase for better GPU utilization, decrease if out of memory
- **Epochs**: More epochs for better performance, but risk overfitting
- **Image Size**: Larger images for better accuracy, smaller for speed

### Augmentation Settings

Data augmentation helps improve model robustness:
- **HSV Augmentation**: Color variations
- **Geometric Augmentation**: Rotation, scaling, translation
- **Mosaic Augmentation**: Multi-image composition
- **Mixup**: Image blending for regularization

## Performance Optimization

### Hardware Requirements

**Minimum Requirements:**
- GPU: 4GB VRAM (GTX 1060 or equivalent)
- RAM: 8GB
- Storage: 10GB free space

**Recommended Requirements:**
- GPU: 8GB+ VRAM (RTX 3070 or better)
- RAM: 16GB+
- Storage: 50GB+ free space

### Training Tips

1. **Start Small**: Begin with yolov8n and small batch size
2. **Monitor Metrics**: Watch for overfitting (validation loss increasing)
3. **Use Mixed Precision**: Enable AMP for faster training
4. **Regular Checkpoints**: Save models every 10-20 epochs
5. **Early Stopping**: Stop when validation metrics plateau

### Troubleshooting

**Common Issues:**

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size
   --batch-size 4
   
   # Reduce image size
   --imgsz 416
   ```

2. **Poor Training Performance**
   ```bash
   # Increase learning rate
   --lr0 0.02
   
   # Use larger model
   --model yolov8s
   ```

3. **Overfitting**
   ```bash
   # Increase augmentation
   --augment
   
   # Reduce epochs
   --epochs 50
   ```

## Expected Results

### Performance Targets

**Baseline (Kaggle Model):**
- mAP@0.5: ~0.85
- Average Confidence: ~0.75
- Inference Time: ~50ms

**Target Improvements:**
- mAP@0.5: >0.90 (+5%)
- Average Confidence: >0.80 (+5%)
- Inference Time: <45ms (-10%)

### Training Time Estimates

| Model Size | Epochs | GPU | Estimated Time |
|------------|--------|-----|----------------|
| yolov8n    | 100    | GTX 1060 | 2-4 hours |
| yolov8s    | 100    | RTX 3070 | 4-6 hours |
| yolov8m    | 100    | RTX 3080 | 6-8 hours |
| yolov8l    | 100    | RTX 3090 | 8-12 hours |

## Integration with Main Project

After training, integrate your custom model:

```python
# Update the main detection script
from src.main import RealTimeLicensePlateDetector

# Use custom trained model
detector = RealTimeLicensePlateDetector(
    model_path="training/models/license_plate_detector/weights/best.pt"
)
```

## Monitoring Training

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir training/models/license_plate_detector
```

### Training Logs

Training logs are saved in:
- `training/models/license_plate_detector/results.csv`
- `training/models/license_plate_detector/weights/`
- `training/models/license_plate_detector/plots/`

## Contributing

To improve the training pipeline:

1. **Experiment with Configurations**: Try different model architectures and hyperparameters
2. **Add Custom Augmentations**: Implement domain-specific augmentations
3. **Improve Evaluation**: Add more comprehensive metrics
4. **Optimize Performance**: Profile and optimize training speed

## License

This training pipeline is part of the license plate detection project. See the main project license for details. 