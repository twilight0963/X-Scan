# X-Scan: Bone Fracture Detection Model

A deep learning model for automated bone fracture detection in X-ray images using EfficientNet-B1 architecture with PyTorch.

## Overview

X-Scan is an AI-powered fracture identification system that uses transfer learning with Google's EfficientNet-B1 to classify X-ray images as either fractured or normal. The model includes Grad-CAM visualization to highlight regions of interest in the X-ray images.

## Model Architecture

- **Base Model**: EfficientNet-B1 (pretrained on ImageNet)
- **Framework**: PyTorch
- **Input Size**: 240x240 pixels
- **Output**: Binary classification (Fracture/Normal)
- **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Architecture Details

```
EfficientNet-B1 (Transfer Learning)
├── Frozen layers: First layers
├── Trainable layers: Last 100 layers
└── Custom Classifier:
    ├── Dropout (p=0.2)
    ├── Linear (1280 → 1)
    └── Sigmoid activation
```

## Features

- **Binary Classification**: Detects presence or absence of bone fractures
- **Grad-CAM Visualization**: Highlights regions the model focuses on for predictions
- **CLAHE Preprocessing**: Enhanced contrast for better feature extraction
- **Early Stopping**: Prevents overfitting with patience-based training
- **TensorBoard Integration**: Real-time training monitoring
- **GUI Application**: User-friendly interface with drag-and-drop support

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Bone_Fracture_Binary_Classification
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GUI application:
```bash
pip install -r requirements_gui.txt
```

## Usage

### Training the Model

```bash
cd src
python EfficientNet_pytorch.py
```

**Training Configuration** (in `src/Constants.py`):
- Image Size: 240x240
- Batch Size: 64
- Epochs: 20
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy (BCE)

### Using the GUI Application

Launch the graphical interface:
```bash
python run_gui.py
```

**GUI Features**:
- Upload X-ray images via button or drag-and-drop
- Real-time AI analysis with progress indication
- Grad-CAM heatmap visualization
- Clear fracture/normal classification results

### Programmatic Inference

```python
import torch
from src.EfficientNet_pytorch import EfficientNetModel, load_latest_checkpoint

# Load model
model = EfficientNetModel()
model = load_latest_checkpoint(model, isModel=True)
model.eval()

# Run inference
# (See gui_app.py for complete preprocessing pipeline)
```

## Dataset Structure

```
src/Datasets/
├── train/
│   ├── fractured/
│   └── normal/
├── val/
│   ├── fractured/
│   └── normal/
└── test/
    ├── fractured/
    └── normal/
```

## Model Files

- **Checkpoints**: `src/checkpoints_torch/` - Training checkpoints (best validation accuracy)
- **Final Models**: `src/models_torch/` - Fully trained models
- **Logs**: `src/logs_torch/` - Training logs and TensorBoard data

## Image Preprocessing Pipeline

1. **Grayscale Conversion**: Convert RGB to grayscale
2. **Resize**: Scale to 240x240 pixels
3. **CLAHE**: Apply adaptive histogram equalization (clipLimit=2.0, tileGridSize=8x8)
4. **Channel Stacking**: Replicate grayscale to 3 channels
5. **Normalization**: Scale pixel values to [0, 1]
6. **Tensor Conversion**: Convert to PyTorch tensor format

## Training Features

- **Transfer Learning**: Leverages EfficientNet-B1 pretrained weights
- **Fine-tuning**: Last 100 layers trainable for domain adaptation
- **Early Stopping**: Stops training after 10 epochs without improvement
- **Checkpoint Saving**: Saves best model based on validation accuracy
- **TensorBoard Logging**: Tracks loss and accuracy metrics

## Performance Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir=src/logs_torch/tensorboard
```

Metrics tracked:
- Training/Validation Loss
- Training/Validation Accuracy
- Per-epoch performance

## Model Output

The model outputs a probability score between 0 and 1:
- **Fracture Detected**: Score indicates fracture probability
- **Normal**: Score indicates normal bone probability

The sigmoid activation ensures smooth probability distribution.

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions in the X-ray that most influenced the model's decision:

- **Red/Yellow regions**: High attention areas
- **Blue/Purple regions**: Low attention areas
- Helps validate model decisions and identify potential issues

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Pillow >= 8.0.0

For GUI:
- PySide6 >= 6.5.0
- Matplotlib >= 3.5.0

## File Structure

```
.
├── src/
│   ├── EfficientNet_pytorch.py    # Model training script
│   ├── gui_app.py                 # GUI application
│   ├── Constants.py               # Configuration constants
│   ├── Datasets/                  # Training/validation/test data
│   ├── checkpoints_torch/         # Training checkpoints
│   ├── models_torch/              # Final trained models
│   └── logs_torch/                # Training logs
├── run_gui.py                     # GUI launcher
├── requirements.txt               # Core dependencies
├── requirements_gui.txt           # GUI dependencies
└── MODEL_README.md               # This file
```

## Technical Details

### Loss Function
Binary Cross-Entropy (BCE) Loss for binary classification

### Optimization
- Optimizer: Adam (adaptive learning rate)
- Learning rate: Default PyTorch Adam settings
- Weight decay: None (dropout used instead)

### Regularization
- Dropout: 0.2 in classifier head
- Early stopping: Patience of 10 epochs
- Transfer learning: Frozen base layers prevent overfitting

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `src/Constants.py`:
```python
BATCH_SIZE = 32  # or 16
```

### Model Not Found
Ensure trained model exists at:
```
src/models_torch/efficientnet_model_final2.pt
```

### Image Loading Errors
Supported formats: JPG, JPEG, PNG, BMP, TIFF
Ensure images are valid X-ray scans

## Future Improvements

- Multi-class classification (fracture types)
- Ensemble models for improved accuracy
- Data augmentation strategies
- Model quantization for faster inference
- Mobile deployment support

## License

[Add your license information here]

## Citation

If you use this model in your research, please cite:
```
[Add citation information]
```

## Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch framework by Meta AI
- CLAHE preprocessing technique

## Contact

[Add contact information]

---

**Note**: This model is for research and educational purposes. Always consult qualified medical professionals for clinical diagnosis.
