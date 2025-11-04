# X-Ray Fracture Detection GUI

A modern GUI application for testing your trained PyTorch bone fracture detection model.

## Features

- **Clean Interface**: Three-panel layout with original image, controls, and heatmap visualization
- **File Upload**: Easy image selection with file picker
- **Real-time Inference**: Run predictions with GradCAM visualization
- **Responsive UI**: Threading keeps the interface responsive during model inference
- **Error Handling**: Graceful error handling for file loading and prediction failures

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_gui.txt
```

2. Run the application:
```bash
python run_gui.py
```

## Usage

1. **Upload Image**: Click "Upload Image" to select an X-ray image (.jpg, .png, .jpeg)
2. **Run Prediction**: Click "Run Prediction" to analyze the image
3. **View Results**: 
   - Original image displays in the center panel
   - GradCAM heatmap shows in the right panel
   - Prediction result (FRACTURED/NORMAL) with confidence score

## Model Requirements

The GUI expects the PyTorch model file at:
```
src/models_torch/efficientnet_model_final2.pt
```

The model should be an EfficientNet-B1 with:
- Input: 224x224 RGB images
- Output: Single sigmoid value (fracture probability)
- Preprocessing: CLAHE enhancement on grayscale conversion

## Technical Details

- **Framework**: PySide6 (Qt6) for modern UI
- **Threading**: Model inference runs in separate thread
- **GradCAM**: Visualizes model attention areas
- **CPU Support**: Runs on CPU-only systems
- **Image Formats**: Supports JPG, PNG, JPEG, BMP, TIFF