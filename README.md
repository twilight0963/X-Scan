# X-Scan

AI-powered bone fracture detection in X-ray images using EfficientNet-B1.

## Features

- Binary classification (Fracture/Normal)
- Grad-CAM visualization for interpretability
- User-friendly GUI with drag-and-drop support
- CLAHE preprocessing for enhanced contrast

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (recommended) or CPU

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd X-Scan
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/Mac
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

### Step 3: Install Dependencies

For core functionality (training and inference):

```bash
pip install -r requirements.txt
```

For GUI application:

```bash
pip install -r requirements_gui.txt
```

Or install everything at once:

```bash
pip install -r requirements.txt -r requirements_gui.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## Quick Start

```bash
# Train model
cd src && python EfficientNet_pytorch.py

# Run GUI
python run_gui.py
```

## Model

- **Architecture**: EfficientNet-B1 (PyTorch)
- **Input**: 240x240 grayscale X-rays
- **Output**: Binary classification with confidence score
- **Preprocessing**: CLAHE + normalization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV, NumPy, PySide6

See `requirements.txt` for full list.

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Disclaimer**: For research and educational purposes only. Not for clinical use.
