# X-Scan

AI-powered bone fracture detection in X-ray images using EfficientNet-B1.

## Features

- Binary classification (Fracture/Normal)
- Grad-CAM visualization for interpretability
- User-friendly GUI with drag-and-drop support
- CLAHE preprocessing for enhanced contrast
- **Interactive Feedback System**: Doctors can provide feedback with "I Agree" / "I Disagree" buttons
- **Automatic Data Collection**: Disagreed predictions are saved for model improvement

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

### Training the Model

```bash
cd src
.venv/bin/python EfficientNet_pytorch.py
```

Or if virtual environment is activated:
```bash
cd src
python EfficientNet_pytorch.py
```

### Running the GUI

```bash
.venv/bin/python run_gui.py
```

Or if virtual environment is activated:
```bash
source .venv/bin/activate  # Activate first
python run_gui.py
```

## Feedback System

The GUI includes an interactive feedback system for medical professionals:

1. **Upload and Analyze**: Upload an X-ray image and run AI analysis
2. **Review Results**: Check the AI prediction and Grad-CAM visualization
3. **Provide Feedback**: 
   - Click **"I Agree"** if the prediction is correct
   - Click **"I Disagree"** if the prediction is incorrect
4. **Automatic Logging**: 
   - All feedback is logged to `src/feedback_data/feedback_log.json`
   - Disagreed cases are automatically saved to `src/feedback_data/disagreed/`
   - Images are timestamped and labeled with the AI prediction

### Using Feedback Data

Feedback data can be used to:
- Monitor model performance in real-world scenarios
- Identify difficult cases or edge scenarios
- Collect training data for model improvement
- Calculate expert-validated accuracy metrics

See `src/feedback_data/README.md` for detailed information on analyzing and using feedback data.

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
