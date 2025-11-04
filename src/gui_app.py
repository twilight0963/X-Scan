import sys
import os
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve, QUrl
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor, QDragEnterEvent, QDropEvent

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io


class DragDropFrame(QFrame):
    """Custom frame that accepts drag and drop for images"""
    file_dropped = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.drag_active = False
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            # Check if any of the URLs are image files
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        event.acceptProposedAction()
                        self.drag_active = True
                        self.update_drag_style(True)
                        return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        self.drag_active = False
        self.update_drag_style(False)
        event.accept()
    
    def dropEvent(self, event: QDropEvent):
        self.drag_active = False
        self.update_drag_style(False)
        
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        self.file_dropped.emit(file_path)
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def update_drag_style(self, is_dragging):
        if is_dragging:
            self.setStyleSheet("""
                QFrame {
                    background-color: #e3f2fd;
                    border-radius: 12px;
                    border: 3px dashed #2196f3;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border-radius: 12px;
                    border: 2px solid #e9ecef;
                }
            """)


class ModelInferenceThread(QThread):
    """Thread for running model inference to keep UI responsive"""
    finished = Signal(np.ndarray, bool, float)  # heatmap, prediction, confidence
    error = Signal(str)
    
    def __init__(self, image_path, model_path):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
    
    def run(self):
        try:
            # Load model and run inference
            model = self.load_pytorch_model()
            last_conv_layer = self.get_last_conv_layer(model)
            
            # Process image
            img_tensor, original_image = self.preprocess_image_pytorch(self.image_path)
            
            # Generate heatmap
            heatmap = self.generate_gradcam(model, img_tensor, last_conv_layer, target_class=1)
            
            # Get prediction
            with torch.no_grad():
                prediction = model(img_tensor)
                raw_output = prediction.item()
                predicted_class = self.sigmoid(raw_output)
                
                # Based on the original sigmoid logic analysis:
                # Low raw_output (e.g., 0.2) -> Fracture (True)
                # High raw_output (e.g., 0.8) -> Normal (False)
                # This means raw_output is actually "probability of normal"
                
                # So the fracture probability is 1 - raw_output
                fracture_probability = 1 - raw_output
                confidence = fracture_probability if predicted_class else raw_output
            
            self.finished.emit(heatmap, predicted_class, confidence)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def load_pytorch_model(self):
        """Load the PyTorch model"""
        class EfficientNetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.efficientnet = models.efficientnet_b1(weights=None)
                num_ftrs = self.efficientnet.classifier[1].in_features
                self.efficientnet.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(num_ftrs, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.efficientnet(x)
        
        model = EfficientNetWrapper()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def get_last_conv_layer(self, model):
        """Get the last convolutional layer"""
        last_conv_layer = None
        for name, module in reversed(list(model.efficientnet.named_modules())):
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
                break
        return last_conv_layer
    
    def generate_gradcam(self, model, img_tensor, target_layer, target_class=None):
        """Generate GradCAM heatmap"""
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        model.zero_grad()
        output = model(img_tensor)
        
        if target_class is None:
            target_class = 1 if output.item() > 0.5 else 0
        
        if target_class == 0:
            pred = output
        else:
            pred = 1 - output
        
        # Backward pass
        pred.backward()
        
        # Get gradients and activations
        gradients = gradients[0]
        activations = activations[0]
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[2, 3])
        
        # Weight channels by gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[0, i]
            
        # Generate heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.detach().cpu().numpy()
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Clean up hooks
        handle_fwd.remove()
        handle_bwd.remove()
        
        return heatmap
    
    def preprocess_image_pytorch(self, image_path, target_size=(224, 224)):
        """Preprocess image for PyTorch model"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_size)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        
        # Stack to 3 channels
        clahe_img = np.stack([clahe_img]*3, axis=-1)
        
        # Convert to float and normalize
        clahe_img = clahe_img.astype(np.float32) / 255.
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(clahe_img).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0)
        
        return tensor, image
    
    def sigmoid(self, x):
        """Determine prediction class using the exact original logic from notebook"""
        # Original notebook logic - keeping it exactly the same
        if abs(1-x) > abs(x):
            return True
        else:
            return False


class XRayAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.model_path = None
        self.inference_thread = None
        
        self.init_ui()
        self.setup_model_path()
        self.apply_modern_styling()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("X-Ray Fracture Detection - AI Model Tester")
        self.setMinimumSize(1400, 800)
        self.setGeometry(100, 100, 1400, 800)
        
        # Set application palette for modern look
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 249, 250))
        palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
        self.setPalette(palette)
        
        # Central widget with padding
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #f8f9fa;")
        
        # Main layout with margins
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # Create three equal panels
        self.setup_control_panel(main_layout)
        self.setup_input_panel(main_layout)
        self.setup_output_panel(main_layout)
    
    def setup_control_panel(self, main_layout):
        """Setup control panel with modern styling"""
        # Create frame for the panel
        control_frame = QFrame()
        control_frame.setFixedWidth(400)
        control_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e9ecef;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(30, 30, 30, 30)
        control_layout.setSpacing(20)
        
        # Title
        title = QLabel("Control Panel")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #495057; margin-bottom: 10px;")
        control_layout.addWidget(title)
        
        # Upload button
        self.upload_btn = QPushButton("📁 Upload X-Ray Image")
        self.upload_btn.setMinimumHeight(50)
        self.upload_btn.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.clicked.connect(self.upload_image)
        control_layout.addWidget(self.upload_btn)
        
        # Run prediction button
        self.predict_btn = QPushButton("🔍 Run AI Analysis")
        self.predict_btn.setMinimumHeight(50)
        self.predict_btn.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.predict_btn.setCursor(Qt.PointingHandCursor)
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.run_prediction)
        control_layout.addWidget(self.predict_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                text-align: center;
                background-color: #f8f9fa;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 6px;
            }
        """)
        control_layout.addWidget(self.progress_bar)
        
        # Status section
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #dee2e6;
                padding: 15px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)
        
        status_title = QLabel("Status")
        status_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        status_title.setStyleSheet("color: #6c757d; margin-bottom: 5px;")
        status_layout.addWidget(status_title)
        
        self.status_label = QLabel("Ready to analyze X-ray images")
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #495057; line-height: 1.4;")
        status_layout.addWidget(self.status_label)
        
        control_layout.addWidget(status_frame)
        control_layout.addStretch()
        
        main_layout.addWidget(control_frame)
    
    def setup_input_panel(self, main_layout):
        """Setup input image display panel"""
        # Create frame for the panel with fixed width for alignment
        input_frame = QFrame()
        input_frame.setFixedWidth(400)
        input_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e9ecef;
            }
        """)
        
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(25, 25, 25, 25)
        input_layout.setSpacing(15)
        
        # Title
        title = QLabel("Input Image")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(50)
        title.setStyleSheet("color: #495057; margin-bottom: 10px;")
        input_layout.addWidget(title)
        
        # Image display container with drag & drop
        image_container = DragDropFrame()
        image_container.setFixedSize(350, 350)
        image_container.file_dropped.connect(self.handle_dropped_file)
        
        container_layout = QVBoxLayout(image_container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #6c757d;
                font-size: 12px;
                border: none;
                line-height: 1.4;
            }
        """)
        self.image_label.setText("📁 Upload an X-ray image\n\n🖱️ Click 'Upload Image' button\nor\n📂 Drag & drop image here")
        container_layout.addWidget(self.image_label)
        
        input_layout.addWidget(image_container)
        input_layout.addStretch()
        
        main_layout.addWidget(input_frame)
    
    def setup_output_panel(self, main_layout):
        """Setup output heatmap display panel"""
        # Create frame for the panel with fixed width for alignment
        output_frame = QFrame()
        output_frame.setFixedWidth(400)
        output_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e9ecef;
            }
        """)
        
        output_layout = QVBoxLayout(output_frame)
        output_layout.setContentsMargins(25, 25, 25, 25)
        output_layout.setSpacing(15)
        
        # Title
        title = QLabel("Grad-CAM Output")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(50)
        title.setStyleSheet("color: #495057; margin-bottom: 10px;")
        output_layout.addWidget(title)
        
        # Heatmap display container
        heatmap_container = QFrame()
        heatmap_container.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 12px;
                border: 2px solid #e9ecef;
            }
        """)
        heatmap_container.setFixedSize(350, 350)
        
        container_layout = QVBoxLayout(heatmap_container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        
        # Heatmap display
        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #6c757d;
                font-size: 14px;
                border: none;
            }
        """)
        self.heatmap_label.setText("AI analysis results will appear here")
        container_layout.addWidget(self.heatmap_label)
        
        output_layout.addWidget(heatmap_container)
        
        # Prediction result
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedHeight(80)
        self.result_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                color: #495057;
            }
        """)
        output_layout.addWidget(self.result_label)
        
        main_layout.addWidget(output_frame)
    
    def apply_modern_styling(self):
        """Apply modern styling to buttons and components"""
        # Upload button styling
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background-color: #0056b3;
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        
        # Predict button styling
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                padding: 12px 20px;
            }
            QPushButton:hover:enabled {
                background-color: #1e7e34;
                transform: translateY(-1px);
            }
            QPushButton:pressed:enabled {
                background-color: #155724;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
        """)
    
    def setup_model_path(self):
        """Setup model path"""
        base_dir = Path(__file__).parent
        self.model_path = base_dir / "models_torch" / "efficientnet_model_final2.pt"
        
        if not self.model_path.exists():
            QMessageBox.warning(self, "Model Not Found", 
                              f"Model file not found at: {self.model_path}\n"
                              "Please ensure the model file exists.")
    
    def upload_image(self):
        """Handle image upload via button"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select X-Ray Image", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def handle_dropped_file(self, file_path):
        """Handle image upload via drag & drop"""
        self.load_image(file_path)
    
    def load_image(self, file_path):
        """Common method to load image from either upload button or drag & drop"""
        self.image_path = file_path
        self.display_image(file_path)
        self.predict_btn.setEnabled(True)
        self.status_label.setText(f"✅ Image loaded: {Path(file_path).name}")
        
        # Clear previous results
        self.heatmap_label.setText("AI analysis results will appear here")
        self.heatmap_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #6c757d;
                font-size: 14px;
                border: none;
            }
        """)
        self.result_label.setText("")
    
    def display_image(self, image_path):
        """Display image in the input panel with proper scaling"""
        try:
            pixmap = QPixmap(image_path)
            # Scale to fit the fixed container size (330x330 with 20px padding)
            scaled_pixmap = pixmap.scaled(
                330, 330,
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border: none;
                }
            """)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def run_prediction(self):
        """Run model prediction in a separate thread"""
        if not self.image_path or not self.model_path.exists():
            QMessageBox.warning(self, "Error", "Please upload an image and ensure model exists.")
            return
        
        # Disable button and show progress
        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("🔄 Running AI analysis...")
        
        # Start inference thread
        self.inference_thread = ModelInferenceThread(self.image_path, str(self.model_path))
        self.inference_thread.finished.connect(self.on_prediction_finished)
        self.inference_thread.error.connect(self.on_prediction_error)
        self.inference_thread.start()
    
    def on_prediction_finished(self, heatmap, prediction, confidence):
        """Handle successful prediction completion"""
        try:
            # Create heatmap visualization
            original_image = cv2.imread(self.image_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Create overlay
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(original_rgb)
            ax.imshow(heatmap_resized, cmap='inferno', alpha=0.5)
            ax.axis('off')
            
            # Save to memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close()
            
            # Display heatmap with proper scaling
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            
            # Scale to fit the fixed container size (330x330 with 20px padding)
            scaled_pixmap = pixmap.scaled(
                330, 330,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.heatmap_label.setPixmap(scaled_pixmap)
            self.heatmap_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border: none;
                }
            """)
            
            # Display results with modern styling
            status_icon = "⚠️" if prediction else "✅"
            result_text = f"{status_icon} {'FRACTURE DETECTED' if prediction else 'NO FRACTURE DETECTED'}"
            
            self.result_label.setText(result_text)
            
            # Set modern color scheme based on prediction
            if prediction:
                self.result_label.setStyleSheet("""
                    QLabel {
                        padding: 20px;
                        border-radius: 10px;
                        background-color: #f8d7da;
                        border: 2px solid #dc3545;
                        color: #721c24;
                        font-weight: bold;
                    }
                """)
            else:
                self.result_label.setStyleSheet("""
                    QLabel {
                        padding: 20px;
                        border-radius: 10px;
                        background-color: #d4edda;
                        border: 2px solid #28a745;
                        color: #155724;
                        font-weight: bold;
                    }
                """)
            
            self.status_label.setText("✅ Analysis completed successfully!")
            
        except Exception as e:
            self.on_prediction_error(f"Error displaying results: {str(e)}")
        
        finally:
            # Re-enable controls
            self.predict_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def on_prediction_error(self, error_message):
        """Handle prediction errors"""
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{error_message}")
        self.status_label.setText("❌ Analysis failed! Please try again.")
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)


def main():
    app = QApplication(sys.argv)
    
    # Set modern application style
    app.setStyle('Fusion')
    
    # Set application-wide stylesheet for modern look
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f8f9fa;
        }
        QMessageBox {
            background-color: white;
            border-radius: 10px;
        }
        QFileDialog {
            background-color: white;
        }
    """)
    
    window = XRayAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()