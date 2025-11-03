import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np
import datetime
from PIL import Image, ImageFile
from Constants import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b1(weights = models.EfficientNet_B1_Weights.DEFAULT)
        
        # Freeze all layers initially
        for param in self.efficientnet.parameters():
            param.requires_grad = False
            
        # Unfreeze last 100 layers
        for param in list(self.efficientnet.parameters())[-100:]:
            param.requires_grad = True
            
        # Replace classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficientnet(x)

class ImagePreprocessTransform:
    def __call__(self, img):
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Resize
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        
        # Stack to 3 channels
        clahe_img = np.stack([clahe_img]*3, axis=-1)
        
        # Convert to float and normalize
        clahe_img = clahe_img.astype(np.float32) / 255.
        
        # Convert to PyTorch tensor and ensure correct channel order
        tensor = torch.from_numpy(clahe_img).permute(2, 0, 1).float()
        return tensor

def create_dataloaders():
    transform = ImagePreprocessTransform()
    
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    test_dataset = ImageFolder(TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def load_latest_checkpoint(model, isModel=False):
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints_torch")
    if isModel:
        ckpt_dir = os.path.join(BASE_DIR, "models_torch")
    if not os.path.exists(ckpt_dir):
        return model
    
    checkpoints = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not checkpoints:
        return model
    
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"Loading weights from {latest_ckpt}")
    model.load_state_dict(torch.load(latest_ckpt))
    return model

def train_model(model, train_loader, test_loader, num_epochs=EPOCHS, device='cuda'):
    os.makedirs(os.path.join(BASE_DIR, 'logs_torch'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'checkpoints_torch'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'models_torch'), exist_ok=True)
    
    curTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(BASE_DIR, 'logs_torch', 'tensorboard', f'smth{curTime}'))

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters())
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    try:
        for epoch in range(INIT_EPOCH, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(test_loader)
            val_acc = correct / total
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/validation', val_acc, epoch)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save checkpoint if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                         os.path.join(BASE_DIR, 'checkpoints_torch', f'efficientnet_{curTime}.pt'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
        # Save final model
        torch.save(model.state_dict(), 
                  os.path.join(BASE_DIR, 'models_torch', 'efficientnet_model_final2.pt'))
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        writer.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_dataloaders()
    model = EfficientNetModel()
    # model = load_latest_checkpoint(model, True)
    
    # Print model summary
    print(model)
    
    train_model(model, train_loader, test_loader, device=device)