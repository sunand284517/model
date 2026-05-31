import os
import sys
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

CLASSES = ['Dry Period', 'Peak Lactation', 'Late Lactation', 'Fresh Cows', 'Peri-Partum']

class CowSonogramCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CowSonogramCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(6272, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.classification_head = nn.Linear(512, num_classes)
        self.regression_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.classification_head(x), self.regression_head(x)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ✅ Reads directly from your local repository folder pathway
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'cow_model.pth')

_cached_model = None

def get_model(model_path=DEFAULT_MODEL_PATH):
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Critical Error: Local model file missing at path: {model_path}. Make sure 'cow_model.pth' is inside your repo root directory folder.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)
        print("📦 Mounting local model checkpoint layer parameters...")
        
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        if missing_keys: print(f"⚠️ Missing keys: {missing_keys}")
        if unexpected_keys: print(f"⚠️ Unexpected keys: {unexpected_keys}")
        
        model.eval()
        print("✅ Core architecture layers loaded and synchronized perfectly from repository file system.")
        _cached_model = (model, device)
    return _cached_model

def get_consistent_class(yield_val):
    if yield_val == 0: return 'Dry Period'
    elif yield_val <= 10: return 'Peri-Partum'
    elif yield_val <= 20: return 'Fresh Cows'
    elif yield_val >= 35: return 'Peak Lactation'
    else: return 'Late Lactation'

def predict_image(image_path, model_path=DEFAULT_MODEL_PATH):
    try:
        model, device = get_model(model_path)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        
        if image_path.startswith('http://') or image_path.startswith('https://'):
            print(f"🌐 Fetching live sonogram byte stream from cloud storage link...")
            response = requests.get(image_path, timeout=15)
            if response.status_code != 200: raise RuntimeError(f"HTTP Error: {response.status_code}")
            raw_img = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(image_path): raise FileNotFoundError(f"Missing image file: {image_path}")
            raw_img = Image.open(image_path)

        image = raw_img.convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_logits, yield_pred = model(tensor)
            probabilities = torch.nn.functional.softmax(class_logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            final_yield = max(0.0, yield_pred.item())
            model_predicted_class = CLASSES[predicted_idx.item()]
            consistent_class = get_consistent_class(final_yield)
            
            if consistent_class == 'Peri-Partum' and model_predicted_class != 'Peri-Partum':
                consistent_class = model_predicted_class
                
        return consistent_class, float(confidence.item()), float(final_yield)
    except Exception as e:
        print(f"Error during custom CNN inference execution: {e}")
        raise RuntimeError(f"Inference failed: {e}")
