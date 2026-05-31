import os
import requests
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms


# =========================
# CLASSES
# =========================
CLASSES = [
    'Dry Period',
    'Peak Lactation',
    'Late Lactation',
    'Fresh Cows',
    'Peri-Partum'
]


# =========================
# MODEL DEFINITION
# =========================
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
            nn.MaxPool2d(2, 2),
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


# =========================
# MODEL PATH (same folder)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cow_model.pth")


# =========================
# CACHE
# =========================
_cached_model = None


# =========================
# LOAD MODEL (FIXED)
# =========================
def get_model(model_path=MODEL_PATH):
    global _cached_model

    if _cached_model is None:

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

        print("📦 Loading cow_model.pth ...")

        state_dict = torch.load(model_path, map_location=device)

        # FIXED: correct load_state_dict handling
        result = model.load_state_dict(state_dict, strict=False)

        print(f"⚠ Missing keys: {result.missing_keys}")
        print(f"⚠ Unexpected keys: {result.unexpected_keys}")

        model.eval()

        print("✅ Model loaded successfully")

        _cached_model = (model, device)

    return _cached_model


# =========================
# RULE-BASED CLASSIFIER
# =========================
def get_consistent_class(yield_val):
    if yield_val == 0:
        return 'Dry Period'
    elif yield_val <= 10:
        return 'Peri-Partum'
    elif yield_val <= 20:
        return 'Fresh Cows'
    elif yield_val >= 35:
        return 'Peak Lactation'
    else:
        return 'Late Lactation'


# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image_path, model_path=MODEL_PATH):

    try:
        model, device = get_model(model_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # -------------------------
        # Load image
        # -------------------------
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=15)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP error: {response.status_code}")

            image = Image.open(BytesIO(response.content)).convert("RGB")

        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        # -------------------------
        # Inference
        # -------------------------
        with torch.no_grad():
            class_logits, yield_pred = model(tensor)

            probs = torch.nn.functional.softmax(class_logits, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

            final_yield = max(0.0, yield_pred.item())

            model_class = CLASSES[predicted_idx.item()]
            rule_class = get_consistent_class(final_yield)

            # simple correction logic
            final_class = rule_class
            if rule_class == "Peri-Partum" and model_class != "Peri-Partum":
                final_class = model_class

        return final_class, float(confidence.item()), float(final_yield)

    except Exception as e:
        print("❌ Inference error:", str(e))
        raise RuntimeError(f"Inference failed: {str(e)}")
