import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model import CowSonogramCNN, CLASSES
from dataset_loader import CowSonogramDataset
import os
import random

def evaluate_fast(data_dir, subset_size=1000, model_path='cow_model.pth', batch_size=32):
    device = torch.device('cpu')
    print(f"Evaluating on {device} (Fast Subset: {subset_size} images)...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = CowSonogramDataset(data_dir, transform=transform)
        if len(dataset) > subset_size:
            indices = random.sample(range(len(dataset)), subset_size)
            subset = Subset(dataset, indices)
        else:
            subset = dataset
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Could not load dataset from {data_dir}.")
        return
        
    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    correct = 0
    total = 0
    total_ae = 0.0
    
    with torch.no_grad():
        for inputs, class_labels, yield_labels in dataloader:
            inputs = inputs.to(device)
            class_labels = class_labels.to(device)
            yield_labels = yield_labels.to(device)
            
            class_logits, yield_preds = model(inputs)
            
            _, predicted = torch.max(class_logits.data, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
            
            abs_errors = torch.abs(yield_preds - yield_labels)
            total_ae += torch.sum(abs_errors).item()
            
    accuracy = 100 * correct / (total + 1e-9)
    mae = total_ae / (total + 1e-9)
    print(f"Results on {data_dir} ({total} images):")
    print(f" - Classification Accuracy:  {accuracy:.2f}%")
    print(f" - Regression MAE (Yield):   {mae:.2f} liters/day")

if __name__ == "__main__":
    print("--- Evaluating Train Split (Fast) ---")
    evaluate_fast('dataset/train')
    print("\n--- Evaluating Test Split (Fast) ---")
    evaluate_fast('dataset/test')
