import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CowSonogramCNN, CLASSES
from dataset_loader import CowSonogramDataset
import os

def evaluate_model(data_dir, model_path='cow_model.pth', batch_size=32):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Evaluating on {device}...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        return

    try:
        dataset = CowSonogramDataset(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Could not load dataset from {data_dir}.")
        print(e)
        return
        
    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model path {model_path} does not exist. (Training probably hasn't run yet)")
        return
        
    model.eval()
    correct = 0
    total = 0
    total_ae = 0.0 # Absolute Error sum
    
    with torch.no_grad():
        for inputs, class_labels, yield_labels in dataloader:
            inputs = inputs.to(device)
            class_labels = class_labels.to(device)
            yield_labels = yield_labels.to(device)
            
            class_logits, yield_preds = model(inputs)
            
            # Classification
            _, predicted = torch.max(class_logits.data, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
            
            # Regression
            abs_errors = torch.abs(yield_preds - yield_labels)
            total_ae += torch.sum(abs_errors).item()
            
    accuracy = 100 * correct / (total + 1e-9)
    mae = total_ae / (total + 1e-9)
    print(f"Results on {data_dir} ({total} images):")
    print(f" - Classification Accuracy:  {accuracy:.2f}%")
    print(f" - Regression MAE (Yield):   {mae:.2f} liters/day")
    return accuracy, mae

if __name__ == "__main__":
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    
    print("--- Evaluating Train Split ---")
    evaluate_model(train_dir)
    
    print("\n--- Evaluating Test Split ---")
    evaluate_model(test_dir)

