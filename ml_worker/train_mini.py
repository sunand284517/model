import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model import CowSonogramCNN, CLASSES
from dataset_loader import CowSonogramDataset
import os

def train_mini():
    device = torch.device('cpu')
    print(f"Training on {device} (Mini version for speed)...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CowSonogramDataset('dataset/train', transform=transform)
    # Only use first 32 images (1 batch)
    subset = Subset(dataset, range(32))
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    
    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)
    criterion_class = nn.CrossEntropyLoss()
    criterion_yield = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for inputs, class_labels, yield_labels in dataloader:
        optimizer.zero_grad()
        class_logits, yield_preds = model(inputs)
        loss_c = criterion_class(class_logits, class_labels)
        loss_y = criterion_yield(yield_preds, yield_labels)
        combined_loss = loss_c + (0.1 * loss_y)
        combined_loss.backward()
        optimizer.step()
        print(f"Mini-batch trained. Loss: {combined_loss.item():.4f}")
        break

    torch.save(model.state_dict(), 'cow_model.pth')
    print("Saved cow_model.pth (Initial weights after 1 batch)")

if __name__ == "__main__":
    train_mini()
