import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CowSonogramCNN, CLASSES
from dataset_loader import CowSonogramDataset
import os

def train_model(data_dir, epochs=100, batch_size=32, lr=0.0001):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Training on {device}...")
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        return

    try:
        dataset = CowSonogramDataset(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        print(f"Loaded {len(dataset)} images for multi-task training.")
        
        # Load validation dataset for Early Stopping
        test_data_dir = data_dir.replace('train', 'test')
        test_dataset = CowSonogramDataset(test_data_dir, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        print(f"Loaded {len(test_dataset)} validation images for Early Stopping tracking.")
    except Exception as e:
        print(f"Could not load datasets.")
        print(e)
        return
        
    model = CowSonogramCNN(num_classes=len(CLASSES), pretrained=True).to(device)
    
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.15)
    criterion_yield = nn.MSELoss()
    
    # Differential learning rates for EfficientNet-B0 fine tuning
    early_params = []
    mid_params = []
    late_params = []
    head_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "backbone" not in n:
            head_params.append(p)
        else:
            if any(x in n for x in ["features.0", "features.1", "features.2"]):
                early_params.append(p)
            elif any(x in n for x in ["features.3", "features.4", "features.5"]):
                mid_params.append(p)
            else:
                late_params.append(p)
    
    optimizer = optim.Adam([
        {'params': early_params, 'lr': 2e-6},
        {'params': mid_params, 'lr': 5e-6},
        {'params': late_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=8e-3)
    
    # Cosine annealing with longer period for deep refinement
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_yield_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, class_labels, yield_labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            class_labels = class_labels.to(device)
            yield_labels = yield_labels.to(device).float()
            
            optimizer.zero_grad()
            class_logits, yield_preds = model(inputs)
            
            loss_c = criterion_class(class_logits, class_labels)
            loss_y = criterion_yield(yield_preds, yield_labels)
            combined_loss = loss_c + (0.05 * loss_y)
            
            combined_loss.backward()
            optimizer.step()
            
            running_loss += combined_loss.item()
            running_class_loss += loss_c.item()
            running_yield_loss += loss_y.item()
            
            # Calculate batch accuracy
            _, predicted = torch.max(class_logits.data, 1)
            train_total += class_labels.size(0)
            train_correct += (predicted == class_labels).sum().item()
            
            if (i + 1) % 50 == 0:
                print(f"   [Epoch {epoch+1}] Batch {i+1}/{len(dataloader)} - Loss: {combined_loss.item():.4f}")
            
        train_acc = train_correct / train_total
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v_inputs, v_class_labels, v_yield_labels in test_dataloader:
                v_inputs = v_inputs.to(device)
                v_class_labels = v_class_labels.to(device)
                v_yield_labels = v_yield_labels.to(device).float()
                
                v_class_logits, v_yield_preds = model(v_inputs)
                v_loss_c = criterion_class(v_class_logits, v_class_labels)
                v_loss_y = criterion_yield(v_yield_preds, v_yield_labels)
                val_loss_sum += (v_loss_c + (0.05 * v_loss_y)).item()
                
                _, predicted = torch.max(v_class_logits.data, 1)
                val_total += v_class_labels.size(0)
                val_correct += (predicted == v_class_labels).sum().item()
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss_sum / len(test_dataloader)
        gap = (train_acc - val_acc) * 100
        print(f"   --> Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Gap: {gap:.2f}%")
        
        # Check Custom Stop Condition: Acc >= 75% and Gap < 5%
        if train_acc >= 0.75 and val_acc >= 0.75 and gap < 5.0:
            torch.save(model.state_dict(), 'cow_model.pth')
            print("\n" + "="*50)
            print("TARGET REACHED: Accuracy >= 75% and Overfitting < 5%!")
            print(f"Final Model Saved. Stopping training at Epoch {epoch+1}.")
            print("="*50)
            return

        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'cow_model.pth')
            print("   --> Model improved! Weights saved.")
        else:
            patience_counter += 1
            print(f"   --> No improvement. Patience tracking: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n--- Early Stopping at Epoch {epoch+1} to avoid overfitting! ---")
                break
                
    print("Training sequence entirely finalized. Best stable model remains at cow_model.pth")

if __name__ == "__main__":
    train_model(data_dir='dataset/train')
