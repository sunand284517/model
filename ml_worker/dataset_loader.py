import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

class CowSonogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.transform = transform
        self.records = []
        
        labels_path = os.path.join(data_dir, 'labels.csv')
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.records.append({
                        'filename': row['filename'],
                        'class_idx': int(row['class_idx']),
                        'dmy': float(row['dmy'])
                    })
                    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = os.path.join(self.image_dir, record['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, record['class_idx'], torch.tensor([record['dmy']], dtype=torch.float32)
