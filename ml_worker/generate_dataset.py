import os
from PIL import Image
import numpy as np

CLASSES = [
    'Dry Period',
    'Peak Lactation',
    'Late Lactation', 
    'Fresh Cows',
    'Peri-Partum'
]

def generate_dummy_dataset(base_dir='dataset/train', images_per_class=10):
    os.makedirs(base_dir, exist_ok=True)
    
    for cls in CLASSES:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(images_per_class):
            # Generate random noise image (224x224 pixels, RGB)
            random_pixels = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(random_pixels)
            
            img_path = os.path.join(cls_dir, f"{cls.replace(' ', '_').lower()}_{i}.jpg")
            img.save(img_path)
            
    print(f"Generated {images_per_class} images per class in {base_dir}")

if __name__ == '__main__':
    generate_dummy_dataset()
