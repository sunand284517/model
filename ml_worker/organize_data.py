import os
import csv
import shutil
import random

# Paths
source_dataset_dir = '/Users/vundhyalaakeshreddy/Downloads/B-mode_sonograms_bovine_mammary_gland'
labels_csv = '/Users/vundhyalaakeshreddy/Downloads/milk_yield_labels.csv'
target_dataset_dir = '/Users/vundhyalaakeshreddy/Desktop/project/ml_worker/dataset'

CLASSES = ['Dry Period', 'Peak Lactation', 'Late Lactation', 'Fresh Cows', 'Peri-Partum']

def get_class_idx(dmy_str):
    if dmy_str == 'N/A' or dmy_str == '-':
        return CLASSES.index('Peri-Partum')
    try:
        val = float(dmy_str)
        if val == 0:
            return CLASSES.index('Dry Period')
        elif val < 20: 
            return CLASSES.index('Fresh Cows')
        elif val >= 35:
            return CLASSES.index('Peak Lactation')
        else:
            return CLASSES.index('Late Lactation')
    except ValueError:
        return CLASSES.index('Peri-Partum')

def get_dmy_val(dmy_str):
    try:
        return float(dmy_str)
    except:
        return None

def main():
    # 1. Clear existing dataset
    if os.path.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    
    # Create target directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join(target_dataset_dir, split, 'images'), exist_ok=True)

    # 2. Load labels
    dmy_map = {}
    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cow_id = row['Cow'].strip()
            dmy = row['DMY'].strip()
            dmy_map[cow_id] = dmy

    # 3. Process images
    train_records = []
    test_records = []
    missing_labels = 0
    non_numeric_dmy = 0
    
    for cow_folder in os.listdir(source_dataset_dir):
        cow_folder_path = os.path.join(source_dataset_dir, cow_folder)
        if not os.path.isdir(cow_folder_path):
            continue
            
        for img_file in os.listdir(cow_folder_path):
            if not img_file.endswith('.png'):
                continue
                
            parts = img_file.split('_')
            if len(parts) >= 2:
                sample_id = f"{parts[0]}_{parts[1]}"
                
                if sample_id in dmy_map:
                    dmy_str = dmy_map[sample_id]
                    # We need numerical DMY for regression. Skip N/A or '-'
                    exact_dmy = get_dmy_val(dmy_str)
                    
                    if exact_dmy is None:
                        # Skip if it doesn't have a valid yield number.
                        non_numeric_dmy += 1
                        continue

                    class_idx = get_class_idx(dmy_str)
                    
                    src_path = os.path.join(cow_folder_path, img_file)
                    
                    # 80/20 train/test split
                    is_train = random.random() < 0.80
                    target_split = 'train' if is_train else 'test'
                    
                    dst_path = os.path.join(target_dataset_dir, target_split, 'images', img_file)
                    shutil.copy2(src_path, dst_path)
                    
                    record = [img_file, class_idx, exact_dmy]
                    if is_train:
                        train_records.append(record)
                    else:
                        test_records.append(record)
                else:
                    missing_labels += 1

    # Write CSVs
    with open(os.path.join(target_dataset_dir, 'train', 'labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class_idx', 'dmy'])
        writer.writerows(train_records)

    with open(os.path.join(target_dataset_dir, 'test', 'labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class_idx', 'dmy'])
        writer.writerows(test_records)

    print(f"Dataset organization complete!")
    print(f"Total train images: {len(train_records)}")
    print(f"Total test images: {len(test_records)}")
    print(f"Images skipped due to missing labels: {missing_labels}")
    print(f"Images skipped due to non-numeric DMY: {non_numeric_dmy}")

if __name__ == '__main__':
    random.seed(42)  # For reproducible split
    main()
