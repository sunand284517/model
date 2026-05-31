import os
import random
from model import predict_image

images_dir = 'dataset/train/images'
images = os.listdir(images_dir)
test_images = random.sample(images, 10)

for img in test_images:
    img_path = os.path.join(images_dir, img)
    # the function signature is def predict_image(image_path, model_path='cow_model.pth'):
    classification, confidence, yield_val = predict_image(img_path)
    print(f"Image: {img[:15]}... -> Class: {classification}, Conf: {confidence:.4f}, Yield: {yield_val:.2f}")
