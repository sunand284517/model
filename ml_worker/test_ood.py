import os
from PIL import Image
import numpy as np
from model import predict_image

# Create a dummy blank image
img_array = np.zeros((224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save('dummy_black.png')

img_array_white = np.ones((224, 224, 3), dtype=np.uint8) * 255
img_white = Image.fromarray(img_array_white)
img_white.save('dummy_white.png')

img_array_random = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img_random = Image.fromarray(img_array_random)
img_random.save('dummy_random.png')

c1, conf1, y1 = predict_image('dummy_black.png')
print(f"Black Image -> Class: {c1}, Conf: {conf1}, Yield: {y1}")

c2, conf2, y2 = predict_image('dummy_white.png')
print(f"White Image -> Class: {c2}, Conf: {conf2}, Yield: {y2}")

c3, conf3, y3 = predict_image('dummy_random.png')
print(f"Random Image -> Class: {c3}, Conf: {conf3}, Yield: {y3}")
