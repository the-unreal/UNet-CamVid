import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
import matplotlib.pyplot as plt

height = 256
width = 256

def load_data(height, width, image_path, mask_path):
        
    X = np.zeros((200, height, width, 3), dtype = np.float32)
    y = np.zeros((200, height, width, 3), dtype = np.float32)
    
    for i, file in enumerate(os.listdir(image_path)):
        
        images = cv2.imread(os.path.join(image_path, file))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(images, (height,width))
        orig_img = img_to_array(orig_img)
        orig_img = orig_img/255
        X[i] = orig_img
        
        mask_images = cv2.imread(os.path.join(mask_path, file))
        mask_images = cv2.cvtColor(mask_images, cv2.COLOR_BGR2RGB)
        mask_images = cv2.resize(mask_images, (height,width))
        mask_img = img_to_array(mask_images)
        mask_img = mask_img/255
        y[i] = mask_img
        
    X.sort()
    y.sort()
    
    np.save("X.npy", X)
    np.save("y.npy", y)

if __name__ == '__main__':
    
    image_path = "data\\training\\image_2"
    mask_path = "data\\training\\semantic_rgb"
    load_data(256, 256, image_path, mask_path)