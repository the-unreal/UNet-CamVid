from unet import UNet
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import save_model, load_model

def parse_code(l):
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c

label_codes, label_names = zip(*[parse_code(l) for l in open("label_colors.txt")])
label_codes, label_names = list(label_codes), list(label_names)

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}

name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}

def rgb_to_onehot(rgb_image, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    
    print(type(encoded_image))
    return encoded_image

def decode_segmentation(original_image, predicted_image):
    
    img_color = original_image.copy()
    arr_copy = np.argmax(predicted_image, axis = 0)
    for i in range(predicted_image.shape[0]):
        for j in range(predicted_image.shape[1]):
            img_color[i, j] = color_map[str(arr_copy[i, j])]
    
    return img_color

def data_process():
  data_gen = ImageDataGenerator(rescale=1./255)
  img = data_gen.flow_from_directory('data/images', target_size = (572,572), batch_size=4)
  mask = data_gen.flow_from_directory('data/labels', target_size = (572,572), batch_size=4)
  train_gen = zip(img,mask)
  return train_gen

def training():
    
    model = UNet(572, 572, 3, 32)
    train_gen = data_process()
    model.fit(train_gen,epochs=20)
    save_model(model,'models/unet.h5',save_format = 'h5')
    
if __name__ == '__main__':
    
    model = load_model('models/unet.h5')
    image = cv2.imread("", cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    predicted_image = model.predict(image)
    output = decode_segmentation(image, predicted_image)
    f, axarr = plt.subplots(1,2)
    axarr[0,0].imshow(image)
    axarr[0,1].imshow(output)
    plt.show()