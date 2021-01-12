import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def down_conv(tensor, n_filters, size, padding = 'same', initializer = 'he_normal'):
  x = Conv2D(n_filters, kernel_size = size, activation = 'relu', kernel_initializer = initializer)(tensor)
  x = Conv2D(n_filters, kernel_size = size, activation = 'relu', kernel_initializer = initializer)(x)
  return x

def crop_tensor(orig, tar):
  tar_size = tar.shape[2]
  tensor_size = orig.shape[2]
  diff = tensor_size - tar_size
  diff = diff // 2
  return orig[:, diff:tensor_size-diff, diff:tensor_size-diff, :]

def UNet(height, width, channels, n_class, n_filters=64):

  # Converging Block
  input_layer = Input(shape = (height, width, channels))
  conv1 = down_conv(input_layer, n_filters, size = (3,3))
  max1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv1)
  conv2 = down_conv(max1, n_filters*2, size = (3,3))
  max2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv2)
  conv3 = down_conv(max2, n_filters*4, size = (3,3))
  max3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv3)
  conv4 = down_conv(max3, n_filters*8, size = (3,3))
  max4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv4)
  conv5 = down_conv(max4, n_filters*16, size = (3,3))

  # Expansive Block
  exp_path1 = Conv2DTranspose(n_filters*8, kernel_size = (2,2), strides = (2,2), activation = 'relu')(conv5)
  y1 = crop_tensor(conv4, exp_path1)
  exp_path1 = Concatenate(axis = 3)([y1, exp_path1])
  exp_path1 = down_conv(exp_path1, n_filters*8, size = (3,3))

  exp_path2 = Conv2DTranspose(n_filters*4, kernel_size = (2,2), strides = (2,2), activation = 'relu')(exp_path1)
  y2 = crop_tensor(conv3, exp_path2)
  exp_path2 = Concatenate(axis = 3)([y2, exp_path2])
  exp_path2 = down_conv(exp_path2, n_filters*4, size = (3,3))

  exp_path3 = Conv2DTranspose(n_filters*2, kernel_size = (2,2), strides = (2,2), activation = 'relu')(exp_path2)
  y3 = crop_tensor(conv2, exp_path3)
  exp_path3 = Concatenate(axis = 3)([y3, exp_path3])
  exp_path3 = down_conv(exp_path3, n_filters*2, size = (3,3))

  exp_path4 = Conv2DTranspose(n_filters, kernel_size = (2,2), strides = (2,2), activation = 'relu')(exp_path3)
  y3 = crop_tensor(conv1, exp_path4)
  exp_path4 = Concatenate(axis = 3)([y3, exp_path4])
  exp_path4 = down_conv(exp_path4, n_filters, size = (3,3))

  # Output
  output = Conv2D(n_class, kernel_size = (1,1), activation = 'relu')(exp_path4)
  model = Model(inputs = input_layer, outputs = output)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model