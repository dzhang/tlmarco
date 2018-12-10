'''
Copyright 2017 TensorFlow Authors and Dong Zhang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import numpy as np

# Keras Core
from tensorflow.keras.layers import MaxPooling2D, Convolution2D, AveragePooling2D
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization, concatenate
from tensorflow.keras.models import Model
# Backend
from tensorflow.keras import backend as K
# Utils
from tensorflow.keras.utils import get_file


#########################################################################################
# Implements the MARCO model based on Inception Network v3 in Keras. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198883 #
#########################################################################################

WEIGHTS_PATH = '../weights/marco_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '../weights/marco_weights_tf_dim_ordering_tf_kernels_notop.h5'


def preprocess_input(x):
  x = np.divide(x, 255.0)
  x = np.subtract(x, 0.5)
  x = np.multiply(x, 2.0)
  return x


def conv2d_bn(x, nb_filter, num_row, num_col,
            padding='same', strides=(1, 1), use_bias=False):
  """
  Utility function to apply conv + BN. 
  """
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
  x = Convolution2D(nb_filter, (num_row, num_col),
                    strides=strides,
                    padding=padding,
                    use_bias=use_bias)(x)
  x = BatchNormalization(axis=channel_axis, scale=False)(x)
  x = Activation('relu')(x)
  return x

def marco_base(input):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1

  # Input Shape is 599 x 599 x 3 (th) or 3 x 599 x 599 (th)
  net = conv2d_bn(input, 16, 3, 3, strides=(2,2), padding='valid')  # Conv2d_0a_3x3
  net = conv2d_bn(net, 32, 3, 3, strides=(2,2), padding='valid')  # Conv2d_1a_3x3
  net = conv2d_bn(net, 32, 3, 3, padding='valid')   # Conv2d_2a_3x3
  net = conv2d_bn(net, 64, 3, 3)    # Conv2d_2b_3x3
  
  net = MaxPooling2D((3, 3), strides=(2, 2))(net)   # MaxPool_3a_3x3
  
  net = conv2d_bn(net, 80, 1, 1, padding='valid')   # Conv2d_3b_1x1
  net = conv2d_bn(net, 144, 3, 3, padding='valid')  # Conv2d_4a_3x3
  
  net = MaxPooling2D((3, 3), strides=(2, 2))(net)   # MaxPool_5a_3x3

  # Mixed_5b: 
  branch_0 = conv2d_bn(net, 64, 1, 1)

  branch_1 = conv2d_bn(net, 48, 1, 1)
  branch_1 = conv2d_bn(branch_1, 64, 5, 5)

  branch_2 = conv2d_bn(net, 64, 1, 1)
  branch_2 = conv2d_bn(branch_2, 96, 3, 3)
  branch_2 = conv2d_bn(branch_2, 96, 3, 3)

  branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
  branch_3 = conv2d_bn(branch_3, 32, 1, 1)
  
  net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
  
  # Mixed_5c and Mixed_5d: 
  for i in range(2):
    branch0 = conv2d_bn(net, 64, 1, 1)

    branch_1 = conv2d_bn(net, 64, 1, 1)   # changed from 48
    branch_1 = conv2d_bn(branch_1, 64, 5, 5)

    branch_2 = conv2d_bn(net, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch_3 = conv2d_bn(branch_3, 64, 1, 1)
    
    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
  
  # Mixed_6a: 
  branch_0 = conv2d_bn(net, 256, 3, 3, strides=(2, 2), padding='valid')

  branch_1 = conv2d_bn(net, 64, 1, 1)
  branch_1 = conv2d_bn(branch_1, 96, 3, 3)
  branch_1 = conv2d_bn(branch_1, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_2 = MaxPooling2D((3, 3), strides=(2, 2))(net)
  
  net = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)

  # Mixed_6b: 
  branch_0 = conv2d_bn(net, 128, 1, 1)

  branch_1 = conv2d_bn(net, 128, 1, 1)
  branch_1 = conv2d_bn(branch_1, 128, 1, 7)
  branch_1 = conv2d_bn(branch_1, 128, 7, 1)

  branch_2 = conv2d_bn(net, 128, 1, 1)
  branch_2 = conv2d_bn(branch_2, 128, 7, 1)
  branch_2 = conv2d_bn(branch_2, 128, 1, 7)
  branch_2 = conv2d_bn(branch_2, 128, 7, 1)
  branch_2 = conv2d_bn(branch_2, 128, 1, 7)

  branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
  branch_3 = conv2d_bn(branch_3, 128, 1, 1)
  
  net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
  
  # Mixed_6c and Mixed_6d: 
  for i in range(2):
    branch_0 = conv2d_bn(net, 144, 1, 1)

    branch_1 = conv2d_bn(net, 144, 1, 1)
    branch_1 = conv2d_bn(branch_1, 144, 1, 7)
    branch_1 = conv2d_bn(branch_1, 144, 7, 1)

    branch_2 = conv2d_bn(net, 144, 1, 1)
    branch_2 = conv2d_bn(branch_2, 144, 7, 1)   # 6c is 160
    branch_2 = conv2d_bn(branch_2, 144, 1, 7)
    branch_2 = conv2d_bn(branch_2, 144, 7, 1)
    branch_2 = conv2d_bn(branch_2, 144, 1, 7)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch_3 = conv2d_bn(branch_3, 144, 1, 1)
    
    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
      
  # Mixed_6e: 
  branch_0 = conv2d_bn(net, 96, 1, 1)

  branch_1 = conv2d_bn(net, 96, 1, 1)
  branch_1 = conv2d_bn(branch_1, 96, 1, 7)
  branch_1 = conv2d_bn(branch_1, 96, 7, 1)

  branch_2 = conv2d_bn(net, 192, 1, 1) 
  branch_2 = conv2d_bn(branch_2, 96, 7, 1)
  branch_2 = conv2d_bn(branch_2, 96, 1, 7)
  branch_2 = conv2d_bn(branch_2, 96, 7, 1)
  branch_2 = conv2d_bn(branch_2, 96, 1, 7)

  branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
  branch_3 = conv2d_bn(branch_3, 96, 1, 1)
  
  net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
      
  # Mixed_7a: 
  branch_0 = conv2d_bn(net, 96, 1, 1)
  branch_0 = conv2d_bn(branch_0, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_1 = conv2d_bn(net, 96, 1, 1)
  branch_1 = conv2d_bn(branch_1, 96, 1, 7)
  branch_1 = conv2d_bn(branch_1, 96, 7, 1)
  branch_1 = conv2d_bn(branch_1, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_2 = MaxPooling2D((3, 3), strides=(2, 2))(net)
  
  net = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
  
  # Mixed_7b and Mixed_7c: 
  for i in range(2):
    branch_0 = conv2d_bn(net, 192, 1, 1)

    brand_1 = conv2d_bn(net, 192, 1, 1)
    branch_1_1 = conv2d_bn(brand_1, 192, 1, 3)
    branch_1_2 = conv2d_bn(brand_1, 192, 3, 1)
    brand_1 = concatenate([branch_1_1, branch_1_2], axis=channel_axis)

    branch_2 = conv2d_bn(net, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 3, 3)
    branch_2_1 = conv2d_bn(branch_2, 192, 1, 3)
    branch_2_2 = conv2d_bn(branch_2, 192, 3, 1)
    branch_2 = concatenate([branch_2_1, branch_2_2], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)
    
    net = concatenate([branch_0, brand_1, branch_2, branch_3], axis=channel_axis)
      
  return net


def marco(num_classes, dropout_keep_prob, weights, include_top):
  '''
  Creates the marco network

  Args:
    num_classes: number of classes
    dropout_keep_prob: float, the fraction to keep before final layer.
  
  Returns: 
    logits: the logits outputs of the model.
  '''

  # Input Shape is 599 x 599 x 3 (tf) or 3 x 599 x 599 (th)
  if K.image_data_format() == 'channels_first':
    inputs = Input((3, 599, 599))
  else:
    inputs = Input((599, 599, 3))

  # Make inception base
  x = marco_base(inputs)

  # Final pooling and prediction
  if include_top:
    x = AveragePooling2D((8,8), padding='valid')(x)
    x = Dropout(dropout_keep_prob)(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation='softmax')(x)

  model = Model(inputs, x, name='marco')

  # load weights
  if weights == 'imagenet':
    if include_top:
      weights_path = get_file(
        'marco_weights_tf_dim_ordering_tf_kernels.h5',
        WEIGHTS_PATH,
        cache_subdir='models',
        md5_hash='9fe79d77f793fe874470d84ca6ba4a3b')
    else:
      weights_path = get_file(
        'marco_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='9296b46b5971573064d12e4669110969')
    model.load_weights(weights_path, by_name=True)
  return model

def create_model(num_classes=4, dropout_prob=0.8, weights=None, include_top=True):
  return marco(num_classes, dropout_prob, weights, include_top)