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

modified from 
https://github.com/yuyang-huang/keras-inception-resnet-v2/blob/master/inception_resnet_v2.py
based on 
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
'''
import numpy as np
import warnings
from functools import partial

# Keras Core
from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Flatten, Activation
from keras.layers import BatchNormalization, concatenate
from keras.models import Model
# Backend
from keras import backend as K
# Utils
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


#########################################################################################
# Implements the MARCO model based on Inception Network v3 in Keras.
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198883 #
#########################################################################################

def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
      This function applies the "Inception" preprocessing which converts
      the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
      function is different from keras `imagenet_utils.preprocess_input()`.
      # Arguments
          x: a 4D numpy array consists of RGB values within [0, 255].
      # Returns
          Preprocessed array.
      """
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


def conv2d_bn(x, nb_filter, kernel_size, strides=1,
              padding='same', activation='relu', use_bias=False, name=None):
    """
    Utility function to apply conv + BN.
    name: name of the ops; will become `name + '_Activation'`
            for the activation and `name + '_BatchNorm'` for the
            batch norm layer.
    """
    x = Conv2D(nb_filter, kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x


def _generate_layer_name(name, branch_idx=None, prefix=None):
    """Utility function for generating layer names.
    If `prefix` is `None`, returns `None` to use default automatic layer names.
    Otherwise, the returned layer name is:
        - PREFIX_NAME if `branch_idx` is not given.
        - PREFIX_Branch_0_NAME if e.g. `branch_idx=0` is given.
    # Arguments
        name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
        branch_idx: an `int`. If given, will add e.g. `'Branch_0'`
            after `prefix` and in front of `name` in order to identify
            layers in the same block but in different branches.
        prefix: string prefix that will be added in front of `name` to make
            all layer names unique (e.g. which block this layer belongs to).
    # Returns
        The layer name.
    """
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def marco_base(input):
    """
    conv base for the model
    """
    # Input Shape is 599 x 599 x 3 (th) or 3 x 599 x 599 (th)
    x = conv2d_bn(input, 16, (3, 3), strides=2,
                  padding='valid', name='Conv2d_0a_3x3')
    x = conv2d_bn(x, 32, (3, 3), strides=2,
                  padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, (3, 3), padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, (3, 3), name='Conv2d_2b_3x3')

    x = MaxPooling2D((3, 3), strides=2, name='MaxPool_3a_3x3')(x)

    x = conv2d_bn(x, 80, (1, 1), padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 144, (3, 3), padding='valid', name='Conv2d_4a_3x3')

    x = MaxPooling2D((3, 3), strides=2, name='MaxPool_5a_3x3')(x)

    # Mixed_5b
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    name_fmt = partial(_generate_layer_name, prefix='Mixed_5b')
    branch_0 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 48, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 64, (5, 5),
                         name=name_fmt('Conv2d_0b_5x5', 1))

    branch_2 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3),
                         name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3),
                         name=name_fmt('Conv2d_0c_3x3', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 32, (1, 1),
                         name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2, branch_3],
                    axis=channel_axis, name='Mixed_5b')

    # Mixed_5c
    name_fmt = partial(_generate_layer_name, prefix='Mixed_5c')
    branch_0 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 48, (1, 1), name=name_fmt(
        'Conv2d_0b_1x1', 1)) 
    branch_1 = conv2d_bn(branch_1, 64, (5, 5),
                          name=name_fmt('Conv_1_0c_5x5', 1))  

    branch_2 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3),
                          name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3),
                          name=name_fmt('Conv2d_0c_3x3', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 64, (1, 1),
                          name=name_fmt('Conv2d_0b_1x1', 3))
    x = concatenate([branch_0, branch_1, branch_2, branch_3],
                    axis=channel_axis, name='Mixed_5c')

    # Mixed_5d
    name_fmt = partial(_generate_layer_name, prefix='Mixed_5d')
    branch_0 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 48, (1, 1), name=name_fmt(
        'Conv2d_0a_1x1', 1))  
    branch_1 = conv2d_bn(branch_1, 64, (5, 5),
                          name=name_fmt('Conv2d_0b_5x5', 1))  

    branch_2 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3),
                          name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 96, (3, 3),
                          name=name_fmt('Conv2d_0c_3x3', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 64, (1, 1),
                          name=name_fmt('Conv2d_0b_1x1', 3))
    x = concatenate([branch_0, branch_1, branch_2, branch_3],
                    axis=channel_axis, name='Mixed_5d')


    # Mixed_6a
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x, 256, (3, 3), strides=2, padding='valid',
                         name=name_fmt('Conv2d_1a_1x1', 0))  

    branch_1 = conv2d_bn(x, 64, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 96, (3, 3),
                         name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1, 96, (3, 3), strides=2,
                         padding='valid', name=name_fmt('Conv2d_1a_1x1', 1))   

    branch_2 = MaxPooling2D(
        (3, 3), strides=2, name=name_fmt('MaxPool_1a_3x3', 2))(x)

    x = concatenate([branch_0, branch_1, branch_2],
                    axis=channel_axis, name=name_fmt('Mixed_6a'))

    # Mixed_6b
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6b')
    branch_0 = conv2d_bn(x, 128, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 128, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 128, (1, 7),
                         name=name_fmt('Conv2d_0b_1x7', 1))
    branch_1 = conv2d_bn(branch_1, 128, (7, 1),
                         name=name_fmt('Conv2d_0c_7x1', 1))

    branch_2 = conv2d_bn(x, 128, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 128, (7, 1),
                         name=name_fmt('Conv2d_0b_7x1', 2))
    branch_2 = conv2d_bn(branch_2, 128, (1, 7),
                         name=name_fmt('Conv2d_0c_1x7', 2))
    branch_2 = conv2d_bn(branch_2, 128, (7, 1),
                         name=name_fmt('Conv2d_0d_7x1', 2))
    branch_2 = conv2d_bn(branch_2, 128, (1, 7),
                         name=name_fmt('Conv2d_0e_1x7', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 128, (1, 1),
                         name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2,
                     branch_3], axis=channel_axis, name=name_fmt('Mixed_6b'))

    # Mixed_6c
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6c')
    branch_0 = conv2d_bn(x, 144, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 144, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 144, (1, 7),
                          name=name_fmt('Conv2d_0b_1x7', 1))
    branch_1 = conv2d_bn(branch_1, 144, (7, 1),
                          name=name_fmt('Conv2d_0c_7x1', 1))

    branch_2 = conv2d_bn(x, 144, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 160, (7, 1), name=name_fmt(
        'Conv2d_0b_7x1', 2))  
    branch_2 = conv2d_bn(branch_2, 144, (1, 7),
                          name=name_fmt('Conv2d_0c_1x7', 2))
    branch_2 = conv2d_bn(branch_2, 144, (7, 1),
                          name=name_fmt('Conv2d_0d_7x1', 2))
    branch_2 = conv2d_bn(branch_2, 144, (1, 7),
                          name=name_fmt('Conv2d_0e_1x7', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 144, (1, 1),
                          name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2,
                      branch_3], axis=channel_axis, name='Mixed_6c')

    # Mixed_6d
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6d')
    branch_0 = conv2d_bn(x, 144, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 144, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 144, (1, 7),
                          name=name_fmt('Conv2d_0b_1x7', 1))
    branch_1 = conv2d_bn(branch_1, 144, (7, 1),
                          name=name_fmt('Conv2d_0c_7x1', 1))

    branch_2 = conv2d_bn(x, 144, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 144, (7, 1), name=name_fmt(
        'Conv2d_0b_7x1', 2))  
    branch_2 = conv2d_bn(branch_2, 144, (1, 7),
                          name=name_fmt('Conv2d_0c_1x7', 2))
    branch_2 = conv2d_bn(branch_2, 144, (7, 1),
                          name=name_fmt('Conv2d_0d_7x1', 2))
    branch_2 = conv2d_bn(branch_2, 144, (1, 7),
                          name=name_fmt('Conv2d_0e_1x7', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 144, (1, 1),
                          name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2,
                      branch_3], axis=channel_axis, name='Mixed_6d')

    # Mixed_6e
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6e')
    branch_0 = conv2d_bn(x, 96, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 96, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 96, (1, 7),
                         name=name_fmt('Conv2d_0b_1x7', 1))
    branch_1 = conv2d_bn(branch_1, 96, (7, 1),
                         name=name_fmt('Conv2d_0c_7x1', 1))

    branch_2 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, (7, 1),
                         name=name_fmt('Conv2d_0b_7x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, (1, 7),
                         name=name_fmt('Conv2d_0c_1x7', 2))
    branch_2 = conv2d_bn(branch_2, 96, (7, 1),
                         name=name_fmt('Conv2d_0d_7x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, (1, 7),
                         name=name_fmt('Conv2d_0e_1x7', 2))

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 96, (1, 1),
                         name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2,
                       branch_3], axis=channel_axis, name=name_fmt('Mixed_6e'))

    # Mixed_7a
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 96, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0, 96, (3, 3), strides=2,
                         padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))

    branch_1 = conv2d_bn(x, 96, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 96, (1, 7),
                         name=name_fmt('Conv2d_0b_1x7', 1))
    branch_1 = conv2d_bn(branch_1, 96, (7, 1),
                         name=name_fmt('Conv2d_0c_7x1', 1))
    branch_1 = conv2d_bn(branch_1, 96, (3, 3), strides=2,
                         padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))

    branch_2 = MaxPooling2D(
        (3, 3), strides=2, name=name_fmt('MaxPool_1a_3x3', 2))(x)

    x = concatenate([branch_0, branch_1, branch_2],
                    axis=channel_axis, name=name_fmt('Mixed_7a'))

    # Mixed_7b
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7b')
    branch_0 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1_1 = conv2d_bn(branch_1, 192, (1, 3),
                            name=name_fmt('Conv2d_0b_1x3', 1))
    branch_1_2 = conv2d_bn(branch_1, 192, (3, 1),
                            name=name_fmt('Conv2d_0b_3x1', 1))
    branch_1 = concatenate([branch_1_1, branch_1_2], axis=channel_axis)   

    branch_2 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 192, (3, 3),
                          name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2_1 = conv2d_bn(branch_2, 192, (1, 3),
                            name=name_fmt('Conv2d_0c_1x3', 2))
    branch_2_2 = conv2d_bn(branch_2, 192, (3, 1),
                            name=name_fmt('Conv2d_0d_3x1', 2))
    branch_2 = concatenate([branch_2_1, branch_2_2], axis=channel_axis)   

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 96, (1, 1),
                          name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2,
                        branch_3], axis=channel_axis, name='Mixed_7b')

    # Mixed_7c
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7c')
    branch_0 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 0))

    branch_1 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1_1 = conv2d_bn(branch_1, 192, (1, 3),
                            name=name_fmt('Conv2d_0b_1x3', 1))
    branch_1_2 = conv2d_bn(branch_1, 192, (3, 1),
                            name=name_fmt('Conv2d_0c_3x1', 1))
    branch_1 = concatenate([branch_1_1, branch_1_2], axis=channel_axis)   

    branch_2 = conv2d_bn(x, 192, (1, 1), name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 192, (3, 3),
                          name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2_1 = conv2d_bn(branch_2, 192, (1, 3),
                            name=name_fmt('Conv2d_0c_1x3', 2))
    branch_2_2 = conv2d_bn(branch_2, 192, (3, 1),
                            name=name_fmt('Conv2d_0d_3x1', 2))
    branch_2 = concatenate([branch_2_1, branch_2_2], axis=channel_axis)   

    branch_3 = AveragePooling2D(
        (3, 3), strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_3 = conv2d_bn(branch_3, 96, (1, 1),
                          name=name_fmt('Conv2d_0b_1x1', 3))

    x = concatenate([branch_0, branch_1, branch_2,
                        branch_3], axis=channel_axis, name='Mixed_7c')

    return x


def marco(include_top=True,
          weights='marco',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          num_classes=4):
    """Instantiates the marco architecture.
    Optionally loads weights pre-trained on MARCO.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.
    The model and the weights are compatible with both TensorFlow and Theano.
    The data format convention used by the model is the one specified in your
    Keras config file.
    Note that the default input image size for this model is 599x599. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or `'imagenet'` (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(599, 599, 3)` (with `channels_last` data format)
            or `(3, 599, 599)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'marco', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `marco` '
                         '(pre-training on MARCO).')

    if weights == 'marco' and include_top and num_classes != 4:
        raise ValueError('If using `weights` as marco with `include_top`'
                         ' as true, `num_classes` should be 4')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=599,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Make inception base
    x = marco_base(img_input)

    # Final pooling and prediction
    if include_top:
        name_fmt = partial(_generate_layer_name, prefix='Logits')
        x = AveragePooling2D((8,8), strides=2, padding='valid', name=name_fmt('AvgPool_1a_8x8'))(x)
        x = Conv2D(num_classes, (1,1),
               strides=1,
               padding='same',
               use_bias=True,
               name=name_fmt('Conv2d_1c_1x1'))(x)
        x = Flatten()(x)
        x = Activation('softmax', name=name_fmt('Predictions'))(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='AvgPool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='MaxPool')(x)
  

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create final model
    model = Model(inputs, x, name='marco')

    # Load weights
    if weights == 'marco':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_filename = './models/marco_weights_tf_dim_ordering_tf_kernels.h5'
        else:
            weights_filename = './models/marco_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model.load_weights(weights_filename)

    return model
