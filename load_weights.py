'''
load weights from individual npy weight file
'''
import os
import numpy as np
from tensorflow.keras.models import Model
from marco import marco

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # p2.Xlarge only has one GPU

WEIGHTS_DIR = './weights'
MODEL_DIR = './models'
OUTPUT_WEIGHT_FILENAME = 'marco_weights_tf_dim_ordering_tf_kernels.h5'
OUTPUT_WEIGHT_FILENAME_NOTOP = 'marco_weights_tf_dim_ordering_tf_kernels_notop.h5'


print('Instantiating an empty marco model...')
model = marco(num_classes=4, weights=None, input_shape=(599, 599, 3))    # weights=None, no weights loaded

print('Loading weights from', WEIGHTS_DIR)
for layer in model.layers:
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '').replace('/', '_')
            weight_file = layer.name + '_' + weight_name + '.npy'
            print(weight_file)
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)

print('Saving model weights...')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME))

print('Saving model weights (no top)...')
model_notop = Model(model.inputs, model.get_layer(
    'Mixed_7c_Branch_3_Conv2d_0b_1x1_Activation').output)       # exclude fully connected layer
model_notop.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME_NOTOP))
