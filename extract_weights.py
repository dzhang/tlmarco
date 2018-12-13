import os
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # p2.Xlarge only has one GPU

def get_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.

    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('Inception_', '')
    filename = filename.replace(':0', '')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_saved_model(model_path='savedmodel', output_folder='weights'):
    """Extract tensors from a TF savedModel.

    # Arguments
        model_path: savedModel folder
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sess = tf.Session()
    # load both graph and variables
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

    vars_model = tf.global_variables()
    for i in range(0, len(vars_model)):
        key = vars_model[i]
        # not saving the following tensors
        if 'global_step' in key.name:
            continue
        if 'AuxLogit' in key.name:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key.name))
        arr = sess.run(key)
        np.save(path, arr)
        print("tensor_name: ", key)

if __name__ == "__main__":
    extract_tensors_from_saved_model()
