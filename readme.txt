The original MARCO model for protein crystalization image classification is in Tensorflow
this project convert it to Keras

https://medium.com/@daj/how-to-inspect-a-pre-trained-tensorflow-model-5fd2ee79ced0
import_pb_to_tensorboard.py, don't use this code, it will throw parse error due to buffer limit
https://github.com/tensorflow/tensorflow/issues/582

use the saved model api to load the model, downloaded from 
https://github.com/tensorflow/models/tree/master/research/marco
python import_pb_to_tensorboard.py --model_dir savedmodel --log_dir tensorboard

tensorboard --logdir=tensorboard
http://localhost:6006

check graph against, and create marco.py in Keras based on
https://github.com/yuyang-huang/keras-inception-resnet-v2/blob/master/inception_resnet_v2.py
slim inceptionV3 code, in tensorflow
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
keras inceptionV3 code
https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
MARCo papter on modified inceptionV3 model
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198883
extra lay Conv2d_0a_3x3
599x599 image size, depth of 16, 3x3 filter, stride 2
also adjusted depth for some of the layters, see paper for details
also reference
https://github.com/kentsommer/keras-inceptionV4/blob/master/inception_v4.py

what about AuxLogits layer, should it be included for re-training?
we will only re-train layer 7(a,b,c), these are after AuxLogits, so no need to include AuxLogits

extract_weights.py
extract weights from tensorflow model, into individual weight file under weights folder

load_weights.py
load weights from weight files, and save it into keras model (h5)


all code tested on amazon deep learning AMI, p2xlarge
Putty session with tunnel for jupyter
# start Keras2 with tensorflow python3
source activate tensorflow_p36    (source deactivate)
jupyter notebook --no-browser --port=8888

On local machine
Localhost:8888
