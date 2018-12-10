import tensorflow as tf
import pickle

def get_weights():
  sess = tf.Session()
  graph=tf.Graph()
  # load both graph and variables
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'savedmodel')
  #graph = tf.get_default_graph()
  graph = sess.graph
  
  final_weights = []
  current_bn = []   # current batchnorm
  final_lr = []     # output layer

  vars_model = tf.global_variables()
  for i in range(1, len(vars_model)):   # skip global_step:0
    key = vars_model[i]
    if not "Aux" in key.name:   # skip AuxLogits
      value = sess.run(key)
      if not "Logits" in key.name:    # hidden layers
        # 4 variables per layer
        if (i-1) % 4 == 0:
          final_weights.append([value])   # conv weights, e.g. for Conv2d_0a_3x3, 3x3x3x16 (3x3 filter, 3 channel, 16 depth)
        if (i-2) % 4 == 0:   # batchnorm beta        
          current_bn = []     
          current_bn.append(value)
        elif (i-3) % 4 == 0:  # batchnorm mean
          current_bn.append(value)
        elif (i-4) % 4 == 0:  # batchnorm variance
          current_bn.append(value)
          final_weights.append(current_bn)
      elif "Logits" in key.name:    # output layer
        if not "biases" in key.name:
          final_lr.append(value)
        else:
          final_lr.append(value)
          final_weights.append(final_lr)
      else:
        final_weights.append([value])
   
  with open('weights.p', 'wb') as fp:
    pickle.dump(final_weights, fp)  

if __name__ == "__main__":
  get_weights()