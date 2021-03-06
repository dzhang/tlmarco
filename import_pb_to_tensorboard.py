'''
read TF saved model
generate data for Tensorboard, so the graph can be viewed later
'''
import argparse
import sys

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.summary import summary

def import_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

  Args:
    model_dir: The location of the protobuf (`pb`) model to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.

  Usage:
    Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
    #graph = tf.get_default_graph()
    graph = sess.graph
    
    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))
          
    # Print all operation names
    for op in graph.get_operations():
      print(op)

def main(unused_args):
  import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      required=True,
      help="The location of the protobuf (\'pb\') model to visualize.")
  parser.add_argument(
      "--log_dir",
      type=str,
      default="",
      required=True,
      help="The location for the Tensorboard log to begin visualization from.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)