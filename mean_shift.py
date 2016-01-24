import numpy as np
import tensorflow as tf
import Image
import six
from collections import defaultdict
import sys

tf.app.flags.DEFINE_string("logdir","logs","Directory for saving the logs of computation")
tf.app.flags.DEFINE_string("kernel","flat","Kernel function to be used ")
tf.app.flags.DEFINE_integer("bandwidth",50,"Bandwidth for the kernel")
tf.app.flags.DEFINE_integer("max_iter",300,"Maximum number of iteration for mean updates")
tf.app.flags.DEFINE_integer("min_bin_freq",1,"Minimum frequency for a binned point to be taken as a seed")

FLAGS = tf.app.flags.FLAGS
