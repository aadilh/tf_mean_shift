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

def filter_binned_points(binned_points,min_bin_freq):

    bin_freqs = defaultdict(int)

    for bp in binned_points:
        bin_freqs[tuple(bp)]+=1

    fbps = np.array([point for point, freq in six.iteritems(bin_freqs) if freq>min_bin_freq], dtype=np.float32)

    return fbps


def mean_shift(X, b, m_i, k, m_b_f):

    (m,n) = X.shape
    print "Number of points: ",m
    print "Number of features: ",n

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("input") as scope:
            data_points = tf.constant(X,dtype=tf.float32, name="data_points")
            bandwidth = tf.constant(b,dtype=tf.float32, name="bandwidth")
            max_iter = tf.constant(m_i, name="maximum_iterations")
            kernel = tf.constant(k, name=k+"_kernel")
            n_samples = tf.constant(m, name="no_of_samples")
            n_features = tf.constant(n, name="no_of_features")
            min_bin_freq = tf.constant(m_b_f, name="min_bin_freq")

        with tf.name_scope("generate_seeds") as scope:
            binned_points = tf.floordiv(data_points,bandwidth,name="binned_points")
            f_binned_points = tf.py_func(filter_binned_points,[binned_points,min_bin_freq],[tf.float32],name="filtered_binned_points")[0]
            seeds = tf.mul(f_binned_points,bandwidth, name="seeds")


        sess = tf.Session()
        writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)

        s = sess.run(seeds)

        print len(s)
