import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import Image
import six
from collections import defaultdict
import sys
from sklearn.neighbors import NearestNeighbors

tf.app.flags.DEFINE_string("logdir","logs","Directory for saving the logs of computation")
tf.app.flags.DEFINE_string("kernel","flat","Kernel function to be used ")
tf.app.flags.DEFINE_integer("bandwidth",50,"Bandwidth for the kernel")
tf.app.flags.DEFINE_integer("max_iter",300,"Maximum number of iteration for mean updates")
tf.app.flags.DEFINE_integer("min_bin_freq",1,"Minimum frequency for a binned point to be taken as a seed")

FLAGS = tf.app.flags.FLAGS

KERNELS = {"flat": 0, "gaussian": 1}

def filter_binned_points(binned_points,min_bin_freq):

    bin_freqs = defaultdict(int)

    for bp in binned_points:
        bin_freqs[tuple(bp)]+=1

    fbps = np.array([point for point, freq in six.iteritems(bin_freqs) if freq>min_bin_freq], dtype=np.float32)

    return fbps

def radial_neighbors(points,radius):
    #TODO: Implement efficient nearest radial neighbors algorithm
    neighbors = np.array([point for point in points if norm(point)<=radius], dtype=np.float32)

    return neighbors

def update_mean(old_mean, kernel, neighbors):

    if kernel == "gaussian":
        #TODO: Implemented the mean shifting using gaussian kernel
        pass
    else:
        return np.mean(neighbors, axis=0)


def mean_shift(X, b, m_i, k, m_b_f):

    (m,n) = X.shape
    print "Number of points: ",m
    print "Number of features: ",n
    print "Kernel: ",k
    print "Bandwidth: ",b
    print "Minimum Bin Frequency: ",m_b_f

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("input") as scope:
            data_points = tf.constant(X,dtype=tf.float32, name="data_points")
            bandwidth = tf.constant(b,dtype=tf.float32, name="bandwidth")
            max_iter = tf.constant(m_i, name="maximum_iterations")
            kernel = tf.constant(KERNELS[k], name=k+"_kernel")
            n_samples = tf.constant(m, name="no_of_samples")
            n_features = tf.constant(n, name="no_of_features")
            min_bin_freq = tf.constant(m_b_f, name="min_bin_freq")

        with tf.name_scope("generate_seeds") as scope:
            binned_points = tf.floordiv(data_points,bandwidth,name="binned_points")
            f_binned_points = tf.py_func(filter_binned_points,[binned_points,min_bin_freq],[tf.float32],name="filtered_binned_points")[0]
            seeds = tf.mul(f_binned_points, bandwidth, name="seeds")

        with tf.name_scope("mean_shift") as scope:
            old_mean = tf.placeholder(tf.float32, [n], name="old_mean")

            with tf.name_scope("radial_neighbors") as scope:
                # shifted_points = tf.sub(data_points, old_mean, name="shifted_points")
                # neighbors = tf.py_func(radial_neighbors,[shifted_points,bandwidth],[tf.float32],name="neighbors")[0]
                #
                neighbors = tf.placeholder(tf.float32, [None,n], name="neighbors")
                nbrs_shape = tf.shape(neighbors)

            new_mean = tf.py_func(update_mean,[old_mean,kernel,neighbors],[tf.float32], name="new_mean")[0]

            shift_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(old_mean, new_mean), 2)), name="shift_distance")

        sess = tf.Session()
        writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)

        sys.stdout.write("Generating seeds\r")
        sys.stdout.flush()

        gen_seeds = sess.run(seeds)
        n_seeds = len(gen_seeds)

        sys.stdout.write("Generated "+str(n_seeds)+" seeds\n")
        sys.stdout.flush()

        i=0
        center_intensity_dict = {}
        nbrs = NearestNeighbors(radius=bandwidth).fit(X)

        for seed in gen_seeds:
            sys.stdout.write("Completed Mean Shifting on "+str(i)+" seeds\r")
            sys.stdout.flush()

            o_mean = seed
            completed_iter = 0

            while True:
                cnbrs = X[nbrs.radius_neighbors([o_mean], b, return_distance=False)[0]]
                feed = {old_mean: o_mean, neighbors: cnbrs}
                shape, n_mean, dist = sess.run([nbrs_shape, new_mean, shift_distance],feed_dict=feed)

                # print completed_iter,shape, n_mean, dist

                if dist < 1e-3*b or completed_iter == m_i :
                    center_intensity_dict[tuple(n_mean)] = shape[0]
                    break
                else:
                    o_mean = n_mean
                    completed_iter+=1


            i+=1

        sys.stdout.write("Completed Mean Shifting on all seeds \t\t\t\n")
        sys.stdout.flush()
