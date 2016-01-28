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
im_name = "86000"

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

def update_mean(old_mean, kernel, bandwidth, neighbors):

    if kernel == 1:
        #TODO: Implemented the mean shifting using gaussian kernel
        weights = np.exp(-1*norm((neighbors - old_mean)/bandwidth,axis=1))
        new_mean = np.sum(weights[:,None]*neighbors,axis=0)/np.sum(weights)

        return new_mean
    else:
        return np.mean(neighbors, axis=0)

def remove_multiples(centers, radius):
    nbrs = NearestNeighbors(radius=radius).fit(centers)
    unq = np.ones(len(centers), dtype=np.bool)

    for i, center in enumerate(centers):
        if unq[i]:
            nids = nbrs.radius_neighbors([center],return_distance=False)[0]
            unq[nids] = 0
            unq[i] = 1

    return centers[unq]

def save_segmented_image(cluster_centers,labels,r,c,kernel,b):
    fname = "results/"+im_name+"/"+kernel+"/log.txt"
    f = open("results/"+im_name+"/"+kernel+"/log.txt",'a')
    n = len(np.unique(labels))
    labels = np.reshape(labels,[r,c])
    segmented = np.zeros((r,c,3),np.uint8)

    for i in range(r):
        for j in range(c):
                segmented[i][j] = cluster_centers[labels[i][j]][0:3]

    Image.fromarray(segmented).save("results/"+im_name+"/"+kernel+"/"+str(n)+".jpg")
    f.write(str(b)+", "+str(n)+"\n")
    print "\tNumber of clusters: ",n,"\t"
    f.close()



def mean_shift(X, b, m_i, k, m_b_f,r,c):

    (m,n) = X.shape
    print "Number of points: ",m
    print "Number of features: ",n
    print "Kernel: ",k
    # print "Bandwidth: ",b
    print "Minimum Bin Frequency: ",m_b_f

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("input") as scope:
            data_points = tf.constant(X,dtype=tf.float32, name="data_points")
            bandwidth = tf.placeholder(dtype=tf.float32, name="bandwidth")
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

            new_mean = tf.py_func(update_mean,[old_mean, kernel, bandwidth, neighbors],[tf.float32], name="new_mean")[0]

            shift_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(old_mean, new_mean), 2)), name="shift_distance")

        sess = tf.Session()
        writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)

        while b<40:
            print "\tBandwidth: ",b
            sys.stdout.write("\t\tGenerating seeds\r")
            sys.stdout.flush()

            gen_seeds = sess.run(seeds,feed_dict={bandwidth: b})
            n_seeds = len(gen_seeds)

            sys.stdout.write("\t\tGenerated "+str(n_seeds)+" seeds\n")
            sys.stdout.flush()

            i=0
            cluster_centers=np.zeros((n_seeds,n),dtype=np.float32)
            nbrs = NearestNeighbors(radius=bandwidth).fit(X)


            for seed in gen_seeds:
                sys.stdout.write("\t\tCompleted Mean Shifting on "+str(i)+" seeds\r")
                sys.stdout.flush()

                o_mean = seed
                completed_iter = 0

                while True:
                    i_nbrs = nbrs.radius_neighbors([o_mean], b, return_distance=False)
                    # print len(i_nbrs[0])
                    if len(i_nbrs[0]>0):
                        cnbrs = X[i_nbrs[0]]
                        feed = {old_mean: o_mean, neighbors: cnbrs, bandwidth: b}
                        shape, n_mean, dist = sess.run([nbrs_shape, new_mean, shift_distance],feed_dict=feed)
                    else :
                        cluster_centers[i] = o_mean
                        break
                    # print completed_iter,shape, n_mean, dist

                    if dist < 1e-3*b or completed_iter == m_i :
                        cluster_centers[i] = n_mean
                        break
                    else:
                        o_mean = n_mean
                        completed_iter+=1


                i+=1

            sys.stdout.write("\t\tCompleted Mean Shifting on all seeds \t\t\t\n")
            sys.stdout.flush()

            cluster_centers = remove_multiples(cluster_centers, b)

            nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
            labels = np.zeros(m, dtype=np.int)
            dist, ids = nbrs.kneighbors(X)

            labels = ids.flatten()
            save_segmented_image(cluster_centers, labels,r,c,k,b)
            # return cluster_centers, labels

            b+=5


def main(args):

    im = Image.open("images/"+im_name+".jpg")
    X_im = np.array(im)
    r,c,_ = X_im.shape

    X = np.zeros((r,c,5),dtype=np.uint8)

    for i in range(r):
        for j in range(c):
            for k in range(5):
                if k<3 :
                    X[i][j][k] = X_im[i][j][k]
                elif k==3:
                    X[i][j][k] = i
                else :
                    X[i][j][k] = j

    X = X.reshape(r*c,5)
    print
    mean_shift(X, 5, 300,"gaussian",1,r,c)
    mean_shift(X, 5, 300,"flat",1,r,c)
    # print labels.shape
    # print labels
    # print cluster_centers


if __name__ == "__main__":
    tf.app.run()
