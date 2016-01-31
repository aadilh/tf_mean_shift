import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import Image
import six
from collections import defaultdict
import sys

tf.app.flags.DEFINE_string("log_dir","mean_shift_logs001","Directory for saving the summaries")
tf.app.flags.DEFINE_string("bandwidth",50,"Bandwidth for the kernel")
FLAGS = tf.app.flags.FLAGS



def mean_shift(X, bandwidth, max_iter):

    (m,n) = X.shape
    print m,n
    graph = tf.Graph()
    with graph.as_default():

        with tf.name_scope("input") as scope:
            data = tf.constant(X,dtype=tf.float32, name="data_points")
            b = tf.constant(bandwidth,dtype=tf.float32, name="bandwidth")
            max_i = tf.constant(max_iter, name="maximum_iteration")
            n_samples = tf.constant(m, name="no_of_samples")
            n_features = tf.constant(n, name="no_of_features")
            # k = tf.constant(kernel,name="kernel")

        # with tf.name_scope("seeding") as scope:
        #     seed = tf.placeholder(tf.float32, [5], name="seed")

        with tf.name_scope("mean_shifting") as scope:
            old_mean = tf.placeholder(tf.float32, [n], name="old_mean")
            neighbors = tf.placeholder(tf.float32, [None,n], name="neighbors")
            new_mean = tf.reduce_mean(neighbors,0)

            euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(old_mean, new_mean), 2)), name="distance_between_means")

        with tf.name_scope("seeding") as scope:
            binned_points = tf.floordiv(data,b,"projection_of_data_point")

        center_intensity_dict = {}
        nbrs = NearestNeighbors(radius=bandwidth).fit(X)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph_def)

        bps = sess.run(binned_points)
        bin_sizes = defaultdict(int)
        for bp in bps:
            bin_sizes[tuple(bp)]+=1

        bin_seeds = np.array([point for point, freq in six.iteritems(bin_sizes) if freq >= 1], dtype=np.float32)

        bin_seeds = bin_seeds*bandwidth

        n_seeds = len(bin_seeds)

        print n_seeds


        j=0
        i=0
        maxi =0

        for x in bin_seeds:

            sys.stdout.write("Completed "+str(j)+"/"+str(n_seeds)+" seeds \r")
            sys.stdout.flush()
            if (i>maxi):
                maxi=i
            i = 0
            o_mean=x

            while True:
                i_nbrs = nbrs.radius_neighbors([o_mean], bandwidth, return_distance=False)[0]
                points_within = X[i_nbrs]

                feed = {neighbors: points_within}
                n_mean = sess.run(new_mean, feed_dict=feed)

                feed = {new_mean: n_mean, old_mean: o_mean}
                dist = sess.run(euclid_dist, feed_dict=feed)

                if dist < 1e-3*bandwidth or i==max_iter:
                    center_intensity_dict[tuple(n_mean)] = len(i_nbrs)
                    break
                else:
                    o_mean = n_mean

                # print "\t",i,dist,len(i_nbrs)

                i+=1

            # if j>10:
            #     break

            j+=1

        sys.stdout.write("Completed 135/135 seeds MaxI: "+str(maxi)+"\n")
        sys.stdout.flush()

        # print center_intensity_dict

        sorted_by_intensity = sorted(center_intensity_dict.items(),key=lambda tup: tup[1], reverse=True)
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=np.bool)
        nbrs = NearestNeighbors(radius=bandwidth).fit(sorted_centers)
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center],return_distance=False)[0]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]

        nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
        labels = np.zeros(m, dtype=np.int)
        distances, idxs = nbrs.kneighbors(X)

        labels = idxs.flatten()
        return cluster_centers, labels


im_name = "86000"

im = Image.open("images/"+im_name+".jpg")
X = np.array(im)
r,c,_ = X.shape

X = X.reshape(r*c,3)

cluster_centers,labels = mean_shift(X, 40, 300)

number_of_clusters = len(np.unique(labels))

labels = np.reshape(labels,[r,c])
# print labels.shape
# print labels
# print cluster_centers
segmented = np.zeros((r,c,3),np.uint8)

for i in range(r):
    for j in range(c):
            segmented[i][j] = cluster_centers[labels[i][j]][0:3]

Image.fromarray(segmented).save("segmented_"+im_name+".jpg")
print "Number of clusters: ",number_of_clusters,"\t"
