# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys

class RBM(object):
    def __init__(self, n_visible, n_hidden, rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if rng is None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng
        # define tensorflow variables
        self.W = tf.Variable(tf.random_uniform([n_visible, n_hidden],seed=123))
        self.hbias = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
        self.vbias = tf.Variable(tf.constant(0.1, shape=[n_visible]))

        # Session and initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        pass

    def train(self,iters, get_data):
        for i in range(iters):
            batch_xs = get_data()
            self.contrastive_divergence(batch_xs)

    def contrastive_divergence(self, x_in, lr=0.1, k=1):
        """ CD-k """
        ph_mean, ph_sample = self.sample_h_given_v(x_in)
        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(
                    chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(
                    nh_samples)

        self.W += lr * (tf.matmul(tf.transpose(x_in), ph_mean)
                        - tf.matmul(tf.transpose(nv_samples), nh_means))
        self.vbias += lr * tf.reduce_mean(x_in - nv_samples, 0)
        self.hbias += lr * tf.reduce_mean(ph_mean - nh_means, 0)
        pass

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.sess.run(self.propup(v0_sample))
        h1_sample = self.rng.binomial(size=h1_mean.shape,
                                      n=1,
                                      p=h1_mean)
        return [tf.to_float(h1_mean), tf.to_float(h1_sample)]

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.sess.run(self.propdown(h0_sample))
        v1_sample = self.rng.binomial(size=v1_mean.shape,
                                      n=1,
                                      p=v1_mean)
        return [tf.to_float(v1_mean), tf.to_float(v1_sample)]

##    ### Trial tensorflow 
##    def sample_v_given_h(self, h0_sample):
##        v1_mean = self.propdown(h0_sample)
##        v1_sample = tf.less(tf.random_normal(
##            tf.shape(v1_mean), mean=0.5, stddev=0.1, seed=123), v1_mean)
##        return [v1_mean, tf.to_float(v1_sample)]
##    def sample_h_given_v(self, v0_sample):
##        h1_mean = self.propup(v0_sample)
##        h1_sample = tf.less(tf.random_normal(
##            tf.shape(h1_mean), mean=0.5, stddev=0.1, seed=123), h1_mean)
##        return [h1_mean, tf.to_float(h1_sample)]
##    ###

    def propup(self, v):
        pre_sigmoid_activation = tf.matmul(v, self.W) + self.hbias
        return tf.nn.sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = tf.matmul(h, tf.transpose(self.W)) + self.vbias
        return tf.nn.sigmoid(pre_sigmoid_activation)

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)

        return [h1_mean, h1_sample,
                v1_mean, v1_sample]

    def get_reconstruction_cross_entropy(self, x_in):
        pre_sigmoid_activation_h = tf.matmul(x_in, self.W) + self.hbias
        sigmoid_activation_h = tf.nn.sigmoid(pre_sigmoid_activation_h)
        pre_sigmoid_activation_v = tf.matmul(sigmoid_activation_h,
                                             tf.transpose(self.W)) + self.vbias
        sigmoid_activation_v = tf.nn.sigmoid(pre_sigmoid_activation_v)
        cross_entropy = -tf.reduce_mean(
            tf.reduce_sum(x_in * tf.log(sigmoid_activation_v)
                          + (1 - x_in) * tf.log(1 - sigmoid_activation_v), 1))
        #return cross_entropy
        return self.sess.run(cross_entropy)
    
    def reconstruct(self, v):
        h = tf.nn.sigmoid(tf.matmul(v, self.W) + self.hbias)
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vbias)

    pass
