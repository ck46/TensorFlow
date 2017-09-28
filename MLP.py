# -*- coding: utf-8 -*-

import tensorflow as tf

class MLP(object):
    def __init__(self, n_in, hidden, n_out):
        self.x = tf.placeholder("float", [None, n_in])
        self.y_ = tf.placeholder("float", [None, n_out])
        self.n_in = n_in
        self.h_layer = hidden
        self.n_out = n_out

        # define tensorflow variables
        self.W1 = tf.Variable(tf.truncated_normal([n_in, hidden], stddev=0.1))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[hidden]))
        self.W2 = tf.Variable(tf.truncated_normal([hidden, n_out], stddev=0.1))
        self.b2 = tf.Variable(tf.constant(0.1, shape=[n_out]))

        # Session and initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def output(self):
        h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        return tf.nn.softmax(tf.matmul(h1, self.W2) + self.b2)

    def predict(self, x):
        h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        y = tf.nn.softmax(tf.matmul(h1, self.W2) + self.b2)
        return self.sess.run(y, {self.x: [x]})

    def train(self, iters, get_data):
        h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1) #
        y = tf.nn.softmax(tf.matmul(h1, self.W2) + self.b2)
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(y))
        # training step
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        for _ in range(iters):
            batch_xs, batch_ys = get_data()
            self.sess.run(train_step,
                          feed_dict={self.x: batch_xs, self.y_: batch_ys})
        pass

    # TO DO: training using momentum

    def evaluate(self, test_inputs, test_outputs):
        y = self.output()
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.sess.run(accuracy,
                        feed_dict={self.x: test_inputs, self.y_: test_outputs})
    pass
