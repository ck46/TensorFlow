#-------------------------------------------------#
# A class for creating and training a single softmax
# layer using tensorflow.
#-------------------------------------------------#

import tensorflow as tf

class Softmax(object):
    def __init__(self, n_in, n_out):
        self.x = tf.placeholder("float", [None, n_in])
        self.y_ = tf.placeholder("float", [None, n_out])
        self.W = tf.Variable(tf.zeros([n_in, n_out]))
        self.b = tf.Variable(tf.zeros([n_out]))
        self.n_in = n_in
        self.n_out = n_out
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def output(self):
        return tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)

    def predict(self, x):
        y = self.output()
        return self.sess.run(y, {self.x: [x]})

    def train(self,iters, train_data, batch_size):
        y = self.output()
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        for _ in range(iters):
            batch_xs, batch_ys = next_batch(train_data, batch_size)
            self.sess.run(train_step,
                          feed_dict={self.x: batch_xs, self.y_: batch_ys})
        pass

    def evaluate(self, test_inputs, target_outputs):
        y = self.output()
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.sess.run(accuracy,
                        feed_dict={self.x: test_inputs, self.y_: target_outputs})

    pass

# This function has to be defined according to how data is presented.
def next_batch(data, batch_size):
    """
    This particular function was define for data presented by input_data.py from
    tensorflow tutorials @ https://tensorflow.org/tutorial/mnist/tf/index.html
    """
    return data.next_batch(batch_size)
