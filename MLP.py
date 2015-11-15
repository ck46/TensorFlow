import tensorflow as tf

class MLP(object):
    def __init__(self, x_, y_, n_in, hidden, n_out):
        self.x = x_
        self.y = y_

        # construct hidden layers
