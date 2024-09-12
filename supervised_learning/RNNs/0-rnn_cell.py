#!/usr/bin/env python3
"""Recurrent neural network"""

import numpy as np


class RNNCell:
    """class RNN
    class constructor: def __init__(self, i, h, o)
    """

    def __init__(self, i, h, o):
        """i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wh, Wy, bh,
        by that represent the weights and biases of the cell
        Wh and bh are for the concatenated hidden state and input data
        Wy and by are for the output
        The weights should be initialized using a random normal
        distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros"""

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    # Defining the activation function to be used: softmax

    def softmax(self, x):
        """activation fxn (softmax) where
        x is the value to perform softmax"""

        fxn = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = fxn / fxn.sum(axis=1, keepdims=True)
        return softmax

    def forward(self, h_prev, x_t):
        """Performing forward propagation for one time step
        - x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
        - m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state

        The output of the cell should use a softmax activation function
        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell"""

    # ht+1 = f(Xt, ht, Wx, Wh, bh) = f[(Wx . Xt) + (Wh . ht) + bh]
    # yt = f(ht, Wy) = f[(Wy . ht) + by] where f is the activation fxn

        summation = np.concatenate((h_prev, x_t), axis=1)
        h_nxt = np.tanh(np.matmul(summation, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_nxt, self.Wy) + self.by)
        return h_nxt, y
