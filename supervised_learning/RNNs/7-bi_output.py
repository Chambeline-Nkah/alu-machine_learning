#!/usr/bin/env python3
"""bidirectional cell of an RNN"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    class GRU
    class constructor: def __init__(self, i, h, o)
    """

    def __init__(self, i, h, o):
        """class constructor

        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs

        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
        that represent the weights and biases of the cell
        Whf and bhf are for the hidden states in the forward direction
        Whb and bhb are for the hidden states in the backward direction
        Wy and by are for the outputs

        The weights should be initialized using a random normal
        distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

    def forward(self, h_prev, x_t):
        """calculates the hidden state in the
        forward direction for one time step

            x_t is a numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state
            Returns: h_next, the next hidden state
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_nxt = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_nxt

    def backward(self, h_next, x_t):
        """calculates the hidden state in the
        backward direction for one time step
            x_t is a numpy.ndarray of shape (m, i) that
            contains the data input for the cell

            m is the batch size for the data
            h_next is a numpy.ndarray of shape (m, h)
            containing the next hidden state
            Returns: h_pev, the previous hidden state"""

        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        """activation fxn (softmax) where
        x is the value to perform softmax"""

        fxn = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = fxn / fxn.sum(axis=1, keepdims=True)
        return softmax

    def output(self, H):
        """
        Function that calculates all outputs for the RNN:
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions,
        excluding their initialized states

            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states

        Returns: Y, the outputs
        """

        t, m, h = H.shape

        Y = []

        for step in range(t):
            y = self.softmax(np.matmul(H[step], self.Wy) + self.by)
            Y.append(y)

        return np.array(Y)
