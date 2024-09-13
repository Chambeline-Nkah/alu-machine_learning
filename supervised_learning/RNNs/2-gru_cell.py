#!/usr/bin/env python3
"""Gated Recurrent Unit"""

import numpy as np


class GRUCell:
    """class GRU
    class constructor: def __init__(self, i, h, o)
    """

    def __init__(self, i, h, o):
        """class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
        Wz and bz are for the update gate
        Wr and br are for the reset gate
        Wh and bh are for the intermediate hidden state
        Wy and by are for the output

        The weights should be initialized using a random
        normal distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros"""

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """activation fxn (softmax) where
        x is the value to perform softmax"""

        fxn = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = fxn / fxn.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """activation fxn (sigmoid) where
        X is the value to perform the sigmoid on"""
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        The output of the cell should use a softmax activation function

        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell"""
        summation1 = np.concatenate((h_prev, x_t), axis=1)
        update_gate = self.sigmoid(np.matmul(summation1, self.Wz) + self.bz)
        reset_gate = self.sigmoid(np.matmul(summation1, self.Wr) + self.br)

        summation = np.concatenate((reset_gate * h_prev, x_t), axis=1)
        h_nxt = np.tanh(np.matmul(summation, self.Wh) + self.bh)
        h_nxt = update_gate * h_nxt
        h_nxt = h_nxt + (1 - update_gate) * h_prev
        y = self.softmax(np.matmul(h_nxt, self.Wy) + self.by)
        return h_nxt, y
