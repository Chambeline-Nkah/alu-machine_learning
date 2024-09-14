#!/usr/bin/env python3
"""Long Short Term Memory Unit"""

import numpy as np


class LSTMCell:
    """class LSTM
    class constructor: def __init__(self, i, h, o)
    """

    def __init__(self, i, h, o):
        """class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes
        Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo,
        by that represent the weights and biases of the cell

        Wf and bf are for the forget gate
        Wu and bu are for the update gate
        Wc and bc are for the intermediate cell state
        Wo and bo are for the output gate
        Wy and by are for the outputs

        The weights should be initialized using a random
        normal distribution in the order listed above
        The weights will be used on the right
        side for matrix multiplication
        The biases should be initialized as zeros"""

        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

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

    def forward(self, h_prev, c_prev, x_t):
        """Function that performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell
        m is the batche size for the data

        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        c_prev is a numpy.ndarray of shape (m, h)
        containing the previous cell state
        The output of the cell should use
        a softmax activation function

        Returns: h_next, c_next, y
        h_next is the next hidden state
        c_next is the next cell state
        y is the output of the cell"""

        summation = np.concatenate((h_prev, x_t), axis=1)
        forget_gate = self.sigmoid(np.matmul(summation, self.Wf) + self.bf)
        update_gate = self.sigmoid(np.matmul(summation, self.Wu) + self.bu)
        icell_state = np.tanh(np.matmul(summation, self.Wc) + self.bc)

        c_nxt = forget_gate * c_prev + update_gate * icell_state
        output_gt = self.sigmoid(np.matmul(summation, self.Wo) + self.bo)
        h_nxt = output_gt * np.tanh(c_nxt)
        y = self.softmax(np.matmul(h_nxt, self.Wy) + self.by)
        return h_nxt, c_nxt, y
