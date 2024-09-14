#!/usr/bin/env python3
"""Deep Recurrent Neural Network"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward propagation for a deep RNN:

        rnn_cells is a list of RNNCell instances of length l
        that will be used for the forward propagation

        l is the number of layers
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size

        i is the dimensionality of the data
        h_0 is the initial hidden state, given
        as a numpy.ndarray of shape (l, m, h)
        h is the dimensionality of the hidden state

        Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    layers = len(rnn_cells)
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0
    for step in range(t):
        for layer in range(layers):
            if layer == 0:
                h_prev = X[step]
            h_prev, y = rnn_cells[layer].forward(H[step, layer], h_prev)
            H[step + 1, layer, ...] = h_prev
            if layer == layers - 1:
                if step == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)
    return (H, Y)
