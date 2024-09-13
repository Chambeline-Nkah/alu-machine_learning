#!/usr/bin/env python3
"""Recurrent neural network"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Function that performs forward propagation for a simple RNN
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state,
    given as a numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs"""
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    for time_step in range(t):
        h_next, y = rnn_cell.forward(H[time_step], X[time_step])
        H[time_step + 1] = h_next
        if time_step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    new_shape = Y.shape[-1]
    Y = Y.reshape(t, m, new_shape)
    return (H, Y)
