#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for bidirectional RNN
    """

    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, 2, m, h))
    H[0, 0] = h_0
    H[0, 1] = h_T
    for step in range(t):
        h_prev, y = bi_cell.forward(H[step, 0], X[step])
        H[step + 1, 0] = h_prev
        h_next, y = bi_cell.forward(H[step, 1], y)
        H[step + 1, 1] = h_next
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, 2, m, output_shape)
    return (H, Y)
