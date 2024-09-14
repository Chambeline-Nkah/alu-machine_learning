#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_T):
    '''
    Performs forward propagation for bidirectional RNN
    '''

    t, m, i = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    Hf[0] = h_0
    Hb[-1] = h_t

    for step in range(t):
        Hf[step + 1] = bi_cells.forward(Hf[step], X[step])

    for step in range(t-1, -1, -1):
        Hb[step] = bi_cells.backward(Hb[step + 1], X[step])

    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)

    Y = bi_cells.output(H)

    return H, Y
