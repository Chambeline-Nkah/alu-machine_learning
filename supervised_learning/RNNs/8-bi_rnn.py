#!/usr/bin/env python3
"""
This module contains the BiRNN class.
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell: An instance of BidirectionalCell that will be
        used for the forward propagation.
        X: The data to be used, given as a numpy.ndarray of shape (t, m, i).
        h_0: The initial hidden state in the forward direction, given as a
        numpy.ndarray of shape (m, h).
        h_t: The initial hidden state in the backward direction, given as a
        numpy.ndarray of shape (m, h).

    Returns:
        H: numpy.ndarray containing all of the concatenated hidden states.
        Y: numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_prev

    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_next

    H = np.concatenate((H_forward, H_backward), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
