#!/usr/bin/env python3
"""Determining if a markov chain is absorbing"""

import numpy as np


def absorbing(P):
    """Function that determines if a markov chain is absorbing

    P is a is a square 2D numpy.ndarray of shape (n, n)
        representing the standard transition matrix
    P[i, j] is the probability of transitioning
        from state i to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    n, n = P.shape
    if n != P.shape[0]:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False
    if np.any(np.diag(P) == 1):
        return True
    return False
