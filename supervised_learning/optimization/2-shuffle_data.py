#!/usr/bin/env python3
"""Shuffling"""


import numpy as np


def shuffle_data(X, Y):
    """shuffling with permutations"""
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]
