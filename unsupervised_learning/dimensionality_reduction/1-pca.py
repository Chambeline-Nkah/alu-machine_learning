#!/usr/bin/env python3
"""Performing PCA on a dataset"""

import numpy as np


def pca(X, ndim):
    """X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
        containing the transformed version of X"""

    avg = np.mean(X, axis=0, keepdims=True)
    A = X - avg
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
