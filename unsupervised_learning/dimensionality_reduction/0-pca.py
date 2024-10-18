#!/usr/bin/env python3
"""Performing PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset:

        X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        
        all dimensions have a mean of 0 across all data points
        var is the fraction of the variance 
            that the PCA transformation should maintain
        
        Returns: the weights matrix, W, that
            maintains var fraction of Xs original variance
        W is a numpy.ndarray of shape (d, nd) 
            where nd is the new dimensionality of the transformed X"""
    
    u, s, v = np.linalg.svd(X)
    y = list(x / np.sum(s) for x in s)
    vrce = np.cumsum(y)
    nd = np.argwhere(vrce >= var)
    W = v.T[:, :nd]
    return (W)
