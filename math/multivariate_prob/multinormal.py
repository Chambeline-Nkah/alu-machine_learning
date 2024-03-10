#!/usr/bin/env python3
"""Creating class MultiNormal"""
import numpy as np


class MultiNormal():
    """Class that represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """Function that calculates the mean and covariance of a data set"""
        d, n = X.shape
        m = np.mean(X, axis=1, keepdims=True)
        C = np.matmul((X - m), (X - m).T) / (n - 1)
        return m, C
