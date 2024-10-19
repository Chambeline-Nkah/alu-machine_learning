#!/usr/bin/env python3
"""Calculating the probability density function of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability
        density function of a Gaussian distribution
    X is a numpy.ndarray of shape (n, d)
        containing the data points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,)
        containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d)
        containing the covariance of the distribution

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,)
            containing the PDF values for each data point
        All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or d != S.shape[0] or d != S.shape[1]:
        return None
    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    fac = 1 / np.sqrt(((2 * np.pi) ** d) * S_det)
    X_m = X - m
    X_m_dot = np.dot(X_m, S_inv)
    X_m_dot_X_m = np.sum(X_m_dot * X_m, axis=1)
    P = fac * np.exp(-0.5 * X_m_dot_X_m)
    return np.maximum(P, 1e-300)
