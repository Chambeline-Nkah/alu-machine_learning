#!/usr/bin/env python3
"""Calculating the intersection of obtaining this
data with the various hypothetical probabilities"""


import numpy as np


def marginal(x, n, P, Pr):
    """Funtion that calculates the intersection"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or len(Pr.shape) == len(P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i in P:
        if i < 0 or i > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
    for i in Pr:
        if i < 0 or i > 1:
            raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    factorial = np.math.factorial
    combination = factorial(n)/(factorial(x)*factorial(n-x))
    likelihood = combination * (P**x) * ((1-P)**(n-x))
    return likelihood * Pr
