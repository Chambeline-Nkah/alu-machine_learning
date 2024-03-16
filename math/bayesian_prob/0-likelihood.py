#!/usr/bin/env python3
"""Calculating the likelihood of obtaining this data given various
hypothetical probabilities of developing severe side effects"""


import numpy as np


def likelihood(x, n, P):
    """Function that calculates the likelihood"""
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    for i in P:
        if i < 0 or i > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
    factorial = np.math.factorial
    combination = factorial(n)/(factorial(x)*factorial(n-x))
    likelihood = combination * (P**x) * ((1-P)**(n-x))
    return likelihood
