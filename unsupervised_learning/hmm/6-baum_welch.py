#!/usr/bin/env python3
"""Performing the Baum-Welch algorithm for a hidden markov model"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Function that performs the Baum-Welch
        algorithm for a hidden markov model

    Observations is a numpy.ndarray of shape (T,)
        that contains the index of the observation
    T is the number of observations
    Transition is a numpy.ndarray of shape (M, M)
        that contains the initialized transition probabilities
    M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N)
        that contains the initialized emission probabilities
    N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that
        contains the initialized starting probabilities
    iterations is the number of times
        expectation-maximization should be performed

    Returns: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) < 1:
        return None, None

    T = Observations.shape[0]

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None

    M, M_check = Transition.shape
    if M != M_check:
        return None, None

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    M_check, N = Emission.shape
    if M_check != M:
        return None, None

    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    M_check, one = Initial.shape
    if M_check != M or one != 1:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None
    return None, None
