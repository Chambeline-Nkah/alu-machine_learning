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
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if Observations.shape[0] == 0:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0]:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[0] != Emission.shape[0] or Initial.shape[1] != 1:
        return None, None

    return None, None
