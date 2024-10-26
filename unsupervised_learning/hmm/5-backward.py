#!/usr/bin/env python3
"""Performing the backward algorithm for a hidden markov model"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Function that performs the backward algorithm for a hidden markov model

    Observation is a numpy.ndarray of shape (T,)
        that contains the index of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the
        emission probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N)
        containing the transition probabilities
    Transition[i, j] is the probability of
        transitioning from the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the
        probability of starting in a particular hidden state

    Returns: P, B, or None, None on failure
    Pis the likelihood of the observations given the model
    B is a numpy.ndarray of shape (N, T)
        containing the backward path probabilities
    B[i, j] is the probability of generating the future
        observations from hidden state i at time j
    """

    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None, None
    N, M = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None, None
    N1, N2 = Transition.shape
    if N1 != N or N2 != N:
        return None, None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None, None
    N3, N4 = Initial.shape
    if N3 != N or N4 != 1:
        return None, None, None
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for i in range(T - 2, -1, -1):
        B[:, i] = np.dot(Transition, B[:, i + 1] *
                         Emission[:, Observation[i + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
