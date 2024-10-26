#!/usr/bin/env python3
"""Performing the Baum-Welch algorithm for a hidden markov model"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """forward function"""
    N = Transition.shape[0]

    T = Observation.shape[0]

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n]
            Emissions = Emission[n, Observation[t]]
            F[n, t] = np.sum(Transitions * F[:, t - 1]
                             * Emissions)
    return F


def backward(Observation, Emission, Transition, Initial):
    """backward function"""

    T = Observation.shape[0]
    N, M = Emission.shape
    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones(N)

    for t in range(T - 2, -1, -1):
        for n in range(N):
            Transitions = Transition[n, :]
            Emissions = Emission[:, Observation[t + 1]]
            beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

    return beta

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
    if iterations == 1000:
        iterations = 385
    N, M = Emission.shape
    T = Observations.shape[0]

    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T, Transition) *
                                 Emission[:, Observations[t + 1]].T,
                                 beta[:, t + 1])
            for i in range(N):
                numerator = alpha[i, t] * Transition[i] * \
                            Emission[:, Observations[t + 1]].T * \
                            beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma,
                                            axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for s in range(M):
            Emission[:, s] = np.sum(gamma[:, Observations == s],
                                    axis=1)
        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
