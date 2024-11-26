#!/usr/bin/env python3
"""Computing to policy"""

import numpy as np


def policy(matrix, weight):
    """Function that computes to policy with a weight of a matrix"""
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))
    act_prob = exp_z / np.sum(exp_z)
    action = np.random.choice(len(act_prob), p=act_prob)
    return action

def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix."""
    action = policy(state, weight)
    z = np.dot(state, weight)
    exp_z = np.exp(z - np.max(z))
    act_prob = exp_z / np.sum(exp_z)
    gradient = np.outer(state, act_prob)
    gradient[action, :] -= state
    return action, gradient
