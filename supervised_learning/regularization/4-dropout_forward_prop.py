#!/usr/bin/env python3
"""Forward prop -- Dropout"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A = cache['A' + str(i - 1)]
        Z = np.matmul(W, A) + b
        if i < L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            A = A / keep_prob
            cache['D' + str(i)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        cache['A' + str(i)] = A
    return cache
