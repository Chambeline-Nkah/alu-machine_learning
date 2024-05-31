#!/usr/bin/env python3
"""gradient descent -- Dropout"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """gradient descent"""
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = np.matmul(dz, A.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - np.square(A)) * \
                cache['D' + str(i - 1)]
            dz = dz / keep_prob
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
