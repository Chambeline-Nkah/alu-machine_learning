#!/usr/bin/env python3
"""creating a deep neural network"""


import numpy as np


class DeepNeuralNetwork:
    """deep nn"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                # He-et-al initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # He-et-al initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Zero initialization
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """hold all weights"""
        return self.__weights

    def forward_prop(self, X):
        """foward_prop of nn"""
        self.cache["A0"] = X
        for i in range(1, self.L+1):
            W = self.weights['W'+str(i)]
            b = self.weights['b'+str(i)]
            A = self.cache['A'+str(i - 1)]
            z = np.matmul(W, A) + b
            sigmoid = 1 / (1 + np.exp(-z))
            self.cache["A"+str(i)] = sigmoid
        return self.cache["A"+str(i)], self.cache
