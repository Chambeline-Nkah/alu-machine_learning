#!/usr/bin/env python3
"""creating a neural network"""


import numpy as np


class NeuralNetwork:
    """neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """W1"""
        return self.__W1

    @property
    def b1(self):
        """b1"""
        return self.__b1

    @property
    def A1(self):
        """A1"""
        return self.__A1

    @property
    def W2(self):
        """W2"""
        return self.__W2

    @property
    def b2(self):
        """b2"""
        return self.__b2

    @property
    def A2(self):
        """A2"""
        return self.__A2

    def forward_prop(self, X):
        """forward prop for nn"""
        z = np.matmul(self.__W1, X) + self.__b1
        sigmoid_1 = 1 / (1 + np.exp(-z))
        self.__A1 = sigmoid_1
        z_a = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid_2 = 1 / (1 + np.exp(-z_a))
        self.__A2 = sigmoid_2
        return self.__A1, self.__A2
