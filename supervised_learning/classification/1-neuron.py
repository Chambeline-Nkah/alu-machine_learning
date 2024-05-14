#!/usr/bin/env python3
"""creating class neuron"""


import numpy as np


class Neuron:
    """Single neuron performing binary classification"""

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

#Getter function
    @property
    def W(self):
        """weights"""
        return self.__W

    @property
    def b(self):
        """bias"""
        return self.__b

    @property
    def A(self):
        """output"""
        return self.__A
