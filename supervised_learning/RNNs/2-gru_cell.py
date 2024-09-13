#!/usr/bin/env python3
"""Gated Recurrent Unit"""

import numpy as np


class GRUCell:
    """class GRU
    class constructor: def __init__(self, i, h, o)
    """
    
    def __init__(self, i, h, o):
        """class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
        Wz and bz are for the update gate
        Wr and br are for the reset gate
        Wh and bh are for the intermediate hidden state
        Wy and by are for the output

        The weights should be initialized using a random
        normal distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros"""
        
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
    
    def softmax(self, x):
        """activation fxn (softmax) where
        x is the value to perform softmax"""

        fxn = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = fxn / fxn.sum(axis=1, keepdims=True)
        return softmax
    
    def sigmoid(self, x):
        """activation fxn (sigmoid) where
        X is the value to perform the sigmoid on"""
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid