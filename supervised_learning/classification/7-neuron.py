#!/usr/bin/env python3
"""creating class neuron"""


import numpy as np
import matplotlib.pyplot as plt


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

# Getter function
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

    def forward_prop(self, X):
        """forward prop"""
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """calculating cost"""
        cost = - ((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A))
        mean_cost = np.mean(cost)
        return mean_cost

    def evaluate(self, X, Y):
        """evaluate"""
        predict = self.forward_prop(X)
        cost = self.cost(Y, predict)
        predict = np.where(predict > 0.5, 1, 0)
        return (predict, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent"""
        dz = A - Y
        m = X.shape[1]
        db = 1 / m * np.sum(dz)
        dw = (1/m) * np.matmul(dz, X.T)
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """train"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        costs = []
        for i in range(iterations):

            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print('Cost after {} iterations: {}'.format(i, cost))
            if graph and i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
        if graph and costs:
            plt.plot(np.arange(0, iterations, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
