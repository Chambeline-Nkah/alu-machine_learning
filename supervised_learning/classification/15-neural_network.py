#!/usr/bin/env python3
"""creating a neural network"""


import numpy as np
import matplotlib.pyplot as plt


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

    def cost(self, Y, A):
        """calculating cost"""
        cost = - ((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A))
        mean_cost = np.mean(cost)
        return mean_cost

    def evaluate(self, X, Y):
        """evaluate"""
        predict = self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        predict = np.where(self.__A2 >= 0.5, 1, 0)
        return (predict, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradient descent"""
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(X, dz1.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * dw2.T
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1.T
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=1000, alpha=0.05,
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

        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        steps = []
        for i in range(iterations + 1):

            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)

            if i % step == 0:
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, cost))
                if graph:
                    costs.append(cost)
                    steps.append(i)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
