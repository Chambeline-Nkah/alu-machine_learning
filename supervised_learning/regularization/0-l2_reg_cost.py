#!/usr/bin/env python3
"""L2 regularization - weight decay"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculating cost of nn with L2"""

    penality = 0

    for i in range(1, L + 1):
        key = 'W' + str(i)
        penality += np.sum(np.square(weights[key]))

    penality *= (lambtha / (2 * m))

    total_cost = cost + penality

    return total_cost
