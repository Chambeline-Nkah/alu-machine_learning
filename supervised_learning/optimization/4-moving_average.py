#!/usr/bin/env python3
"""Weighted moving average"""


import numpy as np


def moving_average(data, beta):
    """Calculating the weighted moving average of a dataset"""
    v = 0
    result = []
    for x in range(len(data)):
        v = beta * v + (1 - beta) * data[x]
        b = 1 - (beta ** (x + 1))
        result.append(v / b)
    return result
