#!/usr/bin/env python3
"""Normalization"""


import numpy as np


def normalize(X, m, s):
    """normalize
    m contains the mean of all features of X
    s contains the standard deviation of all features of X"""
    return (X - m) / s
