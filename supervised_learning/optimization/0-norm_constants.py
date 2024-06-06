#!/usr/bin/env python3
"""Normalization"""


import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    return X.mean(axis=1), X.std(axis=1)
