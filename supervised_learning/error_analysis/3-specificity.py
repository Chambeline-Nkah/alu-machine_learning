#!/usr/bin/env python3
"""Specificity"""

import numpy as np


def specificity(confusion):
    """Calculating the specificity for each class in a confusion matrix"""
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (TP + FN + FP)
    return TN / (TN + FP)