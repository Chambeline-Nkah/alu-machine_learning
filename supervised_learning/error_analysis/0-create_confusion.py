#!/usr/bin/env python3
""" Confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """creating a confusion matrix"""
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        confusion[np.argmax(labels[i]), np.argmax(logits[i])] += 1
    return confusion
