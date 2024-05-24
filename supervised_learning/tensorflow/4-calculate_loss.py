#!/usr/bin/env python3
"""softmax cross-entropy loss"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
