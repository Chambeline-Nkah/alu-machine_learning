#!/usr/bin/env python3
"""RMSProp"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """RMSProp optimization algorithm"""
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    return optimizer.minimize(loss)
