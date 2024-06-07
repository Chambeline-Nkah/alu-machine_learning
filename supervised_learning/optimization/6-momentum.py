#!/usr/bin/env python3
"""Momentum"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """momentum using tf"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
