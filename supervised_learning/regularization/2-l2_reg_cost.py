#!/usr/bin/env python3
"""L2 regularization - weight decay"""


import tensorflow as tf


def l2_reg_cost(cost):
    """Calculating cost of nn with L2"""
    return cost + tf.losses.get_regularization_losses()
