#!/usr/bin/env python3
"""creating a layer -- Dropout"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creating a layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regu = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regu)
    return layer(prev)
