#!/usr/bin/env python3
"""creating a layer -- Dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creating a layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(rate=keep_prob)
    return dropout(layer(prev))
