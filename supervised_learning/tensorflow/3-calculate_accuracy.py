#!/usr/bin/env python3
""" create layer """


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    a_pred = tf.arg_max(y_pred, 1)
    b_inp = tf.arg_max(y, 1)
    e = tf.equal(b_inp, a_pred)
    return tf.reduce_mean(tf.cast(e, tf))
