#!/usr/bin/env python3
"""
Create a class SelfAttention that inherits from
tensorflow.keras.layers.Layer to calculate the
attention for machine translation based on this paper
"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class to calculate the attention for machine translation

    class constructor:
        def __init__(self, units)

    public instance attribute:
        W: a Dense layer with units number of units,
            to be applied to the previous decoder hidden state
        U: a Dense layer with units number of units,
            to be applied to the encoder hidden state
        V: a Dense layer with 1 units,
            to be applied to the tanh of the sum of the outputs of W and U

    public instance method:
        def call(self, s_prev, hidden_states):
            takes in previous decoder hidden state and returns
                the context vector for decoder and the attention weights
    """
    def __init__(self, units):
        """
        Class constructor def __init__(self, units):
            units is an integer representing the number of
                hidden units in the alignment model

            Sets the following public instance attributes:
            W - a Dense layer with units units,
                to be applied to the previous decoder hidden state
            U - a Dense layer with units units,
                to be applied to the encoder hidden states
            V - a Dense layer with 1 units,
                to be applied to the tanh of the sum
                of the outputs of W and U
        """
        if type(units) is not int:
            raise TypeError(
                "units must be an integer")
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Public instance method def call(self, s_prev, hidden_states):
            s_prev is a tensor of shape (batch, units) containing
                the previous decoder hidden state
            hidden_states is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder
            Returns: context, weights
            context is a tensor of shape (batch, units)
                that contains the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1)
                that contains the attention weights
        """
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        wts = tf.nn.softmax(V, axis=1)
        cont = tf.reduce_sum(wts * hidden_states, axis=1)
        return cont, wts
