#!/usr/bin/env python3
"""Batch norm"""


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a
    neural network using batch normalization"""
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Z_outpt = gamma * Z_norm + beta
    return Z_outpt
