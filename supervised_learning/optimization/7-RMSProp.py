#!/usr/bin/env python3
"""RMSProp"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm"""