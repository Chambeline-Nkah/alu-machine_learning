#!/usr/bin/env python3
"""Writing a function that performs summation"""


def summation_i_squared(n):
    """Function that sums values of i ** 2"""

    if n == 1:
        return 1
    if n < 1:
        return None
    else:
        result = (n*(n+1)*(2*n+1))//6
        return result
