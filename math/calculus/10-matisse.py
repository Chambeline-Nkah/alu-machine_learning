#!/usr/bin/env python3
"""Writing a function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Getting the derivative of a polynomial"""

    deriv = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(len(poly)-1, 0, -1):

        deriv.append(poly[i]*i)
    return deriv[::-1]
