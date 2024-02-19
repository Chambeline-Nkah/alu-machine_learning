#!/usr/bin/env python3
"""Writing a function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Getting the integral of a polynomial"""

    integral = []

    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None

    for i in range(len(poly)-1, 0, -1):
        integral.append(poly[i]/(i+1))

    integral.append(poly[0])
    integral.append(C)

    if len(poly) == 1 and poly[0] == 0:
        integral = [C]
    for i in range(len(integral)):
        if integral[i] % 1 == 0:
            integral[i] = int(integral[i])

    return integral[::-1]
