#!/usr/bin/env python3
"""Adding two matrices elemnt-wise"""


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices: mat1, mat2"""
    sum = []
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if isinstance(mat1[i], list) and len(mat1[i]) != len(mat2[i]):
            return None
        elif not isinstance(mat1[i], list):
            sum.append(mat1[i] + mat2[i])
        elif isinstance(mat1[i], list):
            sum.append([])
            for j in range(len(mat1[i])):

                sum[i].append(mat1[i][j] + mat2[i][j])
    return sum
