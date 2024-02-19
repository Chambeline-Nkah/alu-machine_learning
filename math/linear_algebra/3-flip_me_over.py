#!/usr/bin/env python3
"""Getting the transpose of a matrix"""


def matrix_transpose(matrix):
    """Getting the transpose of a matrix"""
    rows = len(matrix)
    columns = len(matrix[0])

    transpose = []

    for i in range(columns):

        transposed_row = []

        for j in range(rows):

            transposed_row.append(matrix[j][i])
        transpose.append(transposed_row)
    return transpose
