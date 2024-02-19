#!/usr/bin/env python3
"""Writing a function that performs matrix multiplcation"""


def mat_mul(mat1, mat2):
    """Function performing index multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    matrix = []
    for k in range(len(mat1)):
        matrix.append([])
        for i in range(len(mat2[0])):
            dot = 0
            for j in range(len(mat1[0])):
                dot += mat1[k][j] * mat2[j][i]
            matrix[k].append(dot)
    return matrix

# mat1 = [[1, 2],
#         [3, 4],
#         [5, 6]]
# mat2 = [[1, 2, 3, 4],
#         [5, 6, 7, 8]]
# print(mat_mul(mat1, mat2))
