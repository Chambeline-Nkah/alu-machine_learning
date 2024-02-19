#!/usr/bin/env python3
"""Writing a function that concatenates two
matrices along a specific axis"""


def cat_matrices2D(matrix1, matrix2, axis=0):
    """Concatenating two matrices along axis 0"""
    if axis == 0:
        if len(matrix1[0]) != len(matrix2[0]):
            return None
        return matrix1 + matrix2
    if axis == 1:
        if len(matrix1) != len(matrix2):
            return None
        return [matrix1[i] + matrix2[i] for i in range(len(matrix1))]
    return None


# mat1 = [[1, 2], [3, 4]]
# mat2 = [[5, 6],]
# mat3 = [[7], [8]]
# mat4 = cat_matrices2D(mat1, mat2)
# mat5 = cat_matrices2D(mat1, mat3, axis=1)
# print(mat4)
# print(mat5)
# mat1[0] = [9, 10]
# mat1[1].append(5)
# print(mat1)
# print(mat4)
# print(mat5)
