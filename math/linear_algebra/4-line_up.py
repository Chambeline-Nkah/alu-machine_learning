#!/usr/bin/env python3
"""Function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""
    sum = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        if isinstance(arr1[i], list) and len(arr1[i]) != len(arr2[i]):
            return None
        elif not isinstance(arr1[i], list):
            sum.append(arr1[i] + arr2[i])
        elif isinstance(arr1[i], list):
            sum.append([])
            for j in range(len(arr1[i])):

                sum[i].append(arr1[i][j] + arr2[i][j])
    return sum
