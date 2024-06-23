#!/usr/bin/env python3
"""Loading a dataset"""

import pandas as pd

def from_file(filename, delimiter):
    """Loading data"""
    ds = pd.read_csv(filename, delimiter=delimiter)

    return ds
