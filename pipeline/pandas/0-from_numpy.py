#!/usr/bin/env python3
"""Creating df"""

import pandas as pd


def from_numpy(array):
    """Creating a pandas dataframe from a numpy array"""
    columns_nw = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G']
    
    # Create the DataFrame
    df = pd.DataFrame(array, columns=columns_nw)
