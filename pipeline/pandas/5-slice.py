#!/usr/bin/env python3
"""Slicing"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Slice taking every 60th row
df = df.iloc[::60, [df.columns.get_loc('High'), df.columns.get_loc('Low'), df.columns.get_loc('Close'), df.columns.get_loc('Volume_BTC')]]

print(df)
