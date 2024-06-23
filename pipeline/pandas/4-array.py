#!/usr/bin/env python3
"""Getting the last 10 rows of columns 'High' and 'Close'
and converting them to a numpy array"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


rows = df.iloc[-10:][['High', 'Close']]

A = rows.to_numpy()

print(A)
