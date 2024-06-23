#!/usr/bin/env python3
"""Renaming columns in a dataset"""


import pandas as pd


from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Renaming column 'Timestamp' to 'Datetime'
df = df.rename(columns={'Timestamp': 'Datetime'})

#datetime values
df['Datetime'] = pd.to_datetime(df['Datetime'])

#Output
print(df[['Datetime', 'Close']])
