#!/usr/bin/env python3
"""Creating df from a dic of lists"""

import pandas as pd

data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])

print(df)
