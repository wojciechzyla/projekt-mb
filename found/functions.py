from typing import Iterable
import numpy as np
import pandas as pd

# functions checks if there are different values in a series. If thats the case it returns NaN, else it returns the unique value.
def series_single_unique_val_or_nan(series: pd.Series):
    arr = series.unique()   
    length = len(arr)
    return arr[0] if length == 1 else np.NaN # if only one unique elem given return it, else return NaN

# Returns a dictionary where for every distinct element in the iterable a mapping to a unique letter is given. Complexity is linear.
def get_to_char_map(iterable: Iterable) -> dict[str, str]:
    res = {}
    letter = 'a'
    for x in iterable:
        if not x in res:
            res[x] = letter
            letter = chr(ord(letter)+1)
    return res