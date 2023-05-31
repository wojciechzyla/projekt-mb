from functools import cache
from typing import Any, Callable
import numpy as np
import pandas as pd
import scipy.spatial.distance as spd

# improve series_distance_matrix for speed-up!

def series_distance_matrix_primitive_caching(series: pd.Series, compare_func: Callable[[Any, Any], float], default_value: any, normalize: bool = True) -> np.matrix:
    @cache
    def apply_compare_func(x, y, compare_func: Callable[[Any, Any], float]):
        return compare_func(x, y)

    def compare_func_wrapper(x, y, compare_func: Callable[[Any, Any], float]):
        first = x if x < y else y
        second = y if x >= y else y
        return apply_compare_func(first, second, compare_func)
    
    values = series.fillna(default_value).to_numpy().ravel()
    size = values.size
    res = np.zeros((size, size))

    for i in range(size):
        for j in range(i+1, size):
            res[i][j] = compare_func_wrapper(values[i], values[j], compare_func)
            res[j][i] = res[i][j]

    if normalize:
        normalize_factor = max(abs(res.max()),abs(res.min())) # there may exist negative distances hypothetically as compare_func can be anything
        if normalize_factor != 0:
            res = res / normalize_factor
    return res

# calculates the distance matrix for a series. If normalize==True (default) the distance matrix is normalized by dividing every distance by the maximum distance
# this function supports negative distances
def series_distance_matrix(series: pd.Series, compare_func: Callable[[Any, Any], float], default_value: any, normalize: bool = True) -> np.matrix:
    reshaped_vals = series.fillna(default_value).to_numpy().reshape(-1, 1) # to_numpy does not change the order of elements
    res = spd.pdist(reshaped_vals, lambda x,y: compare_func(x[0], y[0]))
    if normalize:
        normalize_factor = max(abs(res.max()),abs(res.min())) # there may exist negative distances hypothetically as compare_func can be anything
        if normalize_factor != 0:
            res = res / normalize_factor
    res = spd.squareform(res)
    return res

# creates a distance matrix for a dataframe based on the given attributes. Only the given attributes are considered. If no attributes are given the result is a 0 matrix of the size of the dataframe
def df_distance_matrix(df: pd.DataFrame, attr_def: list[dict['name': str, 'weight': float, 'default': any, 'compare_func': Callable[[Any, Any], float]]]) -> np.matrix:
    res = np.zeros((df.shape[0], df.shape[0]))
    relevant_attributes = [attr['name'] for attr in attr_def if attr['weight'] != 0 and attr['name'] in df.columns]
    attr_def_dict = {attr['name']: attr for attr in attr_def if attr['name'] in relevant_attributes}

    weight_sum = 0
    for attr, attr_def in attr_def_dict.items():
        res += attr_def['weight'] * series_distance_matrix(df[attr], attr_def['compare_func'], attr_def['default'], True)
        weight_sum += attr_def['weight']
    
    if weight_sum != 0:
        res = res / weight_sum

    return res