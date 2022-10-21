import numpy as np

def find_bounds(X):
    lower_bound = list(X.min(axis=1))
    upper_bound = list(X.max(axis=1))
    return lower_bound,upper_bound