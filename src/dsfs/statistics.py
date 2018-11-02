"""Module containing functions from Chapter 5."""
import math
from collections import Counter

from src.dsfs.vector import sum_of_squares, dot


# Central tendencies
def mean(x):
    """Mean of a vector."""
    return sum(x) / len(x)


def median(v):
    """Median of v vector."""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    if n % 2 == 1:
        # odd
        return sorted_v[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2


def quantile(x, p):
    """Returns the pth-percentile value in x."""
    p_index = int(p * len(x))
    return sorted(x)[p_index]


def mode(x):
    """Returns a list of most common values in x."""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


# Dispersion
def data_range(x):
    """Returns difference between min and max values of x."""
    return max(x) - min(x)


def de_mean(x):
    """Translate x so the mean of the result is 0."""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


def variance(x):
    """Variance ( assumes x has at least 2 elements)"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(x):
    """Standard deviation."""
    return math.sqrt(variance(x))


def interquartile_range(x):
    """Difference  between 25th and 75th percentiles."""
    return quantile(x, 0.75) - quantile(x, 0.25)


# Correlation
def covariance(x, y):
    """Covariance of x and y."""
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


def correklation(x, y):
    """Correlation between x and y."""
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / (stdev_x * stdev_y)
    else:
        return 0    # if no variation, correlation is zero


