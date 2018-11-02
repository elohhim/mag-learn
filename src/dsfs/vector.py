"""Module containing functions from Chapter 4 section Vectors."""
import math
import operator
from functools import reduce


def apply_to_corr_elements(func, v, w):
    """Applies function to corresponding vector elements."""
    return [func(v_i, w_i) for v_i, w_i in zip(v, w)]


def vector_add(v, w):
    """Adds two vectors by adding corresponding elements."""
    return apply_to_corr_elements(operator.add, v, w)


def vector_subtract(v, w):
    """Subtracts two vectors by subtracting corresponding elements."""
    return apply_to_corr_elements(operator.sub, v, w)


def vector_sum(vectors):
    """Sums vectors by summing all corresponding elements."""
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    """Multiplies vector v by scalar c."""
    return [c * v_i for v_i in v]


def dot(v, w):
    """Dot product of two vectors."""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


def magnitude(v):
    """Magnitude (or length) of vector."""
    return math.sqrt(sum_of_squares(v))


def distance(v, w):
    return magnitude(vector_subtract(v, w))