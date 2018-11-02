"""Module containing functions from Chapter 8."""
import random

from src.dsfs.vector import distance


def sum_of_squares(v):
    """Computes the sum of squared elements of v."""
    return sum(v_i ** 2 for v_i in v)


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f, v, i, h):
    """Compute the ith partial difference quotient of f at v."""
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]


def step(v, direction, step_size):
    """Move step_size in the direction from v."""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


def gradient_example():
    # pick a random starting point
    v = [random.randint(-10, 10) for _ in range(3)]
    tolerance = 0.0000001
    i = 0
    while True:
        i += 1
        gradient = sum_of_squares_gradient(v)
        next_v = step(v, gradient, -0.01)
        if distance(next_v, v) < tolerance:
            break
        v = next_v
    return v


def safe(f):
    """return a new function that's the same as f,
        except that it outputs infinity whenever f produces an error"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf') # this means "infinity" in Python
    return safe_f
