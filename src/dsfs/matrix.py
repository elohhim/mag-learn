"""Module containing functions from Chapter 4 section Matrices."""


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A, i):
    return A[i]


def get_column(A, j):
    return [A_i[j] for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    """Returns matrix of shape num_rows x num_cols whose i,j element is
    entry_fn(i, j)."""
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def eye(size):
    """Returns identity matrix of given size."""
    return make_matrix(size, size, lambda i, j: 1 if i == j else 0)