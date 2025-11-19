"""
Defining the 1D harmonic oscillator eigenfunctions for the dimensionless hamiltonian (6)
Louis Liu 19/11
"""

import numpy as np


def eigenfunctions(x: np.ndarray | float, n: int) -> tuple:
    """
    Compute the nth eigenfunction of the 1D harmonic oscillator at position x.
    Parameters:
        x : float | np.ndarray, The position(s) at which to evaluate the eigenfunction.
        n : int, The quantum number of the eigenfunction.
    Returns:
        tuple: The value(s) of the nth eigenfunction at x, and the dimensionless energy eigenvalue.
    """
    H = 0
    if n == 0:
        H = 1.0

    if n == 1:
        H = 2 * x

    h_prev = 1.0  # H_0
    h_curr = 2 * x  # H_1

    for i in range(1, n):
        h_next = (2 * x * h_curr) - (2 * i * h_prev)
        h_prev = h_curr
        h_curr = h_next

    H = h_curr
    return np.exp(-x**2 / 2) * H, n + 0.5


def test():
    f, _ = eigenfunctions(np.array([0.0]), 0)
