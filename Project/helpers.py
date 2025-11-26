"""
A helper module containing utiltiy functions 
"""

from typing import Callable
import numpy as np


def rms(data):
    """
    Compute the root mean square of an array.
    Parameters:
        data: array-like, The input data
    Returns:
        float: The root mean square of the data
    """
    import numpy as np
    data = np.asarray(data)
    return np.sqrt(np.mean(data**2))


def hydrogen_anstatz(theta: float) -> Callable:
    """
    The hydrogen atom trial wavefunction
    Parameters:
        theta: float, The variational parameter
    Returns:
        callable: The trial wavefunction
    """
    def f(coords: np.ndarray) -> np.ndarray | float:
        coords = np.asarray(coords)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return np.exp(-theta * np.sqrt(x**2 + y**2 + z**2))
    return f


def harmonic_eigenfunctions(n: int) -> tuple:
    """
    Compute the nth eigenfunction of the 1D harmonic oscillator at position x.
    Parameters:
        n : int, The quantum number of the eigenfunction.
    Returns:
        tuple: The callable eigenfunction, and the dimensionless energy eigenvalue.
    """

    def f(x):
        x = np.asarray(x)
        H = 0
        if n == 0:
            H = 1.0

        elif n == 1:
            H = 2 * x

        else:
            h_prev = 1.0  # H_0
            h_curr = 2 * x  # H_1

            for i in range(1, n):
                h_next = (2 * x * h_curr) - (2 * i * h_prev)
                h_prev = h_curr
                h_curr = h_next

            H = h_curr
        return np.array(np.exp(-x**2 / 2) * H)

    return f, n + 0.5


def harmonic_second_derivative(n: int):
    """
    Compute the analytical second derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
    Parameters:
        n: int, The quantum number of the eigenfunction
    Returns:
        Callable: The second derivative function
    """
    eigenfunction, _ = harmonic_eigenfunctions(n)

    def f(x):
        x = np.asarray(x)
        return (x**2 - (2*n + 1)) * eigenfunction(x)
    return f


def harmonic_first_derivative(n: int):
    """
    Compute the analytical first derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
    Parameters:
        n: int, The quantum number of the eigenfunction
    Returns:
        Callable: The first derivative function
    """
    psi_n, _ = harmonic_eigenfunctions(n)

    if n == 0:
        def f_ground(x):
            x = np.asarray(x)
            return -x * psi_n(x)
        return f_ground

    psi_n_minus_1, _ = harmonic_eigenfunctions(n - 1)
    sqrt_2n = np.sqrt(2 * n)

    def f(x):
        x = np.asarray(x)
        return sqrt_2n * psi_n_minus_1(x) - x * psi_n(x)
    return f


def get_acf(series, max_lag=200):
    series = series.flatten()
    series = series - np.mean(series)
    corr = np.correlate(series, series, mode='full')
    corr = corr[len(corr)//2:]
    corr = corr / corr[0]
    return corr[:max_lag]
