"""
A module contianing the finite difference diffrentiators to evaulate the local energies of wavefunctions.
Louis Liu 21/11
"""

import numpy as np
from typing import Callable


def double_central_difference(f: Callable, x: float | np.ndarray, h: float = 1e-5, order='h2') -> float | np.ndarray:
    """
    Compute the second derivative of a function using the double central difference method.
    Order O(h^2) or O(h^4)
    Parameters:
        f: callable, The function to differentiate
        x: float | np.ndarray, The position(s) at which to evaluate the second derivative
        h: float, The step size for the finite difference
    Returns:
        float | np.ndarray: The second derivative of f at position x
    """
    x = np.asarray(x)
    if order == 'h2':
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

    elif order == 'h4':
        return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h ** 2)

    elif order == 'h6':
        return (2*f(x + 3*h) - 27*f(x + 2*h) + 270*f(x + h) - 490*f(x) + 270*f(x - h) - 27*f(x - 2*h) + 2*f(x - 3*h)) / (180 * h ** 2)

    elif order == 'h8':
        return (-9*f(x + 4*h) + 128*f(x + 3*h) - 1008*f(x + 2*h) + 8064*f(x + h) - 14350*f(x) + 8064*f(x - h) - 1008*f(x - 2*h) + 128*f(x - 3*h) - 9*f(x - 4*h)) / (5040 * h ** 2)

    elif order == 'h10':
        return (5*f(x + 5*h) - 72*f(x + 4*h) + 495*f(x + 3*h) - 2200*f(x + 2*h) + 6600*f(x + h) - 12650*f(x) + 6600*f(x - h) - 2200*f(x - 2*h) + 495*f(x - 3*h) - 72*f(x - 4*h) + 5*f(x - 5*h)) / (27720 * h ** 2)

    else:
        raise ValueError(f"Unknown order '{order}'")


def double_forward_difference(f: Callable, x: float | np.ndarray, h: float = 1e-5) -> float | np.ndarray:
    """
    Compute the second derivative of a function using the double forward difference method
    Order O(h^2)
    Parameters:
        f: callable, The function to differentiate
        x: float | np.ndarray, The position(s) at which to evaluate the second derivative
        h: float, The step size for the finite difference
    Returns:
        float | np.ndarray: The second derivative of f at position x
    """
    x = np.asarray(x)
    return (2*f(x) - 5*f(x + h) + 4*f(x + 2*h) - f(x + 3*h)) / (h ** 2)


def double_backward_difference(f: Callable, x: float | np.ndarray, h: float = 1e-5) -> float | np.ndarray:
    """
    Compute the second derivative of a function using the double backward difference method
    Order O(h^2)
    Parameters:
        f: callable, The function to differentiate
        x: float | np.ndarray, The position(s) at which to evaluate the second derivative
        h: float, The step size for the finite difference
    Returns:
        float | np.ndarray: The second derivative of f at position x
    """
    x = np.asarray(x)
    return (2*f(x) - 5*f(x - h) + 4*f(x - 2*h) - f(x - 3*h)) / (h ** 2)
