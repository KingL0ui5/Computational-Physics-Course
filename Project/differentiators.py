"""
A module contianing the finite difference diffrentiators to evaulate the local energies of wavefunctions.
Louis Liu 21/11
"""

import numpy as np
from typing import Callable


def double_central_difference(f: Callable, x: np.ndarray, h: float = 1e-5, order: int = 2) -> float | np.ndarray:
    """
    Compute the second derivative of a function using the double central difference method.
    Order O(h^2), O(h^4), O(h^6), O(h^8), or O(h^10)
    Parameters:
        f: callable, The function to differentiate
        x: 2d np.ndarray, The position(s) at which to evaluate the second derivative
        h: float, The step size for the finite difference
        order: int, The order of the finite difference approximation (2, 4, 6, 8, or 10)
    Returns:
        float | np.ndarray: The second derivative of f at position x
    """
    x = np.asarray(x, dtype=float)
    n_dims, n_samples = x.shape
    d2f = np.zeros_like(x)

    coeffs = {
        2:  ([1, -2, 1], 1),
        4:  ([-1, 16, -30, 16, -1], 12),
        6:  ([2, -27, 270, -490, 270, -27, 2], 180),
        8:  ([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9], 5040),
        10: ([5, -72, 495, -2200, 6600, -12650, 6600, -2200, 495, -72, 5], 27720)
    }

    if order not in coeffs:
        raise ValueError(f"Order {order} not supported")

    weights, divisor = coeffs[order]
    weights = np.array(weights)
    radius = len(weights) // 2
    k_steps = np.arange(-radius, radius + 1)

    for i in range(n_dims):
        val_sum = np.zeros(n_samples)
        current_h = h[i]

        for k, w in zip(k_steps, weights):
            x_perturbed = x.copy()
            x_perturbed[i, :] += k * current_h

            res = f(x_perturbed)

            val_sum += w * np.asarray(res).flatten()

        d2f[i, :] = val_sum / (divisor * current_h**2)

    return d2f


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
