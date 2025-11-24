"""
A module containing the minimisers to find the ground state energy of wavefunctions.
Louis Liu 22/11
"""

from typing import Callable
import numpy as np


def gradient_desecent(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, N_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        N_iter: int, the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    for _ in range(N_iter):
        x = x - stepsize * df(x)
    return x, f(x)


def stochastic_GD(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, noise: float, N_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method with added noise. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        noise: float, the level of noise to introduce to the derivative
        N_iter: int, the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    for _ in range(N_iter):
        d = df(x)
        sigma = noise * stepsize
        d += np.random.normal(loc=0, scale=sigma, size=x.shape)
        x = x - stepsize * d
    return x, f(x)


def RMSProp_GD(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, forgetting: float, N_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the RMSProp adjusted gradient descent method. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        forgetting: float, the forgetting factor of the minimizer
        N_iter: int, the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    v = np.zeros_like(x)
    for _ in range(N_iter):
        d = df(x)
        v = forgetting * v + (1 - forgetting)*(d**2)
        x = x - (stepsize / np.sqrt(v)) * d
    return x, f(x)


def quasi_newton(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, N_iter: int = 10000) -> tuple:
    pass
