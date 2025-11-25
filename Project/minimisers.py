"""
A module containing the minimisers to find the ground state energy of wavefunctions.
Louis Liu 22/11
"""

from typing import Callable
import numpy as np


def gradient_desecent(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, max_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        max_iter: int, the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    for _ in range(max_iter):
        x = x - stepsize * df(x)
    return x, f(x)


def stochastic_GD(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, noise: float, max_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method with added noise. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        noise: float, the level of noise to introduce to the derivative
        max_iter: int, the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    for _ in range(max_iter):
        d = df(x)
        sigma = noise * stepsize
        d += np.random.normal(loc=0, scale=sigma, size=x.shape)
        x = x - stepsize * d
    return x, f(x)


def RMSProp_GD(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, forgetting: float, max_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the RMSProp adjusted gradient descent method. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimiser 
        forgetting: float, the forgetting factor of the minimiser
        max_iter: int, the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    v = np.zeros_like(x)
    for _ in range(max_iter):
        d = df(x)
        v = forgetting * v + (1 - forgetting)*(d**2)
        x = x - (stepsize / np.sqrt(v)) * d
    return x, f(x)


def quasi_newton(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, method: str = "DFP", max_iter: int = 10000) -> tuple:
    """
    A method to find the minima of a function using the quasi-newton method using a chosen method to approximate the hessian
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimiser
        method: str, default = "DFP" the method to use for the hessian approximation - allowed values "DFP", "BFGS"
        max_iter: int, default = 10000 the number of iterations to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = np.array(x_0, dtype=float)
    n_dim = len(x)
    G = np.eye(n_dim)

    def DFP(x_new, grad_new, x, grad, G):
        delta = x_new - x
        gamma = grad_new - grad

        G = G + np.outer(delta, delta) / np.dot(gamma, delta) - \
            (G @ np.outer(gamma, gamma) @ G) / np.dot(gamma, G @ gamma)
        return G

    def BFGS(x_new, grad_new, x, grad, G):
        delta = x_new - x
        gamma = grad_new - grad
        dim = len(delta)

        temp = 1.0 / np.dot(gamma, delta)
        I = np.eye(dim)
        G = (I - temp * np.outer(delta, gamma)) @ G @ (I - temp * np.outer(gamma, delta)) \
            + temp * np.outer(delta, delta)

        return G

    if method == "DFP":
        grad_update = DFP

    elif method == "BFGS":
        grad_update = BFGS

    for _ in range(max_iter):
        grad = df(x)
        x_new = x - stepsize * G @ grad
        grad_new = df(x_new)

        G = grad_update(x_new, grad_new, x, grad, G)
        x = x_new

    return x, f(x)
