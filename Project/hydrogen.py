"""
Simulation of the hydrogen atom using variational Monte Carlo methods.
Louis Liu 22/11
"""

import numpy as np
from function_sampling import metropolis_hastings
from differentiators import double_central_difference
from typing import Callable


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


def gradient():
    pass


class hydrogen_wavefunction:
    def __init__(self, theta: float):
        self.f = hydrogen_anstatz(theta)

    def psi(self, coords: np.array) -> np.ndarray | float:
        """
        The trial wavefunction
        Parameters:
            coords: np.ndarray, The position(s) as an array of shape (N, 3)
        Returns:
            np.ndarray | float: The value of the trial wavefunction at the given position(s)
        """
        return self.f(coords)

    def probability_density(self, coords: np.ndarray) -> np.ndarray | float:
        """
        The probability density of the trial wavefunction
        Parameters:
            coords: np.ndarray, The position(s) as an array of shape (N, 3)
        Returns:
            np.ndarray | float: The probability density at the given position(s)
        """
        return np.abs(self.f(coords))**2


def hydrogen_local_energy(wf: hydrogen_wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    The local energy of the hydrogen atom trial wavefunction
    Parameters:
        wf: hydrogen_wavefunction, The trial wavefunction
        coords: np.ndarray, The position(s) as an array of shape (N, 3)
    Returns:
        np.ndarray: The local energy at the given position(s)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    d2psi = double_central_difference(
        wf.psi, coords, h=[1e-5, 1e-5, 1e-5], order=8)

    E = -0.5 * (np.sum(d2psi) / wf.psi(coords) -
                1 / np.sqrt(x**2 + y**2 + z**2))
    mask = np.isfinite(E)
    return E[mask]


if __name__ == "__main__":
    N_s = 1000000
    psi = hydrogen_wavefunction(theta=5)
    samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[1., 1., 1.], xmin=[-10., -10., -10.], xmax=[10., 10., 10.], N=N_s, kwrgs={
        'sigma': 2.}, detail=True)

    #  discard burn in
    samples = samples[N_s//10:]

    localenergy_arr = hydrogen_local_energy(psi, samples)
    exp_energy = np.mean(localenergy_arr)
    print(
        f"Expected Energy: {exp_energy}")
