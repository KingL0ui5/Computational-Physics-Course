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
    def f(x: np.ndarray | float, y: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        return np.exp(-theta * np.sqrt(x**2 + y**2 + z**2))
    return f


class hydrogen_wavefunction:
    def __init__(self, theta: float):
        self.f = hydrogen_anstatz(theta)

    def psi(self, x: np.ndarray | float, y: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        """
        The trial wavefunction
        Parameters:
            x: np.ndarray | float, The x position(s)
            y: np.ndarray | float, The y position(s)
            z: np.ndarray | float, The z position(s)
        Returns:
            np.ndarray | float: The value of the trial wavefunction at the given position(s)
        """
        return self.f(x, y, z)

    def probability_density(self, x: np.ndarray | float, y: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        """
        The probability density of the trial wavefunction
        Parameters:
            x: np.ndarray | float, The x position(s)
            y: np.ndarray | float, The y position(s)
            z: np.ndarray | float, The z position(s)
        Returns:
            np.ndarray | float: The probability density at the given position(s)
        """
        return np.abs(self.f(x, y, z))**2


def hydrogen_local_energy(wf: hydrogen_wavefunction, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    The local energy of the hydrogen atom trial wavefunction
    Parameters:
        wf: hydrogen_wavefunction, The trial wavefunction
        x: np.ndarray, The x position(s)
        y: np.ndarray, The y position(s)
        z: np.ndarray, The z position(s)
    Returns:
        np.ndarray: The local energy at the given position(s)
    """
    r_vec = np.column_stack((x, y, z))

    d2psi = double_central_difference(
        wf.psi, r_vec, h=[1e-5, 1e-5, 1e-5], order=8)

    E = -0.5 * (np.sum(d2psi) / wf.psi(r_vec) -
                1 / np.sqrt(x**2 + y**2 + z**2))
    mask = np.isfinite(E)
    return E[mask]


if __name__ == "__main__":
    N_s = 1000000
    psi = hydrogen_wavefunction(theta=5)
    samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[1., 1., 1.], xmin=[-10., -10., -10.], xmax=[10., 10., 10.], N=N_s, kwrgs={
        'sigma': 2.})
    print(samples.shape)

    #  discard burn in
    x, y, z = samples[N_s//10:].T

    localenergy_arr = hydrogen_local_energy(psi, x, y, z)
    exp_energy = np.mean(localenergy_arr)
    print(
        f"Expected Energy: {exp_energy}, Theoretical Energy: {psi.energy()}, error = {np.abs(exp_energy - psi.energy())}")
