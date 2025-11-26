"""
Defining the 1D harmonic oscillator eigenfunctions for the dimensionless hamiltonian (6)
Louis Liu 19/11
"""

import numpy as np
from function_sampling import metropolis_hastings
from differentiators import double_central_difference
from helpers import harmonic_eigenfunctions


class wavefunction:
    def __init__(self, n: int):
        self.n = n
        self.f, self.E = harmonic_eigenfunctions(n)

    def psi(self, x):
        return self.f(x)

    def energy(self):
        return self.E

    def probability_density(self, x):
        return np.abs(self.f(x))**2


def localenergy(wf: wavefunction, x: np.ndarray) -> np.ndarray:
    """
    Compute the local energy of the harmonic oscillator at position x for a given wavefunction.
    Parameters:
        wf : wavefunction, The wavefunction object containing the eigenfunction and energy.
        x : np.ndarray, The position(s) at which to compute the local energy.
    Returns:
        np.ndarray: The local energy at the given position(s).
    """
    x = np.asarray(x)
    d2psi = double_central_difference(wf.psi, x, h=[1e-5], order=8)
    E = -0.5 * (d2psi / wf.psi(x)) + 0.5 * x**2
    mask = np.isfinite(E)
    return E[mask]


if __name__ == "__main__":
    N_s = 100000
    psi = wavefunction(n=1)
    x = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[1.], xmin=[-10.], xmax=[10.], N=N_s, kwrgs={
        'sigma': 2.})

    #  discard burn in
    x = x[N_s//10:, :]

    localenergy_arr = localenergy(psi, x)
    exp_energy = np.mean(localenergy_arr)
    print(
        f"Expected Energy: {exp_energy}, Theoretical Energy: {psi.energy()}, error = {np.abs(exp_energy - psi.energy())}")
