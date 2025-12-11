"""
Defining the 1D harmonic oscillator eigenfunctions for the dimensionless hamiltonian (6)
Louis Liu 19/11
"""

import numpy as np
from modules.function_sampling import metropolis_hastings, MALA
from modules.differentiators import double_central_difference
from modules.helpers import harmonic_oscillator_helpers as hlp


class wavefunction:
    def __init__(self, n: int):
        self.n = n
        self.f, self.E = hlp.eigenfunctions(n)

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
    return E


if __name__ == "__main__":
    N_s = 100000
    for n in range(5):
        psi = wavefunction(n=n)
        x = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[1.], xmin=[-10.], xmax=[10.], N=N_s, kwrgs={
            'sigma': 1.2})

        # x = MALA(f=psi.probability_density, timestep=0.15, x_0=[
        #     1.4], N=N_s, xmin=[-10.], xmax=[10.], order=8, stepsize=8.008e-03, detail=True)

        #  discard burn in
        x = x[N_s//10:, :]

        localenergy_arr = localenergy(psi, x)
        exp_energy = np.mean(localenergy_arr)
        energy_err = np.std(localenergy_arr) / np.sqrt(len(localenergy_arr))

        print(
            f"Expected Energy: {exp_energy} ± {energy_err}, Theoretical Energy: {psi.energy()}, error = {np.abs(exp_energy - psi.energy())}")

    # import matplotlib.pyplot as plt
    # plt.hist(x, bins=50, density=True, alpha=0.6,
    #          label='Sampled Probability Density')
    # x_vals = np.linspace(-10, 10, 200)
    # plt.plot(x_vals, psi.probability_density(
    #     x_vals), 'r-', label='Theoretical Probability Density')
    # plt.title(f'Harmonic Oscillator n={n} Wavefunction Sampling')
    # plt.xlabel('x')
    # plt.ylabel('Probability Density')
    # plt.legend()
    # plt.show()
