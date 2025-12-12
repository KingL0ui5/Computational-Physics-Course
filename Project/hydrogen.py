"""
Simulation of the hydrogen atom using variational Monte Carlo methods.
Louis Liu 22/11
"""

import numpy as np
from modules.function_sampling import metropolis_hastings
from modules.differentiators import double_central_difference
from modules.helpers import hydrogen_atom_helpers as hlp
from modules.minimisers import hydrogen_atom_minimisers as min
import matplotlib.pyplot as plt

size = 13

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size-2)
plt.rc('figure', titlesize=size)


class hydrogen_wavefunction:
    def __init__(self, theta: float):
        self._f = hlp.anstatz(theta)
        self._theta = theta

    def psi(self, coords: np.ndarray) -> np.ndarray | float:
        """
        The trial wavefunction
        Parameters:
            coords: np.ndarray, The position(s) as an array of shape (N, 3)
        Returns:
            np.ndarray | float: The value of the trial wavefunction at the given position(s)
        """
        return self._f(coords)

    def probability_density(self, coords: np.ndarray) -> np.ndarray | float:
        """
        The probability density of the trial wavefunction
        Parameters:
            coords: np.ndarray, The position(s) as an array of shape (N, 3)
        Returns:
            np.ndarray | float: The probability density at the given position(s)
        """
        return np.abs(self._f(coords))**2

    def theta(self) -> float:
        """
        Returns the theta parameter of the wavefunction
        Returns:
            float: the value of theta for the current wavefunction
        """
        return self._theta

    def local_energy(self, coords: np.ndarray) -> np.ndarray:
        """
        The local energy of the hydrogen atom trial wavefunction
        """
        stepsize = 1.805e-02  # Optimal stepsize for order 8

        coords = np.array(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        r = np.linalg.norm(coords, axis=1)
        psi_val = self.psi(coords)

        d2psi = double_central_difference(
            self.psi, coords, h=[stepsize, stepsize, stepsize], order=8)

        laplacian = np.sum(d2psi, axis=1)

        E = -0.5 * (laplacian / psi_val) - 1.0 / (r + 1e-30)

        return E


def ground_state():
    #  gradient function in terms of theta
    def hydrogen_energy_grad(theta, samples):
        wf = hydrogen_wavefunction(theta)
        return hlp.H_partial_theta(wf, samples, analytic=False)

    theta0 = 0.8
    theta_min, E_min, E_err = min.gradient_descent(
        hydrogen_wavefunction, hydrogen_energy_grad, x_0=theta0, alpha=1., trace=True, N_s=100000, max_iter=100, stop_tol=1e-4, return_error=True, sampling="MH")

    print(
        f"minimum value of theta: {theta_min} \nminimum energy: {E_min} ± {E_err}")

    return theta_min, E_min, E_err


def plot_ground_state():
    import matplotlib.pyplot as plt
    ground_state_wavefunction = hydrogen_wavefunction(1.)
    samples = metropolis_hastings(f=ground_state_wavefunction.probability_density, f_prop='gaussian', x_0=[1., 1., 1.], xmin=[-10., -10., -10.], xmax=[10., 10., 10.], N=10000000, kwrgs={
        'sigma': 1.8}, detail=True)
    samples = samples[len(samples)//10:]
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    plt.figure(figsize=(8, 6))
    plt.hist2d(x, y, bins=200, cmap='inferno', density=True)
    plt.colorbar(label='Number Density')
    plt.title(f'2D Histogram of Electron Position (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig('hydrogen_samples.png')


if __name__ == "__main__":
    plot_ground_state()
