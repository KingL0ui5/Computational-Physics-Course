"""
Simulation of the hydrogen atom using variational Monte Carlo methods.
Louis Liu 22/11
"""

import numpy as np
from modules.function_sampling import metropolis_hastings
from modules.differentiators import double_central_difference
from modules.helpers import hydrogen_atom_helpers as hlp
from modules.minimisers import hydrogen_atom_minimisers as min
e = 1e-15


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

        E = -0.5 * (laplacian / psi_val) - 1.0 / (r + e)

        return E


def ground_state():
    #  gradient function in terms of theta
    def hydrogen_energy_grad(theta, samples):
        wf = hydrogen_wavefunction(theta)
        return hlp.H_partial_theta(wf, samples, analytic=False)

    theta0 = 0.8
    theta_min, E_min, E_err = min.gradient_descent(
        hydrogen_wavefunction, hydrogen_energy_grad, x_0=theta0, alpha=0.5, trace=True, N_s=100000, max_iter=50, stop_tol=1e-3, return_error=True, sampling="MH")

    print(
        f"minimum value of theta: {theta_min} \nminimum energy: {E_min} \nerror estimate: {E_err}")


if __name__ == "__main__":
    # N_s = 100000
    # theta = 1

    # psi = hydrogen_wavefunction(theta=theta)
    # samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[1., 1., 1.], xmin=[-10., -10., -10.], xmax=[10., 10., 10.], N=N_s, kwrgs={
    #     'sigma': 2.}, detail=True)

    # #  discard burn in
    # samples = samples[N_s//10:]

    # #  compute the local energy across all the samples
    # localenergy_arr = hydrogen.analytic_local_energy(psi, samples)

    # #  find the expected energy of the state, given the local energies
    # exp_energy = np.mean(localenergy_arr)
    # print(f"Expected Energy: {exp_energy}")

    # #  find the gradient at this point
    # dH = test_derivative(psi, samples)
    # print(f"Derivative dH/d_theta: {dH}")

    ground_state()
