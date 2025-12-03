"""
Simulation of the hydrogen molecule using variational Monte Carlo methods.
Louis Liu 28/11
"""
import numpy as np
from modules.helpers import hydrogen_molecule_helpers as hlp
from modules.differentiators import double_central_difference
from modules.function_sampling import metropolis_hastings
from modules.minimisers import hydrogen_molecule_minimisers as min
import matplotlib.pyplot as plt
eps = 1e-15


class h2_wavefunction:
    def __init__(self, thetas: np.ndarray, q1: np.ndarray, q2: np.ndarray):
        """
        Constructor
        Parameters:
            theta (np.ndarray): Variational parameters [theta1, theta2, theta3].
            q1 (np.ndarray): Position of the first nucleus.
            q2 (np.ndarray): Position of the second nucleus.
        """
        q1, q2 = np.array(q1), np.array(q2)
        if len(q1.shape) == 1:
            q1, q2 = np.asarray([q1]), np.asarray([q2])

        self.f = hlp.anstatz(thetas, q1, q2)
        self._thetas = thetas
        self._q1 = q1
        self._q2 = q2

    def psi(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray | float:
        """
        The trial wavefunction
        Parameters:
            r1: np.ndarray, The position(s) of electron 1 as an array of shape (N, 3)
            r2: np.ndarray, The position(s) of electron 2 as an array of shape (N, 3)
        Returns:
            np.ndarray | float: The value of the trial wavefunction at the given position(s)
        """
        return self.f(r1, r2)

    def probability_density(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray | float:
        """
        The probability density of the trial wavefunction
        Parameters:
            r1: np.ndarray, The position(s) of electron 1 as an array of shape (N, 3)
            r2: np.ndarray, The position(s) of electron 2 as an array of shape (N, 3)
        Returns:
            np.ndarray | float: The probability density at the given position(s)
        """
        return np.abs(self.f(r1, r2))**2

    def theta(self) -> np.ndarray:
        """
        Returns the theta parameter of the wavefunction
        Returns:
            float: the value of theta for the current wavefunction
        """
        return self._thetas

    def local_energy(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        The local energy of the hydrogen molecule trial wavefunction
        Parameters:
            r1: np.ndarray, The position(s) of electron 1 as an array of shape (N, 3)
            r2: np.ndarray, The position(s) of electron 2 as an array of shape (N, 3)
        Returns:
            np.ndarray: The local energy at the given position(s)
        """

        # this method has the same issues as before. There is a probelm around r=0 due to numerical divergence when it should cancel
        r1, r2 = np.asarray(r1), np.asarray(r2)
        if len(r1.shape) == 1:
            r1, r2 = np.asarray([r1]), np.asarray([r2])

        q1 = self._q1
        q2 = self._q2

        #  function to fix r2
        def f_r1(r1):
            return self.psi(r1, r2)

        #  function to fix r1
        def f_r2(r2):
            return self.psi(r1, r2)

        d2f_r1 = double_central_difference(
            f_r1, r1, h=[1e-5, 1e-5, 1e-5], order=8)

        d2f_r2 = double_central_difference(
            f_r2, r2, h=[1e-5, 1e-5, 1e-5], order=8)

        kinetic = - 0.5 / \
            self.psi(r1, r2) * (np.sum(d2f_r1, axis=1) +
                                np.sum(d2f_r2, axis=1))

        def d(a, b): return np.sqrt(np.sum((a - b)**2, axis=1))

        r1q1, r1q2 = d(r1, q1), d(r1, q2)
        r2q1, r2q2 = d(r2, q1), d(r2, q2)
        r12 = d(r1, r2)
        q1q2 = d(q1, q2)

        potential = - (1/r1q1 + 1/r1q2 + 1/r2q1 + 1/r2q2) \
            + (1/r12) \
            + (1/q1q2)

        return kinetic + potential

    def E_exp(self, r1, r2):
        """
        The energy expecation value of the hydrogen molecule trial wavefunction at given coordinates
        Parameters:
            r1: np.ndarray, The position(s) of electron 1 as an array of shape (N, 3)
            r2: np.ndarray, The position(s) of electron 2 as an array of shape (N, 3)
        Returns:
            np.ndarray: The local energy at the given position(s)
        """
        r1, r2 = np.asarray(r1), np.asarray(r2)
        E_ls = self.local_energy(r1, r2)
        return np.mean(E_ls,)


def test_samples():
    q1 = [0, 1, 0]
    q2 = [0, -1, 0]
    wf = h2_wavefunction(thetas=[1., 1., 1.], q1=q1, q2=q2)

    def wrapper(coords):
        coords = np.asarray(coords)
        r1 = coords[:, 0:3]
        r2 = coords[:, 3:6]

        return wf.probability_density(r1, r2)

    x_0 = np.zeros(6)
    N_s = 1000000
    samples = metropolis_hastings(
        wrapper, f_prop='gaussian', x_0=x_0, xmax=10., xmin=-10., N=N_s, kwrgs={
            'sigma': 0.7}, detail=True)

    r1, r2 = samples[:, 0:3], samples[:, 3:6]
    E_l = wf.local_energy(r1=r1, r2=r2)

    plt.scatter(np.linalg.norm(r1, axis=1), E_l, marker='.', alpha=0.1)
    plt.plot(0, q1[1], 'ro', label='Nucleus 1')
    plt.plot(0, q2[1], 'bo', label='Nucleus 2')
    plt.xlabel("Distance of electron 1 from origin")
    plt.ylabel("Local Energy")
    plt.title("Local energy vs distance of electron 1 from origin")
    plt.show()

    print(f"local energy: {E_l}")
    exp_energy = np.mean(E_l)
    print(f"Expected Energy: {exp_energy}")

    hlp.plot_samples(
        r1, r2, q1=q1, q2=q2, xlim=[-3, 3], ylim=[-3, 3], seaborn=True)
    plt.show()


if __name__ == '__main__':
    q1 = [0, 0, 1]
    q2 = [0, 0, -1]
    wf = h2_wavefunction(thetas=[1., 1., 1.], q1=q1, q2=q2)

    thetas_0 = [1., 1., 1.]  #  for minimiser
    theta_min, E_min = min.gradient_descent(q1, q2, x_0=thetas_0,
                                            alpha=0.5, stop_tol=1e-3, N_s=100000, detail=True)
    print(f"minimum theta: {theta_min}, minimum energy: {E_min}")
