"""
Simulation of the hydrogen molecule using variational Monte Carlo methods.
Louis Liu 28/11
"""
import numpy as np
from modules.helpers import hydrogen_molecule
from modules.differentiators import double_central_difference, central_difference
from modules.function_sampling import metropolis_hastings
from modules.minimisers import quasi_newton
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

        self.f = hydrogen_molecule.anstatz(thetas, q1, q2)
        self._thetass = thetas
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

        r1q1, r1q2 = d(r1, q1) + eps, d(r1, q2) + eps
        r2q1, r2q2 = d(r2, q1) + eps, d(r2, q2) + eps
        r12 = d(r1, r2) + eps
        q1q2 = d(q1, q2) + eps

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
        return np.mean(E_ls)


def dE(q1, q2):
    thetas_0 = [1., 1., 1.]

    def wrapper_1(theta_1, r):
        thetas = [theta_1, thetas_0[1], thetas_0[2]]
        wf = h2_wavefunction(thetas, q1, q2)
        return wf.E_exp(**r)

    def wrapper_2(theta_2, r):
        thetas = [thetas_0[0], theta_2, thetas_0[2]]
        wf = h2_wavefunction(thetas, q1, q2)
        return wf.E_exp(**r)

    def wrapper_3(theta_3, r):
        thetas = [thetas_0[0], thetas_0[1], theta_3]
        wf = h2_wavefunction(thetas, q1, q2)
        return wf.E_exp(**r)

    thetas = np.linspace(-10., 10., 200)
    stepsize = 1.08e-2

    # dE/dtheta

    def minimisation_fn(theta_1, theta_2, theta_3, r):
        return np.array([wrapper_1(theta_1, r), wrapper_2(theta_2, r), wrapper_3(theta_3, r)])

    def df(thetas):
        dE_t1 = central_difference(wrapper_1, x=thetas[0], h=stepsize, order=8)
        dE_t2 = central_difference(wrapper_2, x=thetas[1], h=stepsize, order=8)
        dE_t3 = central_difference(wrapper_3, x=thetas[2], h=stepsize, order=8)

        return np.array([dE_t1, dE_t2, dE_t3])

    theta_min, E_min = quasi_newton(f=minimisation_fn, df=df, x_0=thetas_0,
                                    stepsize=0.5, stop_tol=1e-3, detail=True)

    return theta_min, E_min


if __name__ == '__main__':
    wf = h2_wavefunction(theta=[1., 1., 1.], q1=[1, 0, 0], q2=[0, 1, 0])

    def wrapper(coords):
        coords = np.asarray(coords)
        r1 = coords[:, 0:3]
        r2 = coords[:, 3:6]

        return wf.probability_density(r1, r2)

    x_0 = np.zeros(6)
    N_s = 1000000
    samples = metropolis_hastings(
        wrapper, f_prop='gaussian', x_0=x_0, xmax=10., xmin=-10., N=N_s, kwrgs={
            'sigma': 2.}, detail=True)

    r1, r2 = samples[:, 0:3], samples[:, 3:6]
    E_l = wf.local_energy(r1=r1, r2=r2)
    print(f"local energy: {E_l}")
    exp_energy = np.mean(E_l)
    print(f"Expected Energy: {exp_energy}")

    import matplotlib.pyplot as plt
    hydrogen_molecule.plot_samples(r1, r2, xlim=[-3, 3], ylim=[-3, 3])
    plt.show()
