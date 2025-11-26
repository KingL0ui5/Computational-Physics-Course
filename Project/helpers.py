"""
A helper module containing utiltiy functions
"""
from typing import Callable
import numpy as np


def rms(data):
    """
    Compute the root mean square of an array.
    Parameters:
        data: array-like, The input data
    Returns:
        float: The root mean square of the data
    """
    import numpy as np
    data = np.asarray(data)
    return np.sqrt(np.mean(data**2))


def get_acf(series, max_lag=200):
    series = series.flatten()
    series = series - np.mean(series)
    corr = np.correlate(series, series, mode='full')
    corr = corr[len(corr)//2:]
    corr = corr / corr[0]
    return corr[:max_lag]


class hydrogen:
    def anstatz(theta: float) -> Callable:
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

    def H_partial_theta(wf, local_E: Callable, H_exp: float, coords: np.ndarray, N_s: int):
        """
        Returns the analytic derivative of the expectation value of the hamiltonian in the Hydrogen atom system for minimiser functions
        Parameters:
            wf: hydrogen wavefunction object, the wavefunction to differentiate
            local_E: float, the function to find the local energy of the wavefunction
            H_exp: float, the expected value of the hamiltonian at the parameter theta
            coords: np.ndarray, array of coordinates to evaluate the derivative
            N_s: int, the number of samples used to find the expectation value of the hamiltonian
        """
        psi = wf.psi(coords)
        dtheta = -wf.theta() * psi
        sum = (local_E(coords) - H_exp) * (dtheta / psi)
        return 2/N_s * np.sum(sum)


class harmonic_oscillator:
    def eigenfunctions(n: int) -> tuple:
        """
        Compute the nth eigenfunction of the 1D harmonic oscillator at position x.
        Parameters:
            n : int, The quantum number of the eigenfunction.
        Returns:
            tuple: The callable eigenfunction, and the dimensionless energy eigenvalue.
        """

        def f(x):
            x = np.asarray(x)
            H = 0
            if n == 0:
                H = 1.0

            elif n == 1:
                H = 2 * x

            else:
                h_prev = 1.0  # H_0
                h_curr = 2 * x  # H_1

                for i in range(1, n):
                    h_next = (2 * x * h_curr) - (2 * i * h_prev)
                    h_prev = h_curr
                    h_curr = h_next

                H = h_curr
            return np.array(np.exp(-x**2 / 2) * H)

        return f, n + 0.5

    def second_derivative(self, n: int):
        """
        Compute the analytical second derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
        Parameters:
            n: int, The quantum number of the eigenfunction
        Returns:
            Callable: The second derivative function
        """
        eigenfunction, _ = self.harmonic_eigenfunctions(n)

        def f(x):
            x = np.asarray(x)
            return (x**2 - (2*n + 1)) * eigenfunction(x)
        return f

    def first_derivative(self, n: int):
        """
        Compute the analytical first derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
        Parameters:
            n: int, The quantum number of the eigenfunction
        Returns:
            Callable: The first derivative function
        """
        psi_n, _ = self.harmonic_eigenfunctions(n)

        if n == 0:
            def f_ground(x):
                x = np.asarray(x)
                return -x * psi_n(x)
            return f_ground

        psi_n_minus_1, _ = self.harmonic_eigenfunctions(n - 1)
        sqrt_2n = np.sqrt(2 * n)

        def f(x):
            x = np.asarray(x)
            return sqrt_2n * psi_n_minus_1(x) - x * psi_n(x)
        return f
