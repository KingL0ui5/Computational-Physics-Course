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
            coords = np.array(coords)
            if len(coords.shape) == 1:
                coords = np.array([coords])

            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            return np.exp(-theta * r)
        return f

    def analytic_local_energy(wf, coords: np.ndarray) -> float:
        """
        Returns the analytically determined local energy
        Parameters:
            wf: hydrogen_wavefunction object
            coords: np.ndarray, coordinates for the local energy
        Returns:
            float: the local energy at the input coordinates
        """
        coords = np.asarray(coords)

        if len(coords.shape) == 1:
            coords = np.array([coords])

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12

        return -0.5 * (wf.theta()**2) + (wf.theta() - 1.0) / r

    def H_partial_theta(wf, samples: np.ndarray, analytic: bool = False):
        """
        Returns the analytic derivative of the expectation value of the hamiltonian in the Hydrogen atom system for minimiser functions
        Parameters:
            wf: hydrogen wavefunction object, the wavefunction to differentiate
            coords: np.ndarray, array of coordinates to evaluate the derivative
            analytic: bool, whether to find the local energy analytically or not.
        Returns: 
            float, the derivative of the energy expecation value with respect to theta
        """
        samples = np.asarray(samples)

        if analytic:
            E_l = hydrogen.analytic_local_energy(wf, samples)
        else:
            E_l = wf.local_energy(samples)

        if len(samples.shape) == 1:
            samples = np.array([samples])

        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        sum = (E_l - np.mean(E_l)) * (-r)
        return 2 * np.mean(sum)


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

    @staticmethod
    def second_derivative(n: int):
        """
        Compute the analytical second derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
        Parameters:
            n: int, The quantum number of the eigenfunction
        Returns:
            Callable: The second derivative function
        """
        eigenfunction, _ = harmonic_oscillator.eigenfunctions(n)

        def f(x):
            x = np.asarray(x)
            return (x**2 - (2*n + 1)) * eigenfunction(x)
        return f

    @staticmethod
    def first_derivative(n: int):
        """
        Compute the analytical first derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
        Parameters:
            n: int, The quantum number of the eigenfunction
        Returns:
            Callable: The first derivative function
        """
        psi_n, _ = harmonic_oscillator.eigenfunctions(n)

        if n == 0:
            def f_ground(x):
                x = np.asarray(x)
                return -x * psi_n(x)
            return f_ground

        psi_n_minus_1, _ = harmonic_oscillator.eigenfunctions(n - 1)
        sqrt_2n = np.sqrt(2 * n)

        def f(x):
            x = np.asarray(x)
            return sqrt_2n * psi_n_minus_1(x) - x * psi_n(x)
        return f
