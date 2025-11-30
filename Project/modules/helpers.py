"""
A helper module containing utiltiy functions
"""
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


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


class hydrogen_molecule:
    @staticmethod
    def anstatz(theta, q1, q2):
        """
        Creates the trial wavefunction for the Hydrogen molecule (H2).

        Parameters:
            theta (np.ndarray): Variational parameters [theta1, theta2, theta3].
            q1 (np.ndarray): Position of the first nucleus.
            q2 (np.ndarray): Position of the second nucleus.

        Returns:
            callable: A function f(r1, r2) that evaluates the wavefunction, where
                    r1 and r2 are arrays of electron coordinates.
        """

        def d(a, b): return np.sqrt(np.sum((a - b)**2, axis=1))
        theta1, theta2, theta3 = theta

        def f(r1, r2):
            r1 = np.asarray(r1)
            r2 = np.asarray(r2)
            return (np.exp(-theta1 * (d(r1, q1) + d(r2, q2))) +
                    np.exp(-theta1 * (d(r1, q2) + d(r2, q1)))) * \
                np.exp(-theta2 / (1.0 + theta3 * d(r1, r2)))
        return f

    @staticmethod
    def plot_samples(r1: np.ndarray, r2: np.ndarray, xlim: np.ndarray | tuple, ylim: np.ndarray | tuple) -> None:
        """
        Visualizes the electron probability density as a 2D heatmap.

        Parameters:
            r1 : numpy.ndarray, Array of shape (N, D) containing samples for the first electron.
            r2 : numpy.ndarray, Array of shape (N, D) containing samples for the second electron.
            xlim : tuple or list, A sequence of length 2 defining the x-axis plotting limits (min, max).
            ylim : tuple or list, A sequence of length 2 defining the y-axis plotting limits (min, max).

        Returns:
            None, Displays the figure using plt.show() and returns None.
        """

        x_coords = np.concatenate([r1[:, 0], r2[:, 0]])
        y_coords = np.concatenate([r1[:, 1], r2[:, 1]])

        fig = plt.figure(figsize=(8, 6))
        sns.kdeplot(x=x_coords, y=y_coords, fill=True,
                    cmap="mako", thresh=0, levels=50, cbar=True)

        plt.title('Electron Density Heatmap (x, y)')
        plt.xlabel('x position')
        plt.ylabel('y position')

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        plt.scatter([1, 0], [0, 1], color='red',
                    marker='x', label='Nuclei')
        plt.legend()

        return fig


class hydrogen:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def H_partial_theta(wf, samples: np.ndarray, analytic: bool = False) -> float:
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
    @staticmethod
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
