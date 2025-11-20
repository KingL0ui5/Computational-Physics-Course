"""
Defining the 1D harmonic oscillator eigenfunctions for the dimensionless hamiltonian (6)
Louis Liu 19/11
"""

import numpy as np
from function_sampling import metropolis_hastings


def eigenfunctions(n: int) -> tuple:
    """
    Compute the nth eigenfunction of the 1D harmonic oscillator at position x.
    Parameters:
        x : float | np.ndarray, The position(s) at which to evaluate the eigenfunction.
        n : int, The quantum number of the eigenfunction.
    Returns:
        tuple: The value(s) of the nth eigenfunction at x, and the dimensionless energy eigenvalue.
    """

    def f(x):
        x = np.asarray(x)
        H = 0
        if n == 0:
            H = 1.0

        if n == 1:
            H = 2 * x

        h_prev = 1.0  # H_0
        h_curr = 2 * x  # H_1

        for i in range(1, n):
            h_next = (2 * x * h_curr) - (2 * i * h_prev)
            h_prev = h_curr
            h_curr = h_next

        H = h_curr
        return np.exp(-x**2 / 2) * H

    return f, n + 0.5


def test():
    x = np.linspace(0, 10, 100)
    N = 100000
    f, _ = eigenfunctions(4)
    samples = metropolis_hastings(lambda x: f(x)**2, 'gaussian', [0.], xmin=[0.], xmax=[10.], N=N, kwrgs={
                                  'sigma': 1}).flatten()

    #  discard first 10% of samples as burn-in
    samples = samples[int(0.1 * N):]

    import matplotlib.pyplot as plt
    plt.hist(samples, bins=500, density=True,
             alpha=0.6, label='Sampled Distribution')

    pdf = f(x)**2
    pdf /= np.trapz(pdf, x)
    plt.plot(x, pdf, label='Target Distribution', color='red')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
