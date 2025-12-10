"""
A module to sample functions from a given function space.
Louis Liu 19/11
"""
import numpy as np
from typing import Callable
import time


class ProposalFunction:
    @staticmethod
    def gaussian(x: np.ndarray | float, kwrgs: dict) -> float:
        """
        Gaussian proposal function to generate new samples.
        Parameters:
            x : float, The current position.
            kwrgs : dict, Additional keyword arguments containing 'sigma'.
        Returns:
            float: The proposed new position.
        """
        sigma = kwrgs['sigma']
        return np.random.normal(loc=x, scale=sigma)


def metropolis_hastings(f: Callable, f_prop: str, x_0: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, N: int = 10000, kwrgs: dict = None, thinning: int = 1, detail: bool = False, return_acceptance: bool = False) -> np.ndarray:
    """
    Metropolis algorithm to sample from a target distribution defined by nd function f.

    Parameters:
        f : callable, The target function to sample from.
        f_prop : str, the proposal function for the distribution.
        xmin : list or np.ndarray, The minimum bounds for each dimension.
        xmax : list or np.ndarray, The maximum bounds for each dimension.
        x_0 : list or np.ndarray, The initial position to start sampling from.
        N : int, The number of samples to generate.
        thinning : int, The thinning factor to reduce autocorrelation in samples.
        kwrgs : dict, Additional keyword arguments to pass to the proposal function.
        detail : bool
        return_acceptance : bool, Whether to return the acceptance rate along with samples.

    Returns: np.ndarray, An array of sampled positions in coordinate format.
    """
    if f_prop == 'gaussian':
        f_prop = ProposalFunction.gaussian
    else:
        raise ValueError(f"Unknown proposal function '{f_prop}'")

    current = np.array(x_0, dtype=float)
    n_dims = current.shape[0]
    xmin = np.array(xmin, dtype=float)
    xmax = np.array(xmax, dtype=float)
    samples = np.zeros((N // thinning, n_dims))
    samples[0] = current

    accepted_count = 0

    start1 = time.perf_counter()
    for i in range(N):
        proposal = np.array([f_prop(xi, kwrgs) for xi in current])

        if np.any(proposal < xmin) or np.any(proposal > xmax):
            if i % thinning == 0:
                samples[i // thinning] = (current)
            continue

        prob_current = f([current])
        prob_proposal = f([proposal])

        # greater than 1 means a higher probability
        eps = 1e-30
        log_acceptance_ratio = np.log(
            prob_proposal + eps) - np.log(prob_current + eps)

        if np.log(np.random.uniform(0, 1)) < min(1, log_acceptance_ratio):
            current = proposal
            accepted_count += 1

        if i % thinning == 0:
            samples[i // thinning] = (current)
    end1 = time.perf_counter()
    acceptance_rate = accepted_count / N
    if detail:
        import matplotlib.pyplot as plt
        n_dims = samples.shape[1]
        plot_dims = min(n_dims, 5)

        fig, axes = plt.subplots(plot_dims, 1, figsize=(
            10, 2 * plot_dims), sharex=True)
        if plot_dims == 1:
            axes = [axes]

        for d in range(plot_dims):
            axes[d].plot(samples[:, d], color='black',
                         linewidth=0.5, alpha=0.6)
            axes[d].set_ylabel(f'Dim {d}')

        axes[-1].set_xlabel('Iteration')
        plt.suptitle(
            f'Trace \nacceptance: {acceptance_rate:.2f}, time: {end1-start1:.4f}s')
        plt.tight_layout()
        plt.show()
        elapsed = end1 - start1
        print(f"Time elapsed: {elapsed} for number of samples: {N/thinning}")

    if return_acceptance:
        return np.asarray(samples), acceptance_rate

    return np.asarray(samples)


def MALA(f: Callable, x_0: np.ndarray | list, xmin: np.ndarray | list, xmax: np.ndarray | list, timestep: float, N: int = 10000, order: int = 8, stepsize: float = 8.008e-03, detail: bool = False, return_acceptance: bool = False) -> np.ndarray:
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) to sample from a target distribution defined by nd wavefunction probability density function f.
    Calculates the gradient using central difference method.
    Parameters:
        f : callable, the wavefunction to sample from.
        xmin : list or np.ndarray, The minimum bounds for each dimension.
        xmax : list or np.ndarray, The maximum bounds for each dimension.
        x_0 : list or np.ndarray, The initial position to start sampling from.
        timestep : float, The timestep for the Langevin dynamics.
        order : int, The order of the central difference to use for gradient estimation.
        stepsize : float, The stepsize for the central difference calculation.
        N : int, The number of samples to generate.
        return_acceptance : bool, Whether to return the acceptance rate along with samples.

    Returns: np.ndarray, An array of sampled positions.
    """

    def f_prime(x):
        from modules.differentiators import central_difference
        n_dims = x.size
        return central_difference(f, x, h=[stepsize]*n_dims, order=order)

    def pdf(x): return np.abs(f(x))**2
    current = np.array(x_0, dtype=float)
    n_dims = current.shape[0]

    xmin = np.array(xmin, dtype=float)
    xmax = np.array(xmax, dtype=float)
    samples = np.ones((N, n_dims))
    samples[0] = current.copy()

    accepted_count = 0
    current = current.reshape(1, -1)
    F_current = 2 * np.real(f_prime(current) / f(current))

    start2 = time.perf_counter()
    for i in range(N):
        xi = np.random.normal(size=current.shape)
        forward_mean = current + timestep * F_current

        proposal = forward_mean + \
            np.sqrt(2 * timestep) * xi

        if np.any(proposal < xmin) or np.any(proposal > xmax):
            samples[i] = current.copy()
            continue

        proposal = proposal.reshape(1, -1)
        F_prop = 2 * np.real(f_prime(proposal) / f(proposal))
        backward_mean = proposal + timestep * F_prop

        Rev_distance = np.sum((current - backward_mean)**2)
        Frd_distance = np.sum((proposal - forward_mean)**2)

        log_A = (np.log(pdf(proposal)) - np.log(pdf(current))) + \
            (Frd_distance - Rev_distance) / (4 * timestep)

        if np.log(np.random.uniform(0, 1)) < log_A:
            current = proposal
            F_current = F_prop
            accepted_count += 1

        samples[i] = current.copy()
    end2 = time.perf_counter()
    acceptance_rate = accepted_count / N

    if detail:
        import matplotlib.pyplot as plt
        samples = np.asarray(samples)
        n_dims = samples.shape[1]
        plot_dims = min(n_dims, 5)

        fig, axes = plt.subplots(plot_dims, 1, figsize=(
            10, 2 * plot_dims), sharex=True)
        if plot_dims == 1:
            axes = [axes]

        for d in range(plot_dims):
            axes[d].plot(samples[:, d], color='black',
                         linewidth=0.5, alpha=0.6)
            axes[d].set_ylabel(f'Dim {d}')

        axes[-1].set_xlabel('Iteration')
        plt.suptitle(
            f'Trace \nacceptance: {accepted_count/N:.2f}, time: {end2-start2:.4f}s')
        plt.tight_layout()
        plt.show()
        print(f"MALA acceptance Rate: {acceptance_rate:.2f}")
        print(f"time elapsed: {end2 - start2} to iteration {i}")

    if return_acceptance:
        return np.asarray(samples), acceptance_rate

    return np.asarray(samples)


def stochasticMALA(f: Callable, x_0: np.ndarray | list, xmin: np.ndarray | list, xmax: np.ndarray | list, timestep: float, N: int = 10000, order: int = 8, stepsize: float = 8.008e-03, p_kick: float = 0.1, kick_sigma: float = 1., detail: bool = False, return_acceptance: bool = False) -> np.ndarray:
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) to sample from a target distribution defined by nd wavefunction probability density function f.
    Also including stochastic random walk steps to improve exploration.

    Parameters:
        f : callable, the wavefunction probability density to sample from.
        xmin : list or np.ndarray, The minimum bounds for each dimension.
        xmax : list or np.ndarray, The maximum bounds for each dimension.
        x_0 : list or np.ndarray, The initial position to start sampling from.
        order : int, The order of the central difference method for numerical differentiation.
        stepsize : float, The stepsize for the central difference calculation.
        p_kick : float, The probability of performing a random walk step.
        kick_sigma : float, The standard deviation of the Gaussian used for random walk steps.
        timestep : float, The timestep for the Langevin dynamics.
        N : int, The number of samples to generate.
        return_acceptance : bool, Whether to return the acceptance rate along with samples.

    Returns: np.ndarray, An array of sampled positions.
    """

    def f_prime(x):
        from modules.differentiators import central_difference
        n_dims = x.size
        return central_difference(f, x, h=[stepsize]*n_dims, order=order)

    def pdf(x): return np.abs(f(x))**2

    current = np.array(x_0, dtype=float)
    xmin = np.array(xmin, dtype=float)
    xmax = np.array(xmax, dtype=float)
    samples = []
    samples.append(current.copy())

    accepted_count = 0
    F_current = 2 * np.real(f_prime(current) / f(current))

    start3 = time.perf_counter()
    for _ in range(N - 1):
        #  random kick
        if np.random.rand() < p_kick:
            step = np.random.normal(0, kick_sigma, size=current.shape)
            proposal = current + step

            if np.any(proposal < xmin) or np.any(proposal > xmax):
                samples.append(current.copy())
                continue

            # retain the log to minimmise numerical errors
            log_A = 2 * np.log(pdf(proposal) / pdf(current))

            if np.log(np.random.uniform(0, 1)) < log_A:
                current = proposal
                F_current = 2 * np.real(f_prime(current) / f(current))
                accepted_count += 1

        #  usual MALA
        else:
            xi = np.random.normal(size=current.shape)
            forward_mean = current + timestep * F_current

            proposal = forward_mean + \
                np.sqrt(2 * timestep) * xi

            if np.any(proposal < xmin) or np.any(proposal > xmax):
                samples.append(current.copy())
                continue

            F_prop = 2 * np.real(f_prime(proposal) / f(proposal))
            backward_mean = proposal + timestep * F_prop

            Rev_distance = np.sum((current - backward_mean)**2)
            Frd_distance = np.sum((proposal - forward_mean)**2)

            log_A = (np.log(pdf(proposal+1e-30)) - np.log(pdf(current))) + \
                (Frd_distance - Rev_distance) / (4 * timestep)

            if np.log(np.random.uniform(0, 1)) < log_A:
                current = proposal
                F_current = F_prop
                accepted_count += 1

        samples.append(current.copy())

    end3 = time.perf_counter()

    if detail:
        import matplotlib.pyplot as plt
        samples = np.asarray(samples)
        n_dims = samples.shape[1]
        plot_dims = min(n_dims, 5)

        fig, axes = plt.subplots(plot_dims, 1, figsize=(
            10, 2 * plot_dims), sharex=True)
        if plot_dims == 1:
            axes = [axes]

        for d in range(plot_dims):
            axes[d].plot(samples[:, d], color='black',
                         linewidth=0.5, alpha=0.6)
            axes[d].set_ylabel(f'Dim {d}')

        axes[-1].set_xlabel('Iteration')
        plt.suptitle(
            f'Trace \nacceptance: {accepted_count/N:.2f}, time: {end3-start3:.4f}s')
        plt.tight_layout()
        plt.show()
        print(f"MALA acceptance Rate: {accepted_count/N:.2f}")
        print(f"time elapsed: {end3 - start3} to iteration {N}")
    if return_acceptance:
        return np.asarray(samples), accepted_count / N

    # print(f"Acceptance Rate: {accepted_count / N:.2f}")
    return np.asarray(samples)
