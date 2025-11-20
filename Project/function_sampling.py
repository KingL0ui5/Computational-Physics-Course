"""
A module to sample functions from a given function space. 
Louis Liu 19/11
"""
import numpy as np
from typing import Callable


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
        sigma = kwrgs.get('sigma', 1.0)
        return np.random.normal(loc=x, scale=sigma)


def metropolis_hastings(f: Callable, f_prop: str, x_0: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, N: int = 10000, kwrgs: dict = None) -> np.ndarray:
    """
    Metropolis algorithm to sample from a target distribution defined by nd function f.

    Parameters:
        f : callable, The target function to sample from.
        f_prop : str, the proposal function for the distribution.
        xmin : list or np.ndarray, The minimum bounds for each dimension.
        xmax : list or np.ndarray, The maximum bounds for each dimension.
        x_0 : list or np.ndarray, The initial position to start sampling from.
        N : int, The number of samples to generate.
        kwrgs : dict, Additional keyword arguments to pass to the proposal function.

    Returns: np.ndarray, An array of sampled positions.
    """
    if f_prop == 'gaussian':
        f_prop = ProposalFunction.gaussian
    else:
        raise ValueError(f"Unknown proposal function '{f_prop}'")

    current = np.array(x_0, dtype=float)
    xmin = np.array(xmin, dtype=float)
    xmax = np.array(xmax, dtype=float)
    samples = []
    samples.append(current)

    accepted_count = 0

    for _ in range(N - 1):
        proposal = [f_prop(i, kwrgs) for i in current]

        prob_current = f(current)
        prob_proposal = f(proposal)

        # greater than 1 means a higher probability
        acceptance_ratio = prob_proposal / prob_current

        if np.any(proposal < xmin) or np.any(proposal > xmax):
            samples.append(current.copy())
            continue

        if np.random.uniform(0, 1) < min(1, acceptance_ratio):
            current = proposal
            accepted_count += 1
        samples.append(current)

    # print(f"Acceptance Rate: {accepted_count / N:.2f}")
    return np.vstack(samples)


def MALA():
    pass
