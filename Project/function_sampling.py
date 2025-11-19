"""
A module to sample functions from a given function space. 
Louis Liu 19/11
"""
import numpy as np


def metropolis_hastings(f: function, f_prop: function, x_0: np.ndarray, N: int = 10000, kwrgs: dict = None):
    """
    Metropolis algorithm to sample from a target distribution defined by nd function f.

    Parameters:
        f : callable, The target function to sample from.
        f_prop : callable, The proposal function to generate new samples.
        x_0 : list or np.ndarray, The initial position to start sampling from.
        N : int, The number of samples to generate.
        kwrgs : dict, Additional keyword arguments to pass to the proposal function.

    Returns: np.ndarray, An array of sampled positions.
    """
    samples = []
    current = x_0
    samples.append(current)

    accepted_count = 0

    for _ in range(N - 1):
        proposal = [f_prop(i, kwrgs) for i in current]

        prob_current = f(current)
        prob_proposal = f(proposal)

        # greater than 1 means a higher probability
        acceptance_ratio = prob_proposal / prob_current

        if np.random.uniform(0, 1) < min(1, acceptance_ratio):
            current = proposal
            accepted_count += 1
        samples.append(current)

    # print(f"Acceptance Rate: {accepted_count / N:.2f}")
    return np.array(samples)
