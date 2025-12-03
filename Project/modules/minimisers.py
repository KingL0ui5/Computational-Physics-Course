"""
A module containing the minimisers to find the ground state energy of wavefunctions.
Louis Liu 22/11
"""

from typing import Callable
import numpy as np
import time


def H2_gradient_descent(wf: Callable, dH: Callable, x_0: np.ndarray, stepsize: float, max_iter: int = 100, stop_tol: float = 1e-6, N_s: int = 10000, detail: bool = False) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method.
    Parameters:
        E: callable, the function to minimise
        dH: callable, the first derivative of the function, left for flexibility of method
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer
        max_iter: int, the number of iterations to run
        stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
        N_s: number of samples to take per wavefunction iteration
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    from modules.function_sampling import metropolis_hastings
    from hydrogen_molecule import h2_wavefunction
    q1 = 0
    q2 = 0

    def wavefunction_wrapper(coords):
        r1, r2 = coords[:, 0:3], coords[:, 3:6]
        return wf.probability_density(r1, r2)

    def minimisation_fn(thetas):
        wf = h2_wavefunction(thetas, q1, q2)
        r = metropolis_hastings(f=wavefunction_wrapper, f_prop='gaussian', x_0=r_0, xmin=[
                                0.001, 0.001, 0.001, 0.001, 0.001, 0.001], xmax=[10., 10., 10., 10., 10., 10.], N=Ns, kwrgs={'sigma': 0.8})

        return np.asarray(r)
    x = x_0
    d = 1
    start = time.perf_counter()

    for i in range(max_iter):
        psi = wf(theta=x)  # hydrogen wavefunction given theta
        Ns_i = N_s * int(np.exp((i+1)*0.05))

        samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[
                                      1., 1., 1.], xmin=[0.001, 0.001, 0.001], xmax=[10., 10., 10.], N=Ns_i, kwrgs={'sigma': 0.8})
        samples = samples[Ns_i//10:]

        d = dH(x, samples)

        #  test stopping condition
        if stop_tol:
            if np.linalg.norm(d) < stop_tol:
                break

        x = x - stepsize * d
        #  restrict x to be positive
        if x <= 1e-7:
            x = 1e-7

        if detail:
            print(
                f"iteration {i}, x={x}, d/dx={d}, N_s={Ns_i}, alpha={stepsize}")

    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number GD: {i}")
        print(f"time elapsed GD: {time_elapsed}")
        print(f"final derivative: {d}")
    return x, np.nanmean(psi.local_energy(coords=samples))


def hydrogen_adapted_quasi_newton(wf: Callable, dH: Callable, x_0: np.ndarray, stepsize: float, method: str = "DFP", max_iter: int = 1000, N_s: int = 1000, stop_tol: float = None, detail: bool = False) -> tuple:
    """
    A method to find the minima of a function using the quasi-newton method using a chosen method to approximate the hessian
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimiser
        method: str, default = "DFP" the method to use for the hessian approximation - allowed values "DFP", "BFGS"
        stop_tol: float, default = None, the value of the gradient at which convergence is determined. defaults to none to give a chance to get off local minima
        max_iter: int, default = 10000 the number of iterations to run
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    from modules.function_sampling import metropolis_hastings
    x = np.array(x_0, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    n_dim = len(x)
    G = np.eye(n_dim)

    def DFP(x_new, grad_new, x, grad, G):
        delta = x_new - x
        gamma = grad_new - grad

        if np.dot(gamma, delta) == 0:
            return G

        G = G + np.outer(delta, delta) / np.dot(gamma, delta) - \
            (G @ np.outer(gamma, gamma) @ G) / np.dot(gamma, G @ gamma)
        return G

    def BFGS(x_new, grad_new, x, grad, G):
        delta = x_new - x
        gamma = grad_new - grad
        dim = len(delta)

        temp = 1.0 / np.dot(gamma, delta)
        I = np.eye(dim)
        G = (I - temp * np.outer(delta, gamma)) @ G @ (I - temp * np.outer(gamma, delta)) \
            + temp * np.outer(delta, delta)

        return G

    if method == "DFP":
        grad_update = DFP

    elif method == "BFGS":
        grad_update = BFGS

    start = time.perf_counter()
    for i in range(max_iter):
        psi = wf(theta=x[0])  # hydrogen wavefunction given theta
        Ns_i = N_s * int(np.exp((i+1)*0.1))

        samples = samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[
            1., 1., 1.], xmin=[-20., -20., -20.], xmax=[20., 20., 20.], N=Ns_i, kwrgs={'sigma': 0.8})
        samples = samples[Ns_i//10:]

        grad = np.asarray(dH(x, samples), dtype=float)

        if grad.ndim == 0:
            grad = grad.reshape(1)

        if stop_tol:
            if np.linalg.norm(grad) < stop_tol:
                print("halting QN")
                break

        x_new = x - stepsize * G @ grad
        psi_new = wf(theta=x_new[0])  # hydrogen wavefunction given theta
        samples_new = metropolis_hastings(f=psi_new.probability_density, f_prop='gaussian', x_0=[
            1., 1., 1.], xmin=[-20., -20., -20.], xmax=[20., 20., 20.], N=Ns_i, kwrgs={'sigma': 0.8})
        samples_new = samples_new[N_s//10:]

        grad_new = np.asarray(dH(x_new, samples_new), dtype=float)
        if grad_new.ndim == 0:
            grad_new = grad_new.reshape(1)

        G = grad_update(x_new, grad_new, x, grad, G)
        x = x_new

        if detail:
            print(f"iteration {i}, x={x}, G={G}, gradient={grad}, Ns = {Ns_i}")

    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number QN: {i}")
        print(f"time elapsed QN: {time_elapsed}")
    return x, np.nanmean(psi.local_energy(coords=samples))


def hydrogen_adapted_gradient_desecent(wf: Callable, dH: Callable, x_0: np.ndarray, stepsize: float, max_iter: int = 100, stop_tol: float = 1e-6, N_s: int = 10000, detail: bool = False) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method.
    Parameters:
        E: callable, the function to minimise
        dH: callable, the first derivative of the function, left for flexibility of method
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer
        max_iter: int, the number of iterations to run
        stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
        N_s: number of samples to take per wavefunction iteration
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    from modules.function_sampling import metropolis_hastings
    x = x_0
    d = 1
    start = time.perf_counter()

    for i in range(max_iter):
        psi = wf(theta=x)  # hydrogen wavefunction given theta
        Ns_i = N_s * int(np.exp((i+1)*0.05))

        samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[
                                      1., 1., 1.], xmin=[0.001, 0.001, 0.001], xmax=[10., 10., 10.], N=Ns_i, kwrgs={'sigma': 0.8})
        samples = samples[Ns_i//10:]

        d = dH(x, samples)

        #  test stopping condition
        if stop_tol:
            if np.linalg.norm(d) < stop_tol:
                break

        x = x - stepsize * d
        #  restrict x to be positive
        if x <= 1e-7:
            x = 1e-7

        if detail:
            print(
                f"iteration {i}, x={x}, d/dx={d}, N_s={Ns_i}, alpha={stepsize}")

    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number GD: {i}")
        print(f"time elapsed GD: {time_elapsed}")
        print(f"final derivative: {d}")
    return x, np.nanmean(psi.local_energy(coords=samples))


def hydrogen_adapted_stochastic_GD(wf: Callable, dH: Callable, x_0: np.ndarray, stepsize: float, stop_tol: float = None, max_iter: int = 1000, noise: float = 0.1, N_s: int = 1000, detail: bool = False) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method with added noise. 
    Parameters:
        E: callable, the function to minimise
        dH: callable, the first derivative of the function, left for flexibility of method
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer
        max_iter: int, the number of iterations to run
        noise: float, the level of noise to introduce to the derivative
        stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
        N_s: number of samples to take per wavefunction iteration
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    from modules.function_sampling import metropolis_hastings

    x = np.asarray(x_0, dtype=float)
    start = time.perf_counter()

    for i in range(max_iter):
        psi = wf(theta=x)  # hydrogen wavefunction given theta
        samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[
                                      1., 1., 1.], xmin=[-20., -20., -20.], xmax=[20., 20., 20.], N=N_s, kwrgs={'sigma': 0.8})
        samples = samples[N_s//10:]

        d = dH(x, samples)

        #  stopping conditions
        if stop_tol:
            if np.linalg.norm(d) < stop_tol:
                break

        sigma = noise * stepsize
        d += np.random.normal(loc=0, scale=sigma, size=x.shape)
        x = x - stepsize * d

        if detail:
            print(f"iteration {i}, x={x}")

    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number SGD: {i}")
        print(f"time elapsed SGD: {time_elapsed}")
        print(f"final derivative: {d}")

    return x, np.nanmean(psi.local_energy(coords=samples))


def gradient_desecent(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, max_iter: int = 1000, stop_tol: float = 1e-6, detail: bool = False, **kwargs) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        max_iter: int, the number of iterations to run
        stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    start = time.perf_counter()
    for i in range(max_iter):
        d = df(x, **kwargs)

        if stop_tol:
            if np.linalg.norm(d) < stop_tol:
                break
        x = x - stepsize * d

        if detail:
            print(f'iteration number: {i}, x: {x}, dx: {d}')

    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number GD: {i}")
        print(f"time elapsed GD: {time_elapsed}")
    return x, f(x, **kwargs)


def stochastic_GD(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, noise: float, max_iter: int = 1000, stop_tol: float = None, detail: bool = False, **kwargs) -> tuple:
    """
    A method to find the minima of a function using the gradient descent method with added noise. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimizer 
        noise: float, the level of noise to introduce to the derivative
        max_iter: int, the number of iterations to run
        stop_tol: float, default = None, the value of the gradient at which convergence is determined. Defaulted to none to give it a chance to get off local minima
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = np.asarray(x_0, dtype=float)
    start = time.perf_counter()
    for i in range(max_iter):
        d = df(x, **kwargs)
        if stop_tol:
            if np.linalg.norm(d) < stop_tol:
                break
        sigma = noise * stepsize
        d += np.random.normal(loc=0, scale=sigma, size=x.shape)
        x = x - stepsize * d
    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number SGD: {i}")
        print(f"time elapsed SGD: {time_elapsed}")
    return x, f(x, **kwargs)


def RMSProp_GD(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, forgetting: float, max_iter: int = 1000, stop_tol: float = 1e-6, detail: bool = False, **kwargs) -> tuple:
    """
    A method to find the minima of a function using the RMSProp adjusted gradient descent method. 
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimiser 
        forgetting: float, the forgetting factor of the minimiser
        max_iter: int, the number of iterations to run
        stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = x_0
    v = np.zeros_like(x)
    start = time.perf_counter()
    for i in range(max_iter):
        d = df(x, **kwargs)

        if np.linalg.norm(d) < stop_tol:
            break

        v = forgetting * v + (1 - forgetting)*(d**2)
        x = x - (stepsize / np.sqrt(v)) * d
    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number RMS GD: {i}")
        print(f"time elapsed RMS GD: {time_elapsed}")
    return x, f(x, **kwargs)


def quasi_newton(f: Callable, df: Callable, x_0: np.ndarray, stepsize: float, method: str = "DFP", max_iter: int = 1000, stop_tol: float = None, detail: bool = False) -> tuple:
    """
    A method to find the minima of a function using the quasi-newton method using a chosen method to approximate the hessian
    Parameters:
        f: callable, the function to minimise 
        df: callable, the first derivative of the function, left for flexibility of method 
        x_0: np.ndarray, the starting point of the minimiser
        stepsize: float, the stepsize of the minimiser
        method: str, default = "DFP" the method to use for the hessian approximation - allowed values "DFP", "BFGS"
        stop_tol: float, default = None, the value of the gradient at which convergence is determined. defaults to none to give a chance to get off local minima
        max_iter: int, default = 10000 the number of iterations to run
        detail: bool, default = False, show the time elapsed for the method to run

    Returns:
        tuple: the coordinates of minima, the minimum value of the function at this point
    """
    x = np.array(x_0, dtype=float)
    n_dim = len(x)
    G = np.eye(n_dim)

    def DFP(x_new, grad_new, x, grad, G):
        delta = x_new - x
        gamma = grad_new - grad

        if np.dot(gamma, delta) == 0:
            return G

        G = G + np.outer(delta, delta) / np.dot(gamma, delta) - \
            (G @ np.outer(gamma, gamma) @ G) / np.dot(gamma, G @ gamma)
        return G

    def BFGS(x_new, grad_new, x, grad, G):
        delta = x_new - x
        gamma = grad_new - grad
        dim = len(delta)

        temp = 1.0 / np.dot(gamma, delta)
        I = np.eye(dim)
        G = (I - temp * np.outer(delta, gamma)) @ G @ (I - temp * np.outer(gamma, delta)) \
            + temp * np.outer(delta, delta)

        return G

    if method == "DFP":
        grad_update = DFP

    elif method == "BFGS":
        grad_update = BFGS

    start = time.perf_counter()
    for i in range(max_iter):
        grad = df(x)
        if stop_tol:
            if np.linalg.norm(grad) < stop_tol:
                print("halting QN")
                break

        x_new = x - stepsize * G @ grad
        grad_new = df(x_new)

        G = grad_update(x_new, grad_new, x, grad, G)
        x = x_new

        if detail:
            print(f'iteration number: {i}, x: {x}, dx: {grad}')

    end = time.perf_counter()
    time_elapsed = end - start

    if detail:
        print(f"iteration number QN: {i}")
        print(f"time elapsed QN: {time_elapsed}")
    return x, f(x)
