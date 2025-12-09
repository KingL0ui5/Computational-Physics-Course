"""
A module containing the minimisers to find the ground state energy of wavefunctions.
Louis Liu 22/11
"""

from typing import Callable
import numpy as np
import time


class hydrogen_molecule_minimisers:
    @staticmethod
    def E_fn(thetas, q1, q2, r1, r2):
        from hydrogen_molecule import h2_wavefunction
        wf_temp = h2_wavefunction(thetas, q1, q2)
        return np.nanmean(wf_temp.local_energy(r1, r2))

    @staticmethod
    def log_grad(thetas, q1, q2, r1, r2, stepsize=8.008e-03) -> np.ndarray:
        """
        Calculates the gradient using the log-derivative method and central_difference.
        """
        from modules.differentiators import central_difference
        from hydrogen_molecule import h2_wavefunction

        wf_current = h2_wavefunction(thetas, q1, q2)
        el = wf_current.local_energy(r1, r2)
        e_mean = np.nanmean(el)

        weights = 2.0 * (el - e_mean)

        def wrapper(t_array):
            t = t_array[0]
            wf_temp = h2_wavefunction(t, q1, q2)

            psi_vals = wf_temp.psi(r1, r2)
            psi_vals = np.maximum(psi_vals, 1e-100)

            log_psi = np.log(psi_vals)
            return np.nanmean(weights * log_psi)

        x_params = np.array([thetas])

        grads = central_difference(
            wrapper, x_params, h=[stepsize]*3, order=8)
        return grads.flatten()

    @staticmethod
    #  optimal stepsize for order 2 is 1e-4, order 8 is 8.008e-03
    def grad(thetas, q1, q2, r1, r2, stepsize=4.004e-04, order=4) -> np.ndarray:
        from modules.differentiators import central_difference
        #  d/dt1

        def wrapper_1(theta_1):
            t = [theta_1, thetas[1], thetas[2]]
            return hydrogen_molecule_minimisers.E_fn(t, q1, q2, r1, r2)

        #  d/dt2
        def wrapper_2(theta_2):
            t = [thetas[0], theta_2, thetas[2]]
            return hydrogen_molecule_minimisers.E_fn(t, q1, q2, r1, r2)

        #  d/dt3
        def wrapper_3(theta_3):
            t = [thetas[0], thetas[1], theta_3]
            return hydrogen_molecule_minimisers.E_fn(t, q1, q2, r1, r2)

        dE_t1 = central_difference(
            wrapper_1, x=[thetas[0]], h=[stepsize], order=order).item()
        dE_t2 = central_difference(
            wrapper_2, x=[thetas[1]], h=[stepsize], order=order).item()
        dE_t3 = central_difference(
            wrapper_3, x=[thetas[2]], h=[stepsize], order=order).item()

        # returns the gradient of E
        return np.array([dE_t1, dE_t2, dE_t3])

    @staticmethod
    def sample_coords(wf, Ns, r_0, thinning: int = 20, detail=False) -> tuple:
        """
        Sample wavefunction and process into r1, r2.
        Parameters:
            wf: h2_wavefunction, The hydrogen molecule wavefunction object.
            Ns: int, The number of samples to generate.
            r_0: np.ndarray, The initial position to start sampling from.
            thinning: int, The thinning factor to reduce autocorrelation in samples.
        Returns:
            tuple: r1, r2, positions of electrons as np.ndarrays of shape (N, 3)
        """
        from modules.function_sampling import metropolis_hastings

        def wavefunction_wrapper(coords):
            coords = np.asarray(coords)
            r1, r2 = coords[:, 0:3], coords[:, 3:6]
            return wf.probability_density(r1, r2)

        #  sample once for current state
        samples = metropolis_hastings(f=wavefunction_wrapper, f_prop='gaussian', x_0=r_0, xmin=[
                                      -10.]*6, xmax=[10.]*6, N=Ns, kwrgs={'sigma': 0.8}, thinning=thinning)
        r = samples[len(samples)//10:]
        r1, r2 = r[:, 0:3], r[:, 3:6]

        if detail:
            from modules.helpers import get_acf
            acf = get_acf(samples, max_lag=500)
            ess = len(samples) / (1 + 2 * np.sum(acf[1:]))
            print(f"Effective Sample Size (ESS): {ess:.1f}")
        return r1, r2

    def simulated_annealing(q1: np.ndarray, q2: np.ndarray, x_0: np.ndarray, initial_temp: float, cooling_rate: float, std: float = 0.05, max_iter: int = 100, xmin: float = 0.7, xmax: float = 5., Ns: int = 10000, detail: bool = False):
        """
        Find the minima of a function using the simulated annealing method.
        Parameters:
            q1 (np.ndarray): Position of the first nucleus.
            q2 (np.ndarray): Position of the second nucleus.
            x_0: np.ndarray, the starting point of the minimiser
            Ns: int, number of samples to take per wavefunction iteration
            xmin: float, the minimum value for theta parameters
            xmax: float, the maximum value for theta parameters
            std: float, the standard deviation of the normal distribution used for proposing new states
            initial_temp: float, the starting temperature of the system
            cooling_rate: float, the rate at which the temperature decreases
            max_iter: int, the number of iterations to run
            detail: bool, default = False
        """
        from hydrogen_molecule import h2_wavefunction

        r_0 = np.ones(6, dtype=float)  #  for samples
        x = np.array(x_0, dtype=float)
        T = initial_temp

        wf = h2_wavefunction(x, q1, q2)
        r1, r2 = hydrogen_molecule_minimisers.sample_coords(wf, Ns, r_0)

        E = hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)

        if detail:
            print("starting simulated annealing...")
            Es = [E]
            Ts = [T]
            xs = [x.copy()]

        start = time.perf_counter()
        for i in range(max_iter):
            Ns_i = Ns
            x_new = x + np.random.normal(0, std, size=x.shape)
            #  bound the theta values
            if any(x_new < xmin) or any(x_new > xmax):
                continue

            wf_new = h2_wavefunction(x_new, q1, q2)
            r1_new, r2_new = hydrogen_molecule_minimisers.sample_coords(
                wf_new, Ns_i, r_0)

            E_new = hydrogen_molecule_minimisers.E_fn(
                x_new, q1, q2, r1_new, r2_new)
            delta_E = E_new - E

            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                x, E, r1, r2 = x_new, E_new, r1_new, r2_new

            T *= cooling_rate

            if detail:
                print(f'iteration number: {i}, x: {x}, E: {E}, T: {T}')
                Es.append(E)
                Ts.append(T)
                xs.append(x.copy())

        end = time.perf_counter()
        if detail:
            import matplotlib.pyplot as plt
            time_elapsed = end - start
            print(
                f"time elapsed: {time_elapsed}, per iteration (avg)={time_elapsed/i}")
            xs = np.array(xs)
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))
            iterations = range(len(xs))
            ax[0].plot(iterations, xs[:, 0], label='theta 1')
            ax[0].plot(iterations, xs[:, 1], label='theta 2')
            ax[0].plot(iterations, xs[:, 2], label='theta 3')
            ax[0].set_xlabel("Iteration")
            ax[0].set_ylabel("Theta Values")
            ax[0].set_title("Theta Values over Simulated Annealing Iterations")
            ax[0].legend()

            ax[1].plot(range(len(Ts)), Ts)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Temperature")
            ax[1].set_title("Temperature over Simulated Annealing Iterations")

            ax[2].plot(range(len(Es)), Es)
            ax[2].set_xlabel("Iteration")
            ax[2].set_ylabel("Energy")
            ax[2].set_title("Energy over Simulated Annealing Iterations")

            plt.show()
            print(
                f"final iteration number SA: {i}, final energy: {E}, final x: {x}")
        return x, E

    @staticmethod
    def stochastic_reconfiguration(q1: np.ndarray, q2: np.ndarray, x_0: np.ndarray, alpha: float, max_iter: int = 100, stop_tol: float = 1e-6, N_s: int = 10000, epsilon: float = 1e-3, detail: bool = False) -> tuple:
        """
        Finds the minima using Stochastic Reconfiguration (Natural Gradient Descent).

        Parameters:
            epsilon: float, regularization shift for the S-matrix inversion (stabilizer).
            q1 (np.ndarray): Position of the first nucleus. 
            q2 (np.ndarray): Position of the second nucleus.
            x_0: np.ndarray, the starting point of the minimiser
            alpha: float, the stepsize of the minimiser
            max_iter: int, the number of iterations to run
            stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
            N_s: number of samples to take per wavefunction iteration
            detail: bool, default = False
        Returns:
            tuple: the coordinates of minima, the minimum value of the function at this point
        """
        from hydrogen_molecule import h2_wavefunction

        print("Starting Stochastic Reconfiguration minimiser...")
        x = np.asarray(x_0, dtype=float)
        r_0 = np.ones(6, dtype=float)  # for samples

        gradient_changes = []
        df_last = np.zeros_like(x)
        Es = []
        start = time.perf_counter()
        for i in range(max_iter):
            #  adjust learning rate per iteration
            Ns_i = N_s  # * int(np.exp((i+1)*0.05))
            wf = h2_wavefunction(x, q1, q2)

            #   sample coordinates for current thetas
            r1, r2 = hydrogen_molecule_minimisers.sample_coords(
                wf, Ns_i, r_0, detail=detail)
            N_actual = len(r1)
            df = hydrogen_molecule_minimisers.log_grad(
                x, q1, q2, r1, r2)

            #  scale alpha based on the derivative
            grad_norm = np.linalg.norm(df)
            scaling_factor = np.clip(grad_norm, 0.1, 1.0)
            alpha_i = alpha * scaling_factor

            n_params = len(x)
            h_diff = 1e-6
            #  find S matrix
            O_k = np.zeros((N_actual, n_params))

            for k in range(n_params):
                x_plus = x.copy()
                x_plus[k] += h_diff
                x_minus = x.copy()
                x_minus[k] -= h_diff

                wf_plus = h2_wavefunction(x_plus, q1, q2)
                prob_plus = wf_plus.psi(r1, r2)

                wf_minus = h2_wavefunction(x_minus, q1, q2)
                prob_minus = wf_minus.psi(r1, r2)

                prob_plus = np.maximum(prob_plus, 1e-100)
                prob_minus = np.maximum(prob_minus, 1e-100)
                log_derivs = (np.log(prob_plus) -
                              np.log(prob_minus)) / (2 * h_diff)

                O_k[:, k] = log_derivs

            O_mean = np.mean(O_k, axis=0)
            O_centered = O_k - O_mean
            S = (O_centered.T @ O_centered) / N_actual

            S_reg = S + epsilon * np.eye(len(x))

            try:
                delta_p = np.linalg.solve(S_reg, -df)
            except np.linalg.LinAlgError:
                print("Singular matrix, reverting to gradient descent.")
                delta_p = -df

            #  stopping condition
            if stop_tol and np.linalg.norm(df) < stop_tol:
                break

            if np.isnan(x.any()) or np.isinf(x.any()):
                raise ValueError("Optimal parameters diverged to nan or inf")

            # Update
            x = x + alpha_i * delta_p

            #  restrict x to be positive
            if (x <= 1e-7).any():
                x = np.maximum(x, 1e-7)

            if detail:
                E_min = hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)
                Es.append(E_min)
                print(
                    f"iteration {i}, x={x}, df = {df},||d/df|| = {np.linalg.norm(df)}, \nN_s={Ns_i}, Natural grad update: {delta_p}, alpha={alpha_i},\nLocal energy: {E_min}")
                gradient_changes.append(df-df_last)
                df_last = df

        end = time.perf_counter()
        time_elapsed = end - start

        if detail:
            import matplotlib.pyplot as plt
            iterations = range(i+1)
            gradient_changes = np.array(gradient_changes)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(
                iterations, gradient_changes[:, 0], color='r', label='dTheta1')
            ax[0].plot(
                iterations, gradient_changes[:, 1], color='g', label='dTheta2')
            ax[0].plot(
                iterations, gradient_changes[:, 2], color='b', label='dTheta3')
            ax[0].set_xlabel("Iteration")
            ax[0].set_ylabel("Change in Gradient")
            ax[0].set_title("Gradient Changes Over Iterations")
            ax[0].legend()

            ax[1].plot(range(len(Es)), Es)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Energy")
            ax[1].set_title("Energy over Gradient Descent Iterations")
            plt.show()

            print(f"iteration number stochastic reconfig: {i}")
            print(
                f"time elapsed stochastic reconfig: {time_elapsed}, per iteration (avg)={time_elapsed/i}")
            print(f"final derivative: {df}")
        return x, hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)

    def gradient_descent(q1: np.ndarray, q2: np.ndarray, x_0: np.ndarray, alpha: float, max_iter: int = 100, stop_tol: float = 1e-6, N_s: int = 10000, detail: bool = False) -> tuple:
        """
        A method to find the minima of a function using the gradient descent method.
        Parameters:
            q1 (np.ndarray): Position of the first nucleus.
            q2 (np.ndarray): Position of the second nucleus.
            x_0: np.ndarray, the starting point of the minimiser
            alpha: float, the stepsize of the minimizer
            max_iter: int, the number of iterations to run
            stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
            N_s: number of samples to take per wavefunction iteration
            detail: bool, default = False

        Returns:
            tuple: the coordinates of minima, the minimum value of the function at this point
        """
        from hydrogen_molecule import h2_wavefunction

        print("Starting gradient descent minimiser...")
        x = np.asarray(x_0, dtype=float)

        r_0 = np.ones(6, dtype=float)  #  for samples

        gradient_changes = []
        df_last = np.zeros_like(x)

        #  minimisation loop
        Es = []
        start = time.perf_counter()
        for i in range(max_iter):

            Ns_i = N_s  # * int(np.exp((i+1)*0.05))
            wf = h2_wavefunction(x, q1, q2)

            #   sample coordinates for current thetas
            r1, r2 = hydrogen_molecule_minimisers.sample_coords(wf, Ns_i, r_0)

            df = hydrogen_molecule_minimisers.log_grad(
                x, q1, q2, r1, r2)

            #  scale alpha based on the derivative
            grad_norm = np.linalg.norm(df)
            scaling_factor = np.clip(grad_norm, 0.1, 1.0)
            alpha_i = alpha * scaling_factor

            #   test stopping condition
            if stop_tol:
                if np.linalg.norm(df) < stop_tol:
                    break
            #   halt at nan values
            if np.isnan(x.any()) or np.isinf(x.any()):
                raise ValueError("Optimal parameters diverged to nan or inf")

            x = x - (alpha_i * df)

            #  restrict x to be positive
            # if (x <= 1e-7).any():
            #     x = np.maximum(x, 1e-7)

            if detail:
                print(
                    f"iteration {i}, x={x}, d/dx={df}, N_s={Ns_i}, alpha={alpha_i}")
                E_min = hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)
                Es.append(E_min)
                print(f"local energy: {E_min}")
                gradient_changes.append(df-df_last)
                df_last = df

        end = time.perf_counter()
        time_elapsed = end - start

        if detail:
            import matplotlib.pyplot as plt
            iterations = range(i+1)
            gradient_changes = np.array(gradient_changes)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(
                iterations, gradient_changes[:, 0], color='r', label='dTheta1')
            ax[0].plot(
                iterations, gradient_changes[:, 1], color='g', label='dTheta2')
            ax[0].plot(
                iterations, gradient_changes[:, 2], color='b', label='dTheta3')
            ax[0].set_xlabel("Iteration")
            ax[0].set_ylabel("Change in Gradient")
            ax[0].set_title("Gradient Changes Over Iterations")
            ax[0].legend()

            ax[1].plot(range(len(Es)), Es)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Energy")
            ax[1].set_title("Energy over Gradient Descent Iterations")
            plt.show()

            print(f"iteration number GD: {i}")
            print(
                f"time elapsed GD: {time_elapsed}, per iteration (avg)={time_elapsed/i}")
            print(f"final derivative: {df}")

        return x, hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)

    def RMSProp_GD(q1: np.ndarray, q2: np.ndarray, x_0: np.ndarray, alpha: float, forgetting: float, max_iter: int = 1000, stop_tol: float = 1e-6, N_s: int = 10000, detail: bool = False) -> tuple:
        """
        A method to find the minima of a function using the RMSProp adjusted gradient descent method.
        Parameters:
            q1 (np.ndarray): Position of the first nucleus.
            q2 (np.ndarray): Position of the second nucleus.
            x_0: np.ndarray, the starting point of the minimiser
            N_s: int, number of samples to take per wavefunction iteration
            stepsize: float, the stepsize of the minimiser
            forgetting: float, the forgetting factor of the minimiser
            max_iter: int, the number of iterations to run
            stop_tol: float, default = 1e-6, the value of the gradient at which convergence is determined
            detail: bool, default = False

        Returns:
            tuple: the coordinates of minima, the minimum value of the function at this point
        """
        from hydrogen_molecule import h2_wavefunction
        print("Starting RMSProp gradient descent minimiser...")

        x = np.asarray(x_0, dtype=float)
        r_0 = np.ones(6, dtype=float)  #  for samples
        v = np.ones_like(x)

        gradient_changes = []
        df_last = np.zeros_like(x)
        Es = []
        start = time.perf_counter()
        for i in range(max_iter):
            Ns_i = N_s  # * int(np.exp((i+1)*0.05))

            wf = h2_wavefunction(x, q1, q2)
            r1, r2 = hydrogen_molecule_minimisers.sample_coords(wf, Ns_i, r_0)

            df = hydrogen_molecule_minimisers.log_grad(
                x, q1, q2, r1, r2)

            #  scale alpha based on the derivative
            grad_norm = np.linalg.norm(df)
            scaling_factor = np.clip(grad_norm, 0.1, 1.0)
            alpha_i = alpha * scaling_factor

            if stop_tol:
                if np.linalg.norm(df) < stop_tol:
                    break

            if np.isnan(x.any()) or np.isinf(x.any()):
                raise ValueError("Optimal parameters diverged to nan or inf")

            v = forgetting * v + (1 - forgetting)*(df**2)
            x = x - (alpha / np.sqrt(v)) * df

            #  restrict x to be positive
            if (x <= 1e-7).any():
                x = np.maximum(x, 1e-7)

            if detail:
                print(
                    f"iteration {i}, x={x}, d/dx={df}, N_s={Ns_i}, alpha={alpha_i}")
                E_min = hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)
                Es.append(E_min)
                print(f"local energy: {E_min}")
                gradient_changes.append(df-df_last)
                df_last = df

        end = time.perf_counter()
        time_elapsed = end - start

        if detail:
            import matplotlib.pyplot as plt
            iterations = range(i+1)
            gradient_changes = np.array(gradient_changes)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(
                iterations, gradient_changes[:, 0], color='r', label='dTheta1')
            ax[0].plot(
                iterations, gradient_changes[:, 1], color='g', label='dTheta2')
            ax[0].plot(
                iterations, gradient_changes[:, 2], color='b', label='dTheta3')
            ax[0].set_xlabel("Iteration")
            ax[0].set_ylabel("Change in Gradient")
            ax[0].set_title("Gradient Changes Over Iterations")
            ax[0].legend()

            ax[1].plot(range(len(Es)), Es)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Energy")
            ax[1].set_title("Energy over Gradient Descent Iterations")
            plt.show()

            print(f"iteration number GD: {i}")
            print(f"time elapsed GD: {time_elapsed}")
            print(f"final derivative: {df}")
        return x, hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)

    def quasi_newton(q1: np.ndarray, q2: np.ndarray, x_0: np.ndarray, alpha: float, method: str = "DFP", max_iter: int = 1000, stop_tol: float = None, N_s: int = 10000, detail: bool = False) -> tuple:
        """
        A method to find the minima of a function using the quasi-newton method using a chosen method to approximate the hessian
        Parameters:
            q1 (np.ndarray): Position of the first nucleus.
            q2 (np.ndarray): Position of the second nucleus.
            x_0: np.ndarray, the starting point of the minimiser
            alpha: float, the stepsize of the minimiser
            method: str, default = "DFP" the method to use for the hessian approximation - allowed values "DFP", "BFGS"
            stop_tol: float, default = None, the value of the gradient at which convergence is determined. defaults to none to give a chance to get off local minima
            max_iter: int, default = 10000 the number of iterations to run
            N_s: int, default = 10000, number of samples to take per wavefunction iteration
            detail: bool, default = False

        Returns:
            tuple: the coordinates of minima, the minimum value of the function at this point
        """
        from hydrogen_molecule import h2_wavefunction

        print("Starting quasi-newton minimiser...")
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
            G = (I - temp * np.outer(delta, gamma)
                 ) @ G @ (I - temp * np.outer(gamma, delta))
            + temp * np.outer(delta, delta)
            return G

        if method == "DFP":
            grad_update = DFP

        elif method == "BFGS":
            grad_update = BFGS

        r_0 = np.ones(6, dtype=float)  #  for samples

        Es = []
        gradient_changes = []
        start = time.perf_counter()
        for i in range(max_iter):
            wf = h2_wavefunction(x, q1, q2)
            Ns_i = N_s  # * int(np.exp((i+1)*0.05))
            r1, r2 = hydrogen_molecule_minimisers.sample_coords(wf, Ns_i, r_0)
            grad = hydrogen_molecule_minimisers.log_grad(
                x, q1, q2, r1, r2)

            if stop_tol:
                if np.linalg.norm(grad) < stop_tol:
                    print("halting QN")
                    break

            x_new = x - alpha * G @ grad

            wf_new = h2_wavefunction(x_new, q1, q2)
            r1_new, r2_new = hydrogen_molecule_minimisers.sample_coords(
                wf_new, Ns_i, r_0)
            grad_new = hydrogen_molecule_minimisers.log_grad(
                x_new, q1, q2, r1_new, r2_new)

            G = grad_update(x_new, grad_new, x, grad, G)
            x = x_new

            if detail:
                print(
                    f"iteration {i}, x={x}, d/dx={grad}, N_s={Ns_i}")
                E_min = hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)
                Es.append(E_min)
                print(f"local energy: {E_min}")
                gradient_changes.append(grad_new-grad)

        end = time.perf_counter()
        time_elapsed = end - start

        if detail:
            import matplotlib.pyplot as plt
            iterations = range(i+1)
            gradient_changes = np.array(gradient_changes)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(
                iterations, gradient_changes[:, 0], color='r', label='dTheta1')
            ax[0].plot(
                iterations, gradient_changes[:, 1], color='g', label='dTheta2')
            ax[0].plot(
                iterations, gradient_changes[:, 2], color='b', label='dTheta3')
            ax[0].set_xlabel("Iteration")
            ax[0].set_ylabel("Change in Gradient")
            ax[0].set_title("Gradient Changes Over Iterations")
            ax[0].legend()

            ax[1].plot(range(len(Es)), Es)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Energy")
            ax[1].set_title("Energy over Gradient Descent Iterations")
            plt.show()

            print(f"iteration number GD: {i}")
            print(
                f"time elapsed GD: {time_elapsed}, per iteration (avg)={time_elapsed/i}")
            print(f"final derivative: {grad}")
        return x, hydrogen_molecule_minimisers.E_fn(x, q1, q2, r1, r2)


class hydrogen_atom_minimisers:
    def quasi_newton(wf: Callable, dH: Callable, x_0: np.ndarray, stepsize: float, method: str = "DFP", max_iter: int = 1000, N_s: int = 1000, stop_tol: float = None, detail: bool = False) -> tuple:
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
            detail: bool, default = False
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
            G = (I - temp * np.outer(delta, gamma)
                 ) @ G @ (I - temp * np.outer(gamma, delta))
            + temp * np.outer(delta, delta)

            return G

        if method == "DFP":
            grad_update = DFP

        elif method == "BFGS":
            grad_update = BFGS

        start = time.perf_counter()
        for i in range(max_iter):
            psi = wf(theta=x[0])  # hydrogen wavefunction given theta

            #  increase number of samples each iteration
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
                print(
                    f"iteration {i}, x={x}, G={G}, gradient={grad}, Ns = {Ns_i}")

        end = time.perf_counter()
        time_elapsed = end - start

        if detail:
            print(f"iteration number QN: {i}")
            print(f"time elapsed QN: {time_elapsed}")
        return x, np.nanmean(psi.local_energy(coords=samples))

    def gradient_descent(wf: Callable, dH: Callable, x_0: np.ndarray, stepsize: float, max_iter: int = 100, stop_tol: float = 1e-6, N_s: int = 10000, detail: bool = False) -> tuple:
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
            detail: bool, default = False

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
            alpha = stepsize / np.sqrt(i+1)

            samples = metropolis_hastings(f=psi.probability_density, f_prop='gaussian', x_0=[
                0., 0., 0.], xmin=[-4., -4., -4.], xmax=[4., 4., 4.], N=Ns_i, kwrgs={'sigma': 0.8})
            samples = samples[Ns_i//10:]

            d = dH(x, samples)

            #  test stopping condition
            if stop_tol:
                if np.linalg.norm(d) < stop_tol:
                    break

            x = x - alpha * d
            #  restrict x to be positive - theta should not be negative
            if x <= 1e-7:
                x = 1e-7

            if detail:
                print(
                    f"iteration {i}, x={x}, d/dx={d}, N_s={Ns_i}, alpha={alpha}")

        end = time.perf_counter()
        time_elapsed = end - start

        if detail:
            print(f"iteration number GD: {i}")
            print(f"time elapsed GD: {time_elapsed}")
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
        detail: bool, default = False

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
        detail: bool, default = False

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
        detail: bool, default = False

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
        G = (I - temp * np.outer(delta, gamma)
             ) @ G @ (I - temp * np.outer(gamma, delta))
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
