"""
Session 3 computational physics 05/11
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm


"""
Problem Sheet 7
"""


def ParabolicPDE(D, u_0, u_N, x_i, x_f, N_time, N_space, delta_t):
    """
    Solves the diffusion equation using explicit euler for time stepping and central 
    difference scheme for spatial derivatives.
    """
    h = (x_f - x_i) / (N_space - 1)
    d = D * delta_t / (h**2)

    u = np.zeros((N_time, N_space))
    #  fix boundaries for all time
    u[:, 0] = u_0
    u[:, -1] = u_N

    for n in range(N_time - 1):
        for i in range(1, N_space - 1):
            u[n+1, i] = (1-2*d) * u[n, i] + d * (u[n, i-1] + u[n, i+1])
    return u


def ps7problem4():

    def sol(x, t, N_terms=500):
        s = 0
        for n in range(1, N_terms + 1):
            s += ((-1)**n / n) * np.sin(n*np.pi*x) * \
                np.exp(-n**2 * np.pi**2 * t**2)
        return x + (2/np.pi) * s

    delta_t = 0.00001
    N_space = 100
    N_time = 100000

    u = ParabolicPDE(D=1., u_0=0, u_N=1,
                     x_i=0, x_f=1,
                     N_time=N_time, N_space=N_space,
                     delta_t=delta_t)

    x = np.linspace(0, 1, N_space)

    plot_times = [0.01, 0.09, 0.25, 1]
    indices = [int(t / delta_t) for t in plot_times]

    fig, ax = plt.subplots(len(plot_times), 1, figsize=(6, 8), sharex=True)

    for i, (t, n) in enumerate(zip(plot_times, indices)):
        ax[i].plot(x, u[n, :], label=f't={t:.3f}s')
        ax[i].set_xlim(0, 1)
        ax[i].legend()
        ax[i].grid()

    plt.xlabel('x')
    plt.tight_layout()
    plt.show()

    sol_001 = sol(x, 0.01)
    sol_009 = sol(x, 0.09)
    sol_025 = sol(x, 0.25)

    plt.plot(x, sol_001, label="t=0.01")
    plt.plot(x, sol_009, label="t=0.09")
    plt.plot(x, sol_025, label="t=0.25")
    plt.legend()
    plt.grid()
    plt.show()
    """
Problem Sheet 6
"""


def AM4_linear_coupled(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    # With AB3 predictor
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
        nsteps = len(times)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    #  for the linear system
    A = np.array([[0, 1], [-1, 0]])
    u_n = np.array([f1_0, f2_0])

    # euler steps
    u_n1 = u_n + A @ u_n * delta_t
    u_n2 = u_n1 + A @ u_n1 * delta_t
    u_n3 = u_n2 + A @ u_n2 * delta_t

    u_n = [u_n, u_n1, u_n2, u_n3]
    for i in range(len(u_n) - 1, nsteps):
        u_pred = ((u_n[i]) + (delta_t/12) * (23 * (A @ u_n[i-3])
                                             # AB3 predictor
                                             - 16 * (A @ u_n[i-2]) + 5 * (A @ u_n[i-1])))

        u_next = (u_n[i] + (delta_t / 24.0) * (9.0 * (A @ u_pred)
                                               + 19.0 * (A @ u_n[i]) - 5.0 * (A @ u_n[i-1]) + 1.0 * (A @ u_n[i-2])))

        u_n.append(u_next)
    return u_n[-1][0], u_n[-1][1]  # returns u and v


def HOT4_linear_coupled(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    #  for the linear system
    A = np.array([[0, 1], [-1, 0]])
    I = np.eye(2)
    u_n = np.array([f1_0, f2_0])

    f1_values = []
    f2_values = []
    for _ in times:
        u_n = (u_n + (A @ u_n * delta_t) + (0.5 * (A @ A) @ u_n * delta_t**2)
               + (1/6 * (A @ A @ A) @ u_n * delta_t**3)
               + (1/24 * (A @ A @ A @ A) @ u_n * delta_t**4))  # 4th order
        f1_values.append(u_n[0])
        f2_values.append(u_n[1])
    return f1_values, f2_values


def HOT2_linear_coupled(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    #  for the linear system
    A = np.array([[0, 1], [-1, 0]])
    I = np.eye(2)
    u_n = np.array([f1_0, f2_0])

    f1_values = []
    f2_values = []
    for _ in times:
        u_n = u_n + (A @ u_n * delta_t) + (0.5 * (A @ A)
                                           @ u_n * delta_t**2)  # 2nd order
        f1_values.append(u_n[0])
        f2_values.append(u_n[1])
    return f1_values, f2_values


def RK2_linear_coupled(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None, alpha=0.5):
    f1_i = f1_0
    f2_i = f2_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    f1_values = []
    f2_values = []
    for t in times:
        f1_a = f1(t, f1_i, f2_i)
        f1_b = f1(t + (alpha * delta_t), f1_i + (f1_a * alpha * delta_t), f2_i)
        f2_a = f2(t, f1_i, f2_i)
        f2_b = f2(t + (alpha * delta_t), f1_i, f2_i + (f2_a * alpha * delta_t))

        f1_i = f1_i + 1/(2*alpha) * (f1_a + ((2*alpha) - 1) * f1_b) * delta_t
        f2_i = f2_i + 1/(2*alpha) * (f2_a + ((2*alpha) - 1) * f2_b) * delta_t
        f2_values.append(f2_i)
        f1_values.append(f1_i)

    return f1_values, f2_values


def linear_implicit_coupled_euler(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    """
    for the system d^2y/dx^2 = -y
    with a perfect step
    """
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    A = np.array([[0, 1], [-1, 0]])
    I = np.eye(2)
    G = np.linalg.inv(I - A * delta_t)
    u = np.array([f1_0, f2_0])

    f1_values = []
    f2_values = []
    for _ in times:
        u = G @ u
        f1_values.append(u[0])
        f2_values.append(u[1])
    return f1_values, f2_values


def impliciteuler_pc(f_prime, y_0, t_0, t_f, delta_t=None, nsteps=None):
    y_i = y_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    y = []
    for t in times:
        f_a = f_prime(t, y_i)
        f_b = f_prime((t + delta_t), y_i + (f_a * delta_t))
        y_i = y_i + (f_b*delta_t)
        y.append(y_i)
    return y


def implicit_coupled_euler(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    f1_i = f1_0
    f2_i = f2_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    f1_values = []
    f2_values = []
    for t in times:
        f1_b = f1_i/(1 + delta_t)
        f2_b = f2_i/(1 + delta_t)
        f1_i = f1_i + (f1(t + delta_t, f1_b, f2_b)*delta_t)
        f2_i = f2_i + (f2(t + delta_t, f1_b, f2_b)*delta_t)
        f1_values.append(f1_i)
        f2_values.append(f2_i)
    return f1_values, f2_values


def euler_solver(f_prime, y_0, t_0, t_f, delta_t=None, nsteps=None):
    y_i = y_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    y = []
    for t in times:
        y_i = y_i + (f_prime(t, y_i)*delta_t)
        y.append(y_i)
    return y


def coupled_euler_solver(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    f1_i = f1_0
    f2_i = f2_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    f1_values = []
    f2_values = []
    for t in times:
        # f1_i_prime = f1_i + (f1(t, f1_i, f2_i)*delta_t)
        # f2_i_prime = f2_i + (f2(t, f1_i, f2_i)*delta_t)
        # perfect step for euler method
        f1_i_prime = f1_i * np.cos(delta_t) + f2_i * np.sin(delta_t)
        f2_i_prime = -f1_i * np.sin(delta_t) + f2_i * np.cos(delta_t)

        f1_values.append(f1_i_prime)
        f2_values.append(f2_i_prime)
        f1_i = f1_i_prime
        f2_i = f2_i_prime

    return f1_values, f2_values


def RK4(f_prime, y_0, t_0, t_f, delta_t=None, nsteps=None):
    y_i = y_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    y = []
    for t in times:
        k1 = f_prime(t, y_i)
        k2 = f_prime(t + delta_t/2, y_i + (k1 * delta_t/2))
        k3 = f_prime(t + delta_t/2, y_i + (k2 * delta_t/2))
        k4 = f_prime(t + delta_t, y_i + (k3 * delta_t))

        y_i = y_i + delta_t * (k1 + 2*k2 + 2*k3 + k4) / 6
        y.append(y_i)
    return y


def coupled_RK4_solver(f1, f2, f1_0, f2_0, t_0, t_f, delta_t=None, nsteps=None):
    f1_i = f1_0
    f2_i = f2_0
    t = t_0
    if nsteps is None:
        times = np.arange(t_0, t_f, delta_t)
    elif delta_t is None:
        times = np.linspace(t_0, t_f, nsteps)
        delta_t = times[1] - times[0]
    else:
        raise ValueError("Please enter only one of timestep or nsteps")

    f1_values = []
    f2_values = []
    for t in times:
        k1_f1 = f1(t, f1_i, f2_i)
        k2_f1 = f1(t + delta_t/2, f1_i + (k1_f1 * delta_t/2), f2_i)
        k3_f1 = f1(t + delta_t/2, f1_i + (k2_f1 * delta_t/2), f2_i)
        k4_f1 = f1(t + delta_t, f1_i + (k3_f1 * delta_t), f2_i)

        k1_f2 = f2(t, f1_i, f2_i)
        k2_f2 = f2(t + delta_t/2, f1_i, f2_i + (k1_f2 * delta_t/2))
        k3_f2 = f2(t + delta_t/2, f1_i, f2_i + (k2_f2 * delta_t/2))
        k4_f2 = f2(t + delta_t, f1_i, f2_i + (k3_f2 * delta_t))

        f2_i = f2_i + delta_t * (k1_f2 + 2*k2_f2 + 2*k3_f2 + k4_f2) / 6
        f2_values.append(f2_i)
        f1_i = f1_i + delta_t * (k1_f1 + 2*k2_f1 + 2*k3_f1 + k4_f1) / 6
        f1_values.append(f1_i)

    return f1_values, f2_values


def problem2c():
    def du(t, u, v):
        return v

    def dv(t, u, v):
        return -u

    stepsizes = np.linspace(0.0000001, 0.1, 5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for stepsize in stepsizes:
        t = np.arange(0, 1, stepsize)
        u, v = coupled_euler_solver(
            du, dv, f1_0=0, f2_0=1, t_0=0, t_f=1, delta_t=stepsize)
        axes[0].plot(t, u, label=f"Δt={stepsize}")
    axes[0].plot(t, np.sin(t), 'k--', label='Analytic (sin t)')
    axes[0].set_title("Explicit Euler")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y(t)")
    axes[0].legend()
    axes[0].grid(True)

    for stepsize in stepsizes:
        t = np.arange(0, 1, stepsize)
        u, v = linear_implicit_coupled_euler(
            du, dv, f1_0=0, f2_0=1, t_0=0, t_f=1, delta_t=stepsize)
        axes[1].plot(t, u, label=f"Δt={stepsize}")
    axes[1].plot(t, np.sin(t), 'k--', label='Analytic (sin t)')
    axes[1].set_title("Implicit Euler (Iterative)")
    axes[1].set_xlabel("t")
    axes[1].legend()
    axes[1].grid(True)


plt.tight_layout()
plt.show()


def problem2d():
    def du(t, u, v):
        return v

    def dv(t, u, v):
        return -u

    sizes = np.logspace(-7, -5, 10)
    errors = []
    for stepsize in sizes:
        u, v = AM4_linear_coupled(
            du, dv, f1_0=0, f2_0=1, t_0=0, t_f=1, delta_t=stepsize)
        t = np.arange(0, 1, stepsize)
        # plt.plot(t, u, label='HOT2')
        # plt.plot(t, np.sin(t), 'k--', label='Analytic (sin t)')
        # plt.xlabel("t")
        # plt.ylabel("y(t)")
        # plt.title("Higher Order Taylor Method")
        # plt.legend()
        # plt.grid()
        # plt.show()

        error = np.abs(np.array(u) - np.sin(t))
        errors.append(np.max(error))

    plt.scatter(sizes, errors)
    plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error in HOT4 Method")
    plt.grid()
    plt.show()


def problem3():
    def bisection(f, xlow, xhigh, max_err=1e-5, max_iter=100):
        for _ in range(max_iter):
            x_mid = (xlow + xhigh) / 2
            fx_mid = f(x_mid)
            if np.abs(fx_mid) < max_err:
                return x_mid
            elif fx_mid > 0:
                xhigh = x_mid
            else:
                xlow = x_mid
        return x_mid

    def shooting(f1, f2, f_0, t_0, f_N, t_N, initial_guesses, nsteps, max_error=1e-2):
        iterations = 0
        guesses = initial_guesses
        found = False
        true_fprime = 0

        f_N_est, _ = coupled_euler_solver(
            f1, f2, f_0, guesses[0], t_0, t_N, nsteps=nsteps)
        error = f_N_est[-1] - f_N
        errors = [error]

        pbar = tqdm(desc="Finding root", unit=" iteration")
        while not found:
            f_N_est, _ = coupled_euler_solver(
                f1, f2, f_0, guesses[-1], t_0, t_N, nsteps=nsteps)
            error = f_N_est[-1] - f_N

            if np.abs(error) < max_error:
                true_fprime = guesses[-1]
                # print(error)
                found = True
                pbar.set_description("Root FOUND!")
                break

            errors.append(error)

            # sort guesses
            guesses, errors = zip(*sorted(zip(guesses, errors)))
            guesses, errors = list(guesses), list(errors)
            error_interp = interp1d(guesses, errors, kind='linear')

            #  find roots of errors that corresponds to a guess
            if errors[-1] * errors[0] > 0:
                raise ValueError(
                    "No sign change detected — cannot perform bisection.")

            root = bisection(error_interp, guesses[0], guesses[-1])
            iterations += 1
            pbar.update(1)
            guesses.append(root)

        pbar.close()
        print(true_fprime)
        sol, _ = coupled_euler_solver(
            f1, f2, f_0, true_fprime, t_0, t_N, nsteps=nsteps)
        return sol, iterations

    def matrix_method(dv, u_0, u_N, t_0, t_N, nsteps, u_guess=None, iterations=100):
        # here, du is not needed
        if u_guess is None:
            u_guess = np.zeros(nsteps - 1)  # a filler if dv only depends on t

        def inv_A(N):  # represents the inverse of the second derivative matrix
            Ai = np.zeros((N-1, N-1))
            for i in range(N-1):
                Ai[i][0] = 1 + i - N
                Ai[0][i] = Ai[i][0]
                for j in range(1, i+1):
                    Ai[i][j] = (j+1) * Ai[i][0]
                    Ai[j][i] = Ai[i][j]
            return Ai / N

        delta_t = (t_N - t_0)/nsteps
        # iniialize arrays of t, u, v
        t_n = np.linspace(t_0, t_N, nsteps - 1)
        u_n = u_guess
        v_n = np.zeros(nsteps - 1)
        # construct b
        b = np.zeros(nsteps - 1)

        for _ in range(iterations):
            for i in range(nsteps - 1):
                f_i = dv(t_n[i], u_n[i], v_n[i])
                b[i] = f_i * delta_t**2
            b[0] -= u_0
            b[-1] -= u_N

            u_n = inv_A(nsteps).dot(b)
        return u_n

    def x_analytic(t):
        return (t**3)/6 + (t/3)

    def problem3c():
        def x_2prime(t, y):
            return t

        nsteps = 100
        x_prime = euler_solver(x_2prime, y_0=1/3, t_0=0, t_f=1, nsteps=nsteps)

        t = np.linspace(0, 1, nsteps)
        x_interp = interp1d(t, x_prime, kind='linear')

        def x_prime_f(t, y):
            return x_interp(t)

        x = euler_solver(x_prime_f, y_0=0, t_0=0, t_f=1, nsteps=nsteps)

        plt.plot(t, x, label='euler method')
        plt.plot(t, x_analytic(t), label='analytic solution')
        plt.xlabel("t")
        plt.ylabel("x")
        plt.legend()
        plt.grid()
        plt.show()

    def du(t, u, v):
        return v

    def dv(t, u, v):
        return t

    def gen_steps(n):
        steps = [10]
        for i in range(1, n+1):
            steps.append(steps[i-1] * 2)
        return steps

    matrix_errors = []
    shooting_errors = []
    eps = 1e-12

    steps = gen_steps(8)
    print(steps)
    for step in steps:
        t = np.linspace(0, 1, step - 1)
        analytic_points = x_analytic(t)
        u_n = matrix_method(dv, u_0=0, u_N=1/2, t_0=0, t_N=1, nsteps=step)
        relative_error_matrix = np.abs(
            u_n - analytic_points) / (np.abs(analytic_points) + eps)
        solution, _ = shooting(du, dv, f_0=0, f_N=1/2, t_0=0,
                               t_N=1, initial_guesses=[0, 1], nsteps=step-1)
        relative_error_shooting = np.abs(
            np.array(solution) - analytic_points) / (np.abs(x_analytic(t)) + eps)
        matrix_errors.append(np.max(relative_error_matrix))
        shooting_errors.append(np.max(relative_error_shooting))

        # plt.plot(t, u_n, label='matrix method')
        # plt.plot(t, solution, label='shooting method')
        # plt.plot(t, analytic_points, label='analytic solution')
        # plt.xlabel("t")
        # plt.ylabel("x")
        # plt.legend()
        # plt.grid()
        # plt.show()

    plt.plot(steps, shooting_errors, label='shooting method')
    plt.plot(steps, matrix_errors, label='matrix method')
    plt.yscale('log')
    plt.xlabel("number of steps")
    plt.ylabel("relative error")
    plt.legend()
    plt.grid()
    plt.show()

    def f1(t, x, v):
        return v

    def f2(t, x, v):
        return 3*x - 0.5 * t**3

    u_guess = [t_i/2 for t_i in np.linspace(0, 1, 100)]
    solution = matrix_method(f2, u_0=0, u_N=0.5, t_0=0,
                             t_N=1, nsteps=100, u_guess=u_guess)
    t = np.linspace(0, 1, 99)
    plt.plot(t, solution, label='matrix method')
    plt.plot(t, x_analytic(t), label='analytic solution')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.grid()
    plt.show()

    # def f(t, y):
    #     return np.cos(y)

    # y_0 = 0
    # t_0 = 0
    # t_f = 10
    # delta_t = 1e-5
    # y = euler_solver(f, y_0, t_0, t_f, delta_t)
    # t = np.arange(t_0, t_f, delta_t)

    # plt.plot(t,y)
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    ps7problem4()
