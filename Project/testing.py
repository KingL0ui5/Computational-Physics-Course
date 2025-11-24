from harmonic_oscillator import eigenfunctions
from function_sampling import metropolis_hastings, MALA, stochasticMALA
import differentiators

import numpy as np
import seaborn as sns
from helpers import rms
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")

# %% analytical functions for testing


def get_acf(series, max_lag=200):
    series = series.flatten()
    series = series - np.mean(series)
    corr = np.correlate(series, series, mode='full')
    corr = corr[len(corr)//2:]
    corr = corr / corr[0]
    return corr[:max_lag]


def analytical_derivative(n: int):
    """
    Compute the analytical second derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
    Parameters:
        n: int, The quantum number of the eigenfunction
    Returns:
        Callable: The second derivative function
    """
    eigenfunction, _ = eigenfunctions(n)

    def f(x):
        x = np.asarray(x[0])
        return [(x**2 - (2*n + 1)) * eigenfunction(x)]
    return f


def analytical_first_derivative(n: int):
    """
    Compute the analytical first derivative of the nth eigenfunction of the 1D harmonic oscillator at position x
    Parameters:
        n: int, The quantum number of the eigenfunction
    Returns:
        Callable: The first derivative function
    """
    psi_n, _ = eigenfunctions(n)

    if n == 0:
        def f_ground(x):
            x = np.asarray(x[0])
            return [-x * psi_n(x)]
        return f_ground

    psi_n_minus_1, _ = eigenfunctions(n - 1)
    sqrt_2n = np.sqrt(2 * n)

    def f(x):
        x = np.asarray(x[0])
        return [sqrt_2n * psi_n_minus_1(x) - x * psi_n(x)]
    return f

# %%


def test_samples():
    x = [np.linspace(0, 10, 100)]
    N = 100000

    n = 3
    f, _ = eigenfunctions(n)
    f2x = analytical_first_derivative(n)
    samples_MH = metropolis_hastings(lambda x: f(x)**2, 'gaussian', [0.], xmin=[0.], xmax=[10.], N=N, kwrgs={
        'sigma': 1.}, detail=True)

    # samples = MALA(f=f, f_prime=f2x, x_0=[1.], timestep=0.0001, xmin=[
    #                0.], xmax=[10.], N=N, detail=True)

    samples_MALA = stochasticMALA(f=f, f_prime=f2x, x_0=[1.], timestep=0.001, xmin=[
        0.], xmax=[10.], N=N, p_kick=0.5, kick_sigma=0.5, detail=True)

    #  discard first 10% of samples as burn-in
    burn_in = N//10
    samples_MALA = samples_MALA[burn_in:]
    samples_MH = samples_MH[burn_in:]

    acf_mala = get_acf(samples_MALA)
    acf_mh = get_acf(samples_MH)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].hist(samples_MH, bins=100, density=True,
               alpha=1, label='Sampled Distribution')
    ax[1].hist(samples_MALA, bins=100, density=True,
               alpha=1, label='Sampled Distribution')

    pdf = f(x)**2
    pdf /= np.trapezoid(pdf, x)
    ax[0].plot(x, pdf, label='Target Distribution', color='red')
    ax[1].plot(x, pdf, label='Target Distribution', color='red')
    ax[0].set_title(f"Metropolis Hastings, nsamples: {N}")
    ax[1].set_title(f"Stochastic Metropolis-adjusted Langevin, nsamples: {N}")
    ax[0].legend()
    ax[1].legend()
    plt.ylabel("$\psi (x)$")
    plt.xlabel("x")
    plt.show()

    #  autocorrelation
    plt.figure(figsize=(10, 5))
    plt.plot(acf_mh, label='Metropolis-Hastings', color='C0')
    plt.plot(acf_mala, label='Stochastic MALA', color='C1')

    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axhline(1/np.e, color='gray', lw=0.5, ls=':',
                label='Correlation Time ($1/e$)')

    plt.xlabel('Lag (k steps)')
    plt.ylabel('Autocorrelation $C(k)$')
    plt.title('Sampler Efficiency Comparison (Steep Drop = Better)')
    plt.legend()
    plt.grid(True)

    plt.show()


def test_diffrentiators():
    import matplotlib.pyplot as plt
    min_errors = []

    N = 5  # quantum numbers to test
    for n in range(1, N):
        f, _ = eigenfunctions(n)

        x = [np.linspace(-10, 10, 200)]
        d2f = analytical_derivative(n)
        h = np.linspace(1e-5, 10, 500)

        # mean_error_d2bc = []
        # mean_error_d2fw = []
        mean_error_d2c_h2 = []
        mean_error_d2c_h4 = []
        mean_error_d2c_h6 = []
        mean_error_d2c_h8 = []
        mean_error_d2c_h10 = []

        for step in h:
            step = np.array([step], dtype=float)
            step = [step]
            d2c_h2 = differentiators.double_central_difference(
                f, x, h=step, order=2)
            error_d2c_h2 = (np.abs(d2c_h2 - d2f(x)))
            mean_error_d2c_h2.append(rms(error_d2c_h2))

            d2c_h4 = differentiators.double_central_difference(
                f, x, h=step, order=4)
            error_d2c_h4 = (np.abs(d2c_h4 - d2f(x)))
            mean_error_d2c_h4.append(rms(error_d2c_h4))

            d2c_h6 = differentiators.double_central_difference(
                f, x, h=step, order=6)
            error_d2c_h6 = (np.abs(d2c_h6 - d2f(x)))
            mean_error_d2c_h6.append(rms(error_d2c_h6))

            d2c_h8 = differentiators.double_central_difference(
                f, x, h=step, order=8)
            error_d2c_h8 = (np.abs(d2c_h8 - d2f(x)))
            mean_error_d2c_h8.append(rms(error_d2c_h8))

            d2c_h10 = differentiators.double_central_difference(
                f, x, h=step, order=10)
            error_d2c_h10 = (np.abs(d2c_h10 - d2f(x)))
            mean_error_d2c_h10.append(rms(error_d2c_h10))

            # d2fw = differentiators.double_forward_difference(f, x, h=step)
            # error_d2fw = (np.abs(d2fw - d2f(x)))
            # mean_error_d2fw.append(np.mean(error_d2fw))

            # d2bc = differentiators.double_backward_difference(f, x, h=step)
            # error_d2bc = (np.abs(d2bc - d2f(x)))
            # mean_error_d2bc.append(np.mean(error_d2bc))

            # methods = [
            #     ("Double Central Difference O(h^2)", d2c_h2, error_d2c_h2),
            #     ("Double Central Difference O(h^4)", d2c_h4, error_d2c_h4),
            #     ("Double Forward Difference O(h^2)", d2fw, error_d2fw),
            #     ("Double Backward Difference O(h^2)", d2bc, error_d2bc),
            # ]

            # for title, approx, err in methods:
            #     _, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            #     ax[0].plot(x, approx, label=f"{title}, h={step:.1e}")
            #     ax[0].plot(x, d2f(x), linestyle="dashed",
            #                label="Analytical Second Derivative")
            #     ax[0].set_title(title)
            #     ax[0].legend()

            #     ax[1].plot(x, err, label=f"{title} Error, h={step:.1e}")
            #     ax[1].set_yscale("log")
            #     ax[1].set_title(f"{title} Error (log scale)")
            #     ax[1].legend()

            #     print(
            #         f"Mean error for {title} with h={step:.1e}: {np.mean(err):.3e}")

            # plt.show()

        print(f"\nQuantum Number n={n} Differentiator Error Analysis:")

        plt.plot(h, mean_error_d2c_h2,
                 label="Central Difference O($h^2$)")
        plt.plot(h, mean_error_d2c_h4,
                 label="Central Difference O($h^4$)")
        plt.plot(h, mean_error_d2c_h6,
                 label="Central Difference O($h^6$)")
        plt.plot(h, mean_error_d2c_h8,
                 label="Central Difference O($h^8$)")
        plt.plot(h, mean_error_d2c_h10,
                 label="Central Difference O($h^(10)$)")

        # plt.plot(h, mean_error_d2fw,
        #          label="Mean Error Double Forward Difference O(h^2)")
        # plt.plot(h, mean_error_d2bc,
        #          label="Mean Error Double Backward Difference O(h^2)")

        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Step size h")
        plt.ylabel("RMS Absolute Error")
        plt.title(f"RMS Absolute Error vs Step Size for quantum number {n}")
        plt.legend()
        plt.show()

        min_h_index_h2 = np.argmin(mean_error_d2c_h2)
        min_h_index_h4 = np.argmin(mean_error_d2c_h4)
        min_h_index_h6 = np.argmin(mean_error_d2c_h6)
        min_h_index_h8 = np.argmin(mean_error_d2c_h8)
        min_h_index_h10 = np.argmin(mean_error_d2c_h10)

        min_h_h2 = h[min_h_index_h2]
        min_h_h4 = h[min_h_index_h4]
        min_h_h6 = h[min_h_index_h6]
        min_h_h8 = h[min_h_index_h8]
        min_h_h10 = h[min_h_index_h10]
        min_errors.append([mean_error_d2c_h2[min_h_index_h2], mean_error_d2c_h4[min_h_index_h4],
                           mean_error_d2c_h6[min_h_index_h6], mean_error_d2c_h8[min_h_index_h8],
                           mean_error_d2c_h10[min_h_index_h10]])

        print("optimal mean errors: \n"
              f" Double Central Difference O(h^2): {mean_error_d2c_h2[min_h_index_h2]:.3e} at h={min_h_h2:.3e}\n"
              f" Double Central Difference O(h^4): {mean_error_d2c_h4[min_h_index_h4]:.3e} at h={min_h_h4:.3e}\n"
              f" Double Central Difference O(h^6): {mean_error_d2c_h6[min_h_index_h6]:.3e} at h={min_h_h6:.3e}\n"
              f" Double Central Difference O(h^8): {mean_error_d2c_h8[min_h_index_h8]:.3e} at h={min_h_h8:.3e}\n"
              f" Double Central Difference O(h^10): {mean_error_d2c_h10[min_h_index_h10]:.3e} at h={min_h_h10:.3e}\n"
              )

    for i in range(5):
        plt.plot(range(1, N), [min_errors[n-1][i] for n in range(1, N)],
                 label=f"Order O(h^{2*(i+1)})")
    plt.yscale("log")
    plt.xlabel("Quantum Number n")
    plt.ylabel("Optimal Step Size h")
    plt.title("Optimal Step Size vs Quantum Number for Different Orders")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_samples()
    test_diffrentiators()
