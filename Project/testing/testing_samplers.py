import numpy as np
import seaborn as sns
from function_sampling import metropolis_hastings, MALA, stochasticMALA
from helpers import harmonic_oscillator, get_acf
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


def test_samples():
    x = np.linspace(0, 10, 100)
    N = 100000

    n = 3
    f, _ = harmonic_oscillator.eigenfunctions(n)
    df = harmonic_oscillator.harmonic_first_derivative(n)
    samples_MH = metropolis_hastings(lambda x: f(x)**2, 'gaussian', [0.], xmin=[0.], xmax=[10.], N=N, kwrgs={
        'sigma': 1.}, detail=True)

    # samples = MALA(f=f, f_prime=f2x, x_0=[1.], timestep=0.0001, xmin=[
    #                0.], xmax=[10.], N=N, detail=True)

    samples_MALA = stochasticMALA(f=f, f_prime=df, x_0=[1.], timestep=0.001, xmin=[
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


def test_sampling_3d():
    import matplotlib.pyplot as plt

    def psi_3d(coords):
        coords = np.asarray(coords)
        r2 = np.sum(coords**2)
        return np.exp(-0.5 * r2)

    def grad_psi_3d(coords):
        coords = np.asarray(coords)
        return -coords * psi_3d(coords)

    def prob_density_3d(coords):
        coords = np.asarray(coords)
        return psi_3d(coords)**2

    N_samples = 40000
    start_pos = np.array([1.0, 1.0, 1.0])

    xmin = np.array([-6.0, -6.0, -6.0])
    xmax = np.array([6.0, 6.0, 6.0])

    samples_MH = metropolis_hastings(
        f=prob_density_3d,
        f_prop='gaussian',
        x_0=start_pos,
        xmin=xmin,
        xmax=xmax,
        N=N_samples,
        kwrgs={'sigma': 0.5}
    )
    burn = N_samples // 10
    samples_MH = samples_MH[burn:]

    # radial plot
    r_MH = np.linalg.norm(samples_MH, axis=1)

    def radial_analytic(r):
        return r**2 * np.exp(-r**2)

    r_vals = np.linspace(0, 5, 200)
    p_vals = radial_analytic(r_vals)
    p_vals /= np.trapezoid(p_vals, r_vals)  # Normalize theory curve

    bins = np.linspace(0, 5, 80)
    plt.hist(r_MH, bins=bins, density=True, histtype='stepfilled',
             alpha=0.3, color='blue', label='MH Histogram')
    plt.plot(r_vals, p_vals, 'k--', linewidth=2,
             label='Theory ($r^2 e^{-r^2}$)')

    plt.title("Radial Distribution ($P(r)$)")
    plt.xlabel("Radius $r$")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

    # marginal
    def marginal_analytic(x):
        return np.exp(-x**2)

    x_vals = np.linspace(-4, 4, 200)
    mx_vals = marginal_analytic(x_vals)
    mx_vals /= np.trapezoid(mx_vals, x_vals)

    plt.hist(samples_MH[:, 0], bins=60, density=True,
             alpha=0.5, color='blue', label='MH X-coord')
    plt.plot(x_vals, mx_vals, 'k--', label='Theory ($e^{-x^2}$)')

    plt.title("Marginal Distribution (X-axis)")
    plt.xlabel("X Coordinate")
    plt.legend()
    plt.show()

    #  3d plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    subset = 40000

    ax.scatter(
        samples_MH[:subset, 0],
        samples_MH[:subset, 1],
        samples_MH[:subset, 2],
        s=2,
        alpha=0.6,
        c=r_MH[:subset],
        cmap='viridis'
    )

    ax.set_title(f"3D Scatter Plot (MH, First {subset} pts)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    # traceplot
    samples_x = samples_MH[:, 0]
    N = len(samples_x)
    n_subplots = 6

    fig2, axes = plt.subplots(n_subplots, 1, figsize=(10, 12), sharey=True)
    chunk_size = N // n_subplots

    for i in range(n_subplots):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < (n_subplots-1) else N

        axes[i].plot(range(start, end), samples_x[start:end],
                     color='black', linewidth=0.5, alpha=0.6)

        axes[i].set_ylabel('x Value')
        axes[i].text(0.02, 0.9, f'Segment {i+1}: Iterations {start}-{end}',
                     transform=axes[i].transAxes, fontsize=10, fontweight='bold')

    axes[-1].set_xlabel('Sample Index')
    plt.suptitle(
        'Trace of Metropolis-Hastings Sampling (Split View)', y=0.95, fontsize=14)
    # Removing tight_layout() here slightly helps with suptitle overlap manually,
    # but tight_layout works well if y is adjusted.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    test_samples()
    # test_sampling_3d()
