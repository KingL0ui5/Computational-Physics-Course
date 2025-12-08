import numpy as np
import seaborn as sns
from modules.function_sampling import metropolis_hastings, MALA, stochasticMALA
from modules.helpers import harmonic_oscillator_helpers as hlp, get_acf
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


def test_samples_H2():
    from hydrogen_molecule import h2_wavefunction

    def autocorrelation(x):
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        result = r / (variance * (np.arange(n, 0, -1)))
        return result

    q1 = np.array([0, 0, 1])
    q2 = np.array([0, 0, -1])
    theta_A = np.array([1., 1., 1.])
    wf_A = h2_wavefunction(thetas=theta_A, q1=q1, q2=q2)

    theta_B = np.array([1., 1., 1.])
    wf_B = h2_wavefunction(thetas=theta_B, q1=q1, q2=q2)

    N_samples = 100000

    # Define bounds and start
    r_0 = [1.] * 6

    def wavefunction_wrapper_A(coords):
        coords = np.asarray(coords)
        r1, r2 = coords[:, 0:3], coords[:, 3:6]
        return wf_A.psi(r1, r2)

    def wavefunction_wrapper_B(coords):
        coords = np.asarray(coords)
        r1, r2 = coords[:, 0:3], coords[:, 3:6]
        return wf_B.probability_density(r1, r2)

    # samples_A = metropolis_hastings(f=wavefunction_wrapper_A, f_prop='gaussian', x_0=r_0, xmin=[
    #     -10.]*6, xmax=[10.]*6, N=N_samples, kwrgs={'sigma': 0.8}, thinning=20)

    def f_prime_A(coords):
        from modules.differentiators import central_difference
        coords = np.asarray(coords)
        grad = central_difference(
            wavefunction_wrapper_A, coords, h=[1e-4]*6, order=2)
        return np.sum(grad, axis=0)

    samples_A = MALA(f=wavefunction_wrapper_A, f_prime=f_prime_A, x_0=r_0, timestep=0.5, xmin=[
        -10.]*6, xmax=[10.]*6, N=N_samples, detail=True)

    samples_B = metropolis_hastings(f=wavefunction_wrapper_B, f_prop='gaussian', x_0=r_0, xmin=[
        -10.]*6, xmax=[10.]*6, N=N_samples, kwrgs={'sigma': 0.8}, detail=True, thinning=20)

    A_acf = get_acf(samples_A)
    B_acf = get_acf(samples_B)

    burn_in = int(min(len(samples_B), len(samples_A)) * 0.1)

    X_A = samples_A[burn_in:, 0]
    X_B = samples_B[burn_in:, 0]

    r1_A, r2_A = samples_A[burn_in:, 0:3], samples_A[burn_in:, 3:6]
    r1_B, r2_B = samples_B[burn_in:, 0:3], samples_B[burn_in:, 3:6]

    E_A = wf_A.local_energy(r1_A, r2_A)
    E_B = wf_B.local_energy(r1_B, r2_B)

    print(f"Energy Stats (Filtered |E| < 500 Ha):")
    print(f"Mean E_A: {np.mean(E_A):.5f} Ha")
    print(f"Mean E_B: {np.mean(E_B):.5f} Ha")
    print(f"Difference: {abs(np.mean(E_A) - np.mean(E_B)):.5f} Ha")
    print(f"Variance A: {np.var(E_A):.5f}")

    _, ax = plt.subplots(3, 1, figsize=(10, 12))

    ax[0].plot(X_A[::100], label='Chain A', alpha=0.7, linewidth=0.5)
    ax[0].plot(X_B[::100], label='Chain B', alpha=0.7, linewidth=0.5)
    ax[0].set_title(
        f"Coordinate Trace (Electron 1, x-axis) - Subsampled 1:100")
    ax[0].set_ylabel("Position (Bohr)")
    ax[0].legend()

    ax[1].hist(X_A, bins=100, density=True, alpha=0.5,
               label='Chain A', color='blue')
    ax[1].hist(X_B, bins=100, density=True, alpha=0.5,
               label='Chain B', color='orange')
    ax[1].set_title("Coordinate Distribution Histogram")
    ax[1].set_xlabel("Position (x)")
    ax[1].legend()

    lag_max = 1000
    ac_A = autocorrelation(X_A)[:lag_max]
    ac_B = autocorrelation(X_B)[:lag_max]

    ESS_A = len(X_A) / (1 + 2 * np.sum(ac_A[1:]))
    ESS_B = len(X_B) / (1 + 2 * np.sum(ac_B[1:]))

    ax[2].plot(ac_A, color='black', label='Autocorrelation A')
    ax[2].plot(ac_B, color='red', label='Autocorrelation B')
    ax[2].axhline(0, color='gray', linestyle='--')
    ax[2].axhline(1/np.e, color='red', linestyle=':',
                  label='Correlation Time (1/e)')
    ax[2].set_title("Autocorrelation of Position")
    ax[2].set_xlabel("Lag (steps)")
    ax[2].set_ylabel("Correlation")
    ax[2].legend()

    try:
        tau = np.where(ac_A < 1/np.e)[0][0]
        ax[2].text(tau, 0.5, f"  tau ≈ {tau} steps", color='red')
        print(f"\nEstimated Correlation Time (tau): {tau} steps")
        print(f"Effective Sample Size (ESS) Chain A: {ESS_A:.1f}")
        print(f"Effective Sample Size (ESS) Chain B: {ESS_B:.1f}")
    except IndexError:
        print("\nCorrelation time (tau) exceeds lag max limit.")

    plt.tight_layout()
    plt.show()

    plt.plot(A_acf, label='Metropolis-Hastings (A)', color='C0')
    plt.plot(B_acf, label='Metropolis-Hastings (B)', color='C1')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axhline(1/np.e, color='gray', lw=0.5, ls=':',
                label='Correlation Time ($1/e$)')
    plt.xlabel('Lag (k steps)')
    plt.ylabel('Autocorrelation $C(k)$')
    plt.title('Sampler Autocorrelation Comparison')
    plt.legend()
    plt.show()


def test_samples():
    x = np.linspace(0, 10, 100)
    N = 100000

    n = 3
    f, _ = hlp.eigenfunctions(n)
    df = hlp.first_derivative(n)

    samples_MH = metropolis_hastings(lambda x: f(x)**2, 'gaussian', [0.], xmin=[0.], xmax=[10.], N=N, kwrgs={
        'sigma': 1.}, detail=True)

    samples_MALA = MALA(f=f, f_prime=df, x_0=[1.], timestep=0.5, xmin=[
        0.], xmax=[10.], N=N, detail=True)

    samples_sMALA = stochasticMALA(f=f, f_prime=df, x_0=[1.], timestep=0.01, xmin=[
        0.], xmax=[10.], N=N, p_kick=0.1, kick_sigma=0.5, detail=True)

    #  discard first 10% of samples as burn-in
    burn_in = N//10
    samples_MALA = samples_MALA[burn_in:]
    samples_sMALA = samples_sMALA[burn_in:]
    samples_MH = samples_MH[burn_in:]

    acf_mala = get_acf(samples_MALA)
    acf_s_mala = get_acf(samples_sMALA)
    acf_mh = get_acf(samples_MH)

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].hist(samples_MH, bins=100, density=True,
               alpha=1, label='Sampled Distribution')
    ax[1].hist(samples_MALA, bins=100, density=True,
               alpha=1, label='Sampled Distribution')
    ax[2].hist(samples_sMALA, bins=100, density=True,
               alpha=0.5, label='Sampled Distribution')

    pdf = f(x)**2
    pdf /= np.trapezoid(pdf, x)
    ax[0].plot(x, pdf, label='Target Distribution', color='red')
    ax[1].plot(x, pdf, label='Target Distribution', color='red')
    ax[2].plot(x, pdf, label='Target Distribution', color='red')
    ax[0].set_title(f"Metropolis Hastings, nsamples: {N}")
    ax[1].set_title(f"Metropolis-adjusted Langevin, nsamples: {N}")
    ax[2].set_title(f"Stochastic Metropolis-adjusted Langevin, nsamples: {N}")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.ylabel("$\psi (x)$")
    plt.xlabel("x")
    plt.show()

    #  autocorrelation
    plt.figure(figsize=(10, 5))
    plt.plot(acf_mh, label='Metropolis-Hastings', color='C0')
    plt.plot(acf_mala, label='MALA', color='C1')
    plt.plot(acf_s_mala, label='Stochastic MALA', color='C2')

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


def test_samples_hydrogen():
    from hydrogen import hydrogen_wavefunction

    theta = 1.0
    wf = hydrogen_wavefunction(theta)

    N_samples = 100000
    start_pos = np.array([1.0, 1.0, 1.0])
    xmin = np.array([-20., -20., -20.])
    xmax = np.array([20., 20., 20.])

    samples = metropolis_hastings(
        f=wf.probability_density,
        f_prop='gaussian',
        x_0=start_pos,
        xmin=xmin,
        xmax=xmax,
        N=N_samples,
        kwrgs={'sigma': 0.8},
        detail=True
    )

    samples = samples[N_samples//10:]

    r_samples = np.linalg.norm(samples, axis=1)
    plt.figure(figsize=(10, 6))

    r_vals = np.linspace(0, 50, 200)

    coords_line = np.zeros((len(r_vals), 3))
    coords_line[:, 0] = r_vals
    psi_vals = wf.psi(coords_line)
    if hasattr(psi_vals, 'flatten'):
        psi_vals = psi_vals.flatten()

    p_vals_unnormalized = (r_vals**2) * (psi_vals**2)

    integral = np.trapezoid(p_vals_unnormalized, r_vals)
    p_vals_norm = p_vals_unnormalized / integral

    plt.hist(r_samples, bins=100, density=True, alpha=0.5,
             color='orange', label='Sampled Radial Hist')
    plt.plot(r_vals, p_vals_norm, 'k--', linewidth=2,
             label=r'Analytic $P(r) \propto r^2 |\psi_{ansatz}|^2$')

    plt.title("Hydrogen Radial Distribution (1s Orbital)")
    plt.xlabel("Radius $r$ (Bohr radii)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    slice_mask = np.abs(samples[:, 2]) < 0.2
    slice_samples = samples[slice_mask]

    plt.figure(figsize=(8, 8))
    plt.scatter(slice_samples[:, 0], slice_samples[:,
                1], s=2, alpha=0.3, color='blue')
    plt.title(f"Cross-section at z=0 ({len(slice_samples)} points)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    # test_samples()
    # test_sampling_3d()
    # test_samples_hydrogen()
    test_samples_H2()
