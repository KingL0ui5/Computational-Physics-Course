import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from modules.helpers import harmonic_oscillator_helpers as hlp, rms
import modules.differentiators as differentiators
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


def test_first_order_diffrentiators():
    import matplotlib.pyplot as plt
    min_errors = []

    N = 12  # quantum numbers to test
    for n in range(1, N):
        f, _ = hlp.eigenfunctions(n)

        x = np.linspace(-10, 10, 200)
        df = hlp.first_derivative(n=n)
        h = np.linspace(1e-10, 1e-1, 1000)

        mean_error_dc_h2 = []
        mean_error_dc_h4 = []
        mean_error_dc_h6 = []
        mean_error_dc_h8 = []
        mean_error_dc_h10 = []

        for step in h:
            step = np.array([step], dtype=float)
            dc_h2 = differentiators.central_difference(
                f, x, h=step, order=2).flatten()
            error_dc_h2 = (np.abs(dc_h2 - df(x)))
            mean_error_dc_h2.append(rms(error_dc_h2))

            dc_h4 = differentiators.central_difference(
                f, x, h=step, order=4).flatten()
            error_dc_h4 = (np.abs(dc_h4 - df(x)))
            mean_error_dc_h4.append(rms(error_dc_h4))

            dc_h6 = differentiators.central_difference(
                f, x, h=step, order=6).flatten()
            error_dc_h6 = (np.abs(dc_h6 - df(x)))
            mean_error_dc_h6.append(rms(error_dc_h6))

            dc_h8 = differentiators.central_difference(
                f, x, h=step, order=8).flatten()
            error_dc_h8 = (np.abs(dc_h8 - df(x)))
            mean_error_dc_h8.append(rms(error_dc_h8))

            dc_h10 = differentiators.central_difference(
                f, x, h=step, order=10).flatten()
            error_dc_h10 = (np.abs(dc_h10 - df(x)))
            mean_error_dc_h10.append(rms(error_dc_h10))

            # methods = [
            #     ("Central Difference O(h^2)", dc_h2, error_dc_h2),
            #     ("Central Difference O(h^4)", dc_h4, error_dc_h4),
            #     ("Central Difference O(h^6)", dc_h6, error_dc_h6),
            #     ("Central Difference O(h^8)", dc_h8, error_dc_h8),
            #     ("Central Difference O(h^10)", dc_h10, error_dc_h10),
            # ]

            # for title, approx, err in methods:
            #     _, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            #     ax[0].plot(x, approx, label=f"{title}, h={step[0]:.1e}")
            #     ax[0].plot(x, df(x), linestyle="dashed",
            #                label="Analytical Derivative")
            #     ax[0].set_title(title)
            #     ax[0].legend()

            #     ax[1].plot(x, err, label=f"{title} Error, h={step[0]:.1e}")
            #     ax[1].set_yscale("log")
            #     ax[1].set_title(f"{title} Error (log scale)")
            #     ax[1].legend()

            # print(
            #     f"Mean error for {title} with h={step:.1e}: {np.mean(err):.3e}")

            # plt.show()

        print(f"\nQuantum Number n={n} Differentiator Error Analysis:")

        plt.plot(h, mean_error_dc_h2,
                 label="Central Difference O($h^2$)")
        plt.plot(h, mean_error_dc_h4,
                 label="Central Difference O($h^4$)")
        plt.plot(h, mean_error_dc_h6,
                 label="Central Difference O($h^6$)")
        plt.plot(h, mean_error_dc_h8,
                 label="Central Difference O($h^8$)")
        plt.plot(h, mean_error_dc_h10,
                 label="Central Difference O($h^{10}$)")

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

        min_h_index_h2 = np.argmin(mean_error_dc_h2)
        min_h_index_h4 = np.argmin(mean_error_dc_h4)
        min_h_index_h6 = np.argmin(mean_error_dc_h6)
        min_h_index_h8 = np.argmin(mean_error_dc_h8)
        min_h_index_h10 = np.argmin(mean_error_dc_h10)

        min_h_h2 = h[min_h_index_h2]
        min_h_h4 = h[min_h_index_h4]
        min_h_h6 = h[min_h_index_h6]
        min_h_h8 = h[min_h_index_h8]
        min_h_h10 = h[min_h_index_h10]
        min_errors.append([mean_error_dc_h2[min_h_index_h2], mean_error_dc_h4[min_h_index_h4],
                           mean_error_dc_h6[min_h_index_h6], mean_error_dc_h8[min_h_index_h8],
                           mean_error_dc_h10[min_h_index_h10]])

        print("optimal mean errors: \n"
              f" Central Difference O(h^2): {mean_error_dc_h2[min_h_index_h2]:.3e} at h={min_h_h2:.3e}\n"
              f" Central Difference O(h^4): {mean_error_dc_h4[min_h_index_h4]:.3e} at h={min_h_h4:.3e}\n"
              f" Central Difference O(h^6): {mean_error_dc_h6[min_h_index_h6]:.3e} at h={min_h_h6:.3e}\n"
              f" Central Difference O(h^8): {mean_error_dc_h8[min_h_index_h8]:.3e} at h={min_h_h8:.3e}\n"
              f" Central Difference O(h^10): {mean_error_dc_h10[min_h_index_h10]:.3e} at h={min_h_h10:.3e}\n"
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


def test_second_order_diffrentiators():
    import matplotlib.pyplot as plt
    min_errors = []

    N = 5  # quantum numbers to test
    for n in range(1, N):
        f, _ = hlp.eigenfunctions(n)

        x = np.linspace(-10, 10, 200)
        d2f = hlp.second_derivative(n=n)
        h = np.linspace(1e-5, 1, 500)

        mean_error_d2c_h2 = []
        mean_error_d2c_h4 = []
        mean_error_d2c_h6 = []
        mean_error_d2c_h8 = []
        mean_error_d2c_h10 = []
        mean_error_d2fw = []
        mean_error_d2bc = []

        for step in h:
            step = np.array([step], dtype=float)
            d2c_h2 = differentiators.double_central_difference(
                f, x, h=step, order=2).flatten()
            error_d2c_h2 = (np.abs(d2c_h2 - d2f(x)))
            mean_error_d2c_h2.append(rms(error_d2c_h2))

            d2c_h4 = differentiators.double_central_difference(
                f, x, h=step, order=4).flatten()
            error_d2c_h4 = (np.abs(d2c_h4 - d2f(x)))
            mean_error_d2c_h4.append(rms(error_d2c_h4))

            d2c_h6 = differentiators.double_central_difference(
                f, x, h=step, order=6).flatten()
            error_d2c_h6 = (np.abs(d2c_h6 - d2f(x)))
            mean_error_d2c_h6.append(rms(error_d2c_h6))

            d2c_h8 = differentiators.double_central_difference(
                f, x, h=step, order=8).flatten()
            error_d2c_h8 = (np.abs(d2c_h8 - d2f(x)))
            mean_error_d2c_h8.append(rms(error_d2c_h8))

            d2c_h10 = differentiators.double_central_difference(
                f, x, h=step, order=10).flatten()
            error_d2c_h10 = (np.abs(d2c_h10 - d2f(x)))
            mean_error_d2c_h10.append(rms(error_d2c_h10))

            d2fw = differentiators.double_forward_difference(
                f, x, h=step).flatten()
            error_d2fw = (np.abs(d2fw - d2f(x)))
            mean_error_d2fw.append(rms(error_d2fw))

            d2bc = differentiators.double_backward_difference(
                f, x, h=step).flatten()
            error_d2bc = (np.abs(d2bc - d2f(x)))
            mean_error_d2bc.append(rms(error_d2bc))

            # methods = [
            #     ("Double Central Difference O(h^2)", d2c_h2, error_d2c_h2),
            #     ("Double Central Difference O(h^4)", d2c_h4, error_d2c_h4),
            #     ("Double Central Difference O(h^6)", d2c_h6, error_d2c_h6),
            #     ("Double Central Difference O(h^8)", d2c_h8, error_d2c_h8),
            #     ("Double Central Difference O(h^10)", d2c_h10, error_d2c_h10),
            # ]

            # for title, approx, err in methods:
            #     _, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            #     ax[0].plot(x, approx, label=f"{title}, h={step[0]:.1e}")
            #     ax[0].plot(x, d2f(x), linestyle="dashed",
            #                label="Analytical Second Derivative")
            #     ax[0].set_title(title)
            #     ax[0].legend()

            #     ax[1].plot(x, err, label=f"{title} Error, h={step[0]:.1e}")
            #     ax[1].set_yscale("log")
            #     ax[1].set_title(f"{title} Error (log scale)")
            #     ax[1].legend()

            # print(
            #     f"Mean error for {title} with h={step:.1e}: {np.mean(err):.3e}")

            plt.show()

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
                 label="Central Difference O($h^{10}$)")
        plt.plot(h, mean_error_d2fw,
                 label="Mean Error Double Forward Difference O(h^2)")
        plt.plot(h, mean_error_d2bc,
                 label="Mean Error Double Backward Difference O(h^2)")

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
        min_h_index_d2fw = np.argmin(mean_error_d2fw)
        min_h_index_d2bc = np.argmin(mean_error_d2bc)

        min_h_h2 = h[min_h_index_h2]
        min_h_h4 = h[min_h_index_h4]
        min_h_h6 = h[min_h_index_h6]
        min_h_h8 = h[min_h_index_h8]
        min_h_h10 = h[min_h_index_h10]
        min_h_d2fw = h[min_h_index_d2fw]
        min_h_d2bc = h[min_h_index_d2bc]
        min_errors.append([mean_error_d2c_h2[min_h_index_h2], mean_error_d2c_h4[min_h_index_h4],
                           mean_error_d2c_h6[min_h_index_h6], mean_error_d2c_h8[min_h_index_h8],
                           mean_error_d2c_h10[min_h_index_h10], mean_error_d2fw[min_h_index_d2fw],
                           mean_error_d2bc[min_h_index_d2bc]])

        print("optimal mean errors: \n"
              f" Double Central Difference O(h^2): {mean_error_d2c_h2[min_h_index_h2]:.3e} at h={min_h_h2:.3e}\n"
              f" Double Central Difference O(h^4): {mean_error_d2c_h4[min_h_index_h4]:.3e} at h={min_h_h4:.3e}\n"
              f" Double Central Difference O(h^6): {mean_error_d2c_h6[min_h_index_h6]:.3e} at h={min_h_h6:.3e}\n"
              f" Double Central Difference O(h^8): {mean_error_d2c_h8[min_h_index_h8]:.3e} at h={min_h_h8:.3e}\n"
              f" Double Central Difference O(h^10): {mean_error_d2c_h10[min_h_index_h10]:.3e} at h={min_h_h10:.3e}\n"
              f" Double Forward Difference O(h^2): {mean_error_d2fw[min_h_index_d2fw]:.3e} at h={min_h_d2fw:.3e}\n"
              f" Double Backward Difference O(h^2): {mean_error_d2bc[min_h_index_d2bc]:.3e} at h={min_h_d2bc:.3e}\n"
              )

    for i in range(7):
        plt.plot(range(1, N), [min_errors[n-1][i] for n in range(1, N)],
                 label=f"Order O(h^{2*(i+1)})")
    plt.yscale("log")
    plt.xlabel("Quantum Number n")
    plt.ylabel("Optimal Step Size h")
    plt.title("Optimal Step Size vs Quantum Number for Different Orders")
    plt.legend()
    plt.show()


def test_diffrentiators_3d():
    def f(coords):
        if coords.ndim == 1:
            x, y, z = coords[0], coords[1], coords[2]
            return x**2 + 2*y**3 + z**2
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return x**2 + 2*y**3 + z**2

    def analytic_second_derivative(coords):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        N = len(x)
        res = np.zeros((N, 3))
        res[:, 0] = 2.0
        res[:, 1] = 12.0 * y
        res[:, 2] = 2.0
        return res

    t = np.linspace(-2, 2, 40)
    x = np.column_stack([t, t, t])

    d2x_num = differentiators.double_central_difference(
        f, x, h=[0.001, 0.001, 0.001], order=8)

    d2x_ana = analytic_second_derivative(x)

    rmse = np.sqrt(np.mean((d2x_num - d2x_ana)**2))
    print(f"RMS Error compared to Analytic: {rmse:.5e}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='gray',
            alpha=0.3, label='Trajectory')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], color='blue', s=10)

    ax.quiver(x[:, 0], x[:, 1], x[:, 2],
              d2x_ana[:, 0], d2x_ana[:, 1], d2x_ana[:, 2],
              length=0.4, normalize=True, color='green',
              label='Analytic (Exact)', linewidth=2.5, arrow_length_ratio=0.3)

    ax.quiver(x[:, 0], x[:, 1], x[:, 2],
              d2x_num[:, 0], d2x_num[:, 1], d2x_num[:, 2],
              length=0.4, normalize=True, color='red',
              label='Numerical (Calculated)', linewidth=1.0, linestyle='dashed', arrow_length_ratio=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Calculated vs Analytic Second Derivatives')
    ax.legend()
    plt.show()


def test_hydrogen_laplacian():
    theta = 1.

    def psi_hydrogen(coords):
        r = np.linalg.norm(coords, axis=1)
        return np.exp(-theta * r)

    def analytic_laplacian(coords):
        r = np.linalg.norm(coords, axis=1)
        laplacian = (theta**2 - 2*theta/r) * np.exp(-theta * r)
        return laplacian

    def psi_hydrogen(coords):
        r = np.linalg.norm(coords, axis=1)
        return np.exp(-theta * r)

    def analytic_laplacian(coords):
        r = np.linalg.norm(coords, axis=1)
        laplacian = (theta**2 - 2 * theta / r) * np.exp(-theta * r)
        return laplacian

    r_vals = np.logspace(-3, 1, 200)
    coords = np.zeros((len(r_vals), 3))
    coords[:, 0] = r_vals

    lap_analytic = analytic_laplacian(coords)

    h_vals = np.logspace(-7, -1.5, 500)

    orders = [2, 4, 6, 8, 10]
    errors = {order: [] for order in orders}

    for h in h_vals:
        h_vec = [h, h, h]

        for order in orders:
            d2psi_num = differentiators.double_central_difference(
                psi_hydrogen, coords, h=h_vec, order=order
            )
            lap_num = np.sum(d2psi_num, axis=1)

            error_val = rms(np.abs(lap_num - lap_analytic))
            errors[order].append(error_val)

    plt.figure(figsize=(10, 7))

    for order in orders:
        plt.plot(h_vals, errors[order],
                 label=f"Central Difference O($h^{{{order}}}$)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Step size $h$")
    plt.ylabel("RMS Absolute Error in Laplacian")
    plt.title("Hydrogen Laplacian Error vs Step Size")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Optimal step sizes and minimum errors:")
    for order in orders:
        errs = np.array(errors[order])
        min_idx = np.argmin(errs)
        min_h = h_vals[min_idx]
        min_err = errs[min_idx]
        print(f"  Order {order}: Min Error = {min_err:.3e} at h = {min_h:.3e}")


if __name__ == "__main__":
    # test_first_order_diffrentiators()
    # test_second_order_diffrentiators()
    # test_diffrentiators_3d()
    test_hydrogen_laplacian()
