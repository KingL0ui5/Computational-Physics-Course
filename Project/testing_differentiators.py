import numpy as np
import seaborn as sns
from modules.helpers import rms, harmonic_oscillator
import modules.differentiators as differentiators
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


def test_diffrentiators():
    import matplotlib.pyplot as plt
    min_errors = []

    N = 5  # quantum numbers to test
    for n in range(1, N):
        f, _ = harmonic_oscillator.eigenfunctions(n)

        x = np.linspace(-10, 10, 200)
        d2f = harmonic_oscillator.second_derivative(n=n)
        h = np.linspace(1e-5, 1, 500)

        mean_error_d2c_h2 = []
        mean_error_d2c_h4 = []
        mean_error_d2c_h6 = []
        mean_error_d2c_h8 = []
        mean_error_d2c_h10 = []

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


def test_diffrentiators_3d():
    def f(coords):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return x**2 + y**2 + z**2

    def analytic(coords):
        return 2 * coords

    t = np.linspace(-10, 10, 100)
    x = np.column_stack([t, t, t])

    d2x = differentiators.double_central_difference(
        f, x, h=[0.001, 0.001, 0.001], order=8)
    analytic_val = 2.0
    rmse = np.sqrt(np.mean((d2x - analytic_val)**2))
    print(f"RMS Error compared to analytic (2.0): {rmse:.5e}")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='blue',
            alpha=0.5, label='Trajectory')

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], color='blue', s=20)

    ax.quiver(x[:, 0], x[:, 1], x[:, 2],
              d2x[:, 0], d2x[:, 1], d2x[:, 2],
              length=2.0, normalize=True, color='red', label='Calculated $\\nabla^2$ Vector', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        '3D Visualisation of Numerical Second Derivative\nFunction: $f(x,y,z) = x^2 + y^2 + z^2$')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_diffrentiators()
    # test_diffrentiators_3d()
