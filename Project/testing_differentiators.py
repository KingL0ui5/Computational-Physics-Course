import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from modules.helpers import harmonic_oscillator_helpers as hlp, rms
import modules.differentiators as differentiators
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")

size = 13

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size-2)
plt.rc('figure', titlesize=size)


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

    ax.quiver(x[:, 0], x[:, 1], x[:, 2], d2x_ana[:, 0], d2x_ana[:, 1], d2x_ana[:, 2],
              length=0.4, normalize=True, color='green', label='Analytic (Exact)', linewidth=2.5, arrow_length_ratio=0.3)

    ax.quiver(x[:, 0], x[:, 1], x[:, 2], d2x_num[:, 0], d2x_num[:, 1], d2x_num[:, 2],
              length=0.4, normalize=True, color='red', label='Numerical (Calculated)', linewidth=1.0, linestyle='dashed', arrow_length_ratio=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Calculated vs Analytic Second Derivatives')
    ax.legend(loc='upper left')(loc='upper left')()
    return fig


def test_second_order_diffrentiators(ax):
    min_errors = []
    ns = [5]

    for n in ns:
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
            mean_error_d2c_h2.append(rms(np.abs(d2c_h2 - d2f(x))))

            d2c_h4 = differentiators.double_central_difference(
                f, x, h=step, order=4).flatten()
            mean_error_d2c_h4.append(rms(np.abs(d2c_h4 - d2f(x))))

            d2c_h6 = differentiators.double_central_difference(
                f, x, h=step, order=6).flatten()
            mean_error_d2c_h6.append(rms(np.abs(d2c_h6 - d2f(x))))

            d2c_h8 = differentiators.double_central_difference(
                f, x, h=step, order=8).flatten()
            mean_error_d2c_h8.append(rms(np.abs(d2c_h8 - d2f(x))))

            d2c_h10 = differentiators.double_central_difference(
                f, x, h=step, order=10).flatten()
            mean_error_d2c_h10.append(rms(np.abs(d2c_h10 - d2f(x))))

            d2fw = differentiators.double_forward_difference(
                f, x, h=step).flatten()
            mean_error_d2fw.append(rms(np.abs(d2fw - d2f(x))))

            d2bc = differentiators.double_backward_difference(
                f, x, h=step).flatten()
            mean_error_d2bc.append(rms(np.abs(d2bc - d2f(x))))

        ax.plot(h, mean_error_d2c_h2, label="Central O($h^2$)")
        ax.plot(h, mean_error_d2c_h4, label="Central O($h^4$)")
        ax.plot(h, mean_error_d2c_h6, label="Central O($h^6$)")
        ax.plot(h, mean_error_d2c_h8, label="Central O($h^8$)")
        ax.plot(h, mean_error_d2c_h10, label="Central O($h^{10}$)")
        ax.plot(h, mean_error_d2fw, label="Fwd O($h^2$)")
        ax.plot(h, mean_error_d2bc, label="Bwd O($h^2$)")

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="--")
        ax.set_xlabel("Step size h")
        ax.set_ylabel("RMS Absolute Error")
        ax.set_title(f"Figure 1.2: Second Derivative Error (QHO n={n})")
        ax.legend(loc='upper left')

        min_h_index_h2 = np.argmin(mean_error_d2c_h2)
        min_h_index_h4 = np.argmin(mean_error_d2c_h4)
        min_h_index_h6 = np.argmin(mean_error_d2c_h6)
        min_h_index_h8 = np.argmin(mean_error_d2c_h8)
        min_h_index_h10 = np.argmin(mean_error_d2c_h10)
        min_h_index_d2fw = np.argmin(mean_error_d2fw)
        min_h_index_d2bc = np.argmin(mean_error_d2bc)

        print(f"\nQuantum Number n={n} Differentiator Error Analysis:")
        print("optimal mean errors: \n"
              f" Double Central Difference O(h^2): {mean_error_d2c_h2[min_h_index_h2]:.3e} at h={h[min_h_index_h2]:.3e}\n"
              f" Double Central Difference O(h^4): {mean_error_d2c_h4[min_h_index_h4]:.3e} at h={h[min_h_index_h4]:.3e}\n"
              f" Double Central Difference O(h^6): {mean_error_d2c_h6[min_h_index_h6]:.3e} at h={h[min_h_index_h6]:.3e}\n"
              f" Double Central Difference O(h^8): {mean_error_d2c_h8[min_h_index_h8]:.3e} at h={h[min_h_index_h8]:.3e}\n"
              f" Double Central Difference O(h^10): {mean_error_d2c_h10[min_h_index_h10]:.3e} at h={h[min_h_index_h10]:.3e}\n"
              f" Double Forward Difference O(h^2): {mean_error_d2fw[min_h_index_d2fw]:.3e} at h={h[min_h_index_d2fw]:.3e}\n"
              f" Double Backward Difference O(h^2): {mean_error_d2bc[min_h_index_d2bc]:.3e} at h={h[min_h_index_d2bc]:.3e}\n"
              )


def test_hydrogen_laplacian(ax):
    theta = 1.

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

    h_vals = np.logspace(-7, -3, 500)
    orders = [2, 4, 6, 8]
    errors = {order: [] for order in orders}

    for h in h_vals:
        h_vec = [h, h, h]
        for order in orders:
            d2psi_num = differentiators.double_central_difference(
                psi_hydrogen, coords, h=h_vec, order=order)
            lap_num = np.sum(d2psi_num, axis=1)
            error_val = rms(np.abs(lap_num - lap_analytic))
            errors[order].append(error_val)

    for order in orders:
        ax.plot(h_vals, errors[order], label=f"Central O($h^{{{order}}}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step size $h$")
    ax.set_ylabel("RMS Absolute Error")
    ax.set_title("Figure 1.3: Hydrogen Laplacian Error vs Stepsize")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')

    print("\nHydrogen Laplacian Optimal step sizes:")
    for order in orders:
        errs = np.array(errors[order])
        min_idx = np.argmin(errs)
        print(
            f"  Order {order}: Min Error = {errs[min_idx]:.3e} at h = {h_vals[min_idx]:.3e}")


def test_h2_parameter_gradient_error_analysis(ax):
    q1 = np.array([0.0, 0.0, -0.7])
    q2 = np.array([0.0, 0.0, 0.7])
    r1 = np.array([[0.5, 0.0, 0.0]])
    r2 = np.array([[0.0, 0.5, 0.0]])

    def compute_distances(r1, r2, q1, q2):
        r1A = np.linalg.norm(r1 - q1, axis=1)
        r1B = np.linalg.norm(r1 - q2, axis=1)
        r2A = np.linalg.norm(r2 - q1, axis=1)
        r2B = np.linalg.norm(r2 - q2, axis=1)
        r12 = np.linalg.norm(r1 - r2, axis=1)
        return r1A, r1B, r2A, r2B, r12

    def h2_log_psi(thetas):
        theta1, theta2, theta3 = thetas[:, 0], thetas[:, 1], thetas[:, 2]
        r1A, r1B, r2A, r2B, r12 = compute_distances(r1, r2, q1, q2)
        dist_sum_1 = r1A + r2B
        dist_sum_2 = r1B + r2A
        phi = np.exp(-theta1 * dist_sum_1) + np.exp(-theta1 * dist_sum_2)
        log_jastrow = - (theta2 * r12) / (1 + theta3 * r12)
        return np.log(phi) + log_jastrow

    def h2_analytic_log_grad(thetas):
        theta1, theta2, theta3 = thetas[:, 0], thetas[:, 1], thetas[:, 2]
        r1A, r1B, r2A, r2B, r12 = compute_distances(r1, r2, q1, q2)
        E1 = np.exp(-theta1 * (r1A + r2B))
        E2 = np.exp(-theta1 * (r1B + r2A))
        Phi = E1 + E2
        grad_th1 = -((r1A + r2B) * E1 + (r1B + r2A) * E2) / Phi
        denom = 1 + theta3 * r12
        grad_th2 = -r12 / denom
        grad_th3 = (theta2 * r12**2) / (denom**2)
        return np.column_stack((grad_th1, grad_th2, grad_th3))

    points = np.array([[1.0, 1.0, 1.0]])
    grad_analytic = h2_analytic_log_grad(points)
    h_vals = np.logspace(-7, -1, 500)
    orders = [2, 4, 6, 8]
    errors = {order: [] for order in orders}

    for h in h_vals:
        h_vec = [h, h, h]
        for order in orders:
            grad_num = differentiators.central_difference(
                h2_log_psi, points, h=h_vec, order=order)
            error_val = rms(np.abs(grad_num - grad_analytic))
            errors[order].append(error_val)

    for order in orders:
        ax.plot(h_vals, errors[order], label=f"Central O($h^{{{order}}}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step size $h$")
    ax.set_ylabel("RMS Absolute Error")
    ax.set_title("Figure 1.1: H$_2$ First Derivative Error vs Stepsize")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')

    print("\nH$_2$ Parameter First Derivative Test")
    for order in orders:
        errs = np.array(errors[order])
        min_idx = np.argmin(errs)
        print(
            f"  Order {order}: Min Error = {errs[min_idx]:.3e} at h = {h_vals[min_idx]:.3e}")


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 50})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    test_h2_parameter_gradient_error_analysis(axes[0])
    test_second_order_diffrentiators(axes[1])
    test_hydrogen_laplacian(axes[2])

    plt.tight_layout()
    plt.savefig("derivatives_tested.png", dpi=300, bbox_inches='tight')
    plt.show()

    # test_diffrentiators_3d().show()
