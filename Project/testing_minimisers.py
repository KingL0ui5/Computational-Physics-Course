"""
A module to test minimizer functions
Louis Liu 25/11
"""

from matplotlib import cm
import numpy as np
import seaborn as sns
import modules.minimisers as minimisers
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


def test_minimiser():
    x_start = np.array([5.0])
    stepsize = 0.1
    max_iterations = 100

    def testf(x_arr):
        x = x_arr[0]
        return (x - 4)**2

    def testdf(x_arr):
        x = x_arr[0]
        return np.array([2 * (x - 4)])

    x_GD, y_GD = minimisers.gradient_desecent(
        testf, testdf, x_start, stepsize, max_iter=max_iterations, detail=True)
    print(x_GD)
    final_x_GD = x_GD[0]

    # Stochastic Gradient Descent
    x_SGD, y_SGD = minimisers.stochastic_GD(
        testf, testdf, x_start, stepsize, noise=0.3, max_iter=max_iterations, detail=True)
    final_x_SGD = x_SGD[0]

    # RMSProp
    x_RMS, y_RMS = minimisers.RMSProp_GD(
        testf, testdf, x_start, stepsize, forgetting=0.3, max_iter=max_iterations, detail=True)
    final_x_RMS = x_RMS[0]

    # Quasi-Newton
    x_QN, y_QN = minimisers.quasi_newton(
        testf, testdf, x_start, stepsize, method="BFGS", max_iter=max_iterations, detail=True)
    final_x_QN = x_QN[0]

    target_x = 4.0
    atol = 1

    assert np.isclose(final_x_GD, target_x,
                      atol=atol), f"GD failed: {final_x_GD}"
    assert np.isclose(final_x_SGD, target_x,
                      atol=0.5), f"SGD failed: {final_x_SGD}"
    assert np.isclose(final_x_RMS, target_x,
                      atol=atol), f"RMSProp failed: {final_x_RMS}"
    assert np.isclose(final_x_QN, target_x,
                      atol=atol), f"QN failed: {final_x_QN}"

    print(f"GD Final x:    {final_x_GD:.4f}, Value: {y_GD:.4e}")
    print(f"SGD Final x:   {final_x_SGD:.4f}, Value: {y_SGD:.4e}")
    print(f"RMSProp Final x: {final_x_RMS:.4f}, Value: {y_RMS:.4e}")
    print(f"QN Final x:    {final_x_QN:.4f}, Value: {y_QN:.4e}")

    x_range = np.linspace(2, 6, 100)
    y_range = np.array([(x - 4)**2 for x in x_range])

    plt.figure(figsize=(9, 6))
    plt.plot(x_range, y_range, label='$f(x) = (x-4)^2$',
             color='blue', alpha=0.7)

    plt.plot(x_start[0], testf(x_start), 'go',
             markersize=8, label='Start ($x=5$)')

    plt.plot(final_x_GD, y_GD, 'rD', markersize=7,
             label=f'GD End ($x={final_x_GD:.3f}$)')
    plt.plot(final_x_SGD, y_SGD, 'mP', markersize=7,
             label=f'SGD End ($x={final_x_SGD:.3f}$)')
    plt.plot(final_x_RMS, y_RMS, 'c^', markersize=7,
             label=f'RMSProp End ($x={final_x_RMS:.3f}$)')
    plt.plot(final_x_QN, y_QN, 'ks', markersize=7,
             label=f'QN End ($x={final_x_QN:.3f}$)')

    plt.title('Minimizer Convergence Test')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def test_minimiser_3d():
    def rosenbrock(coords):
        x, y = coords[0], coords[1]
        a, b = 1, 100
        return (a - x)**2 + b * (y - x**2)**2

    def rosenbrock_grad(coords):
        x, y = coords[0], coords[1]
        a, b = 1, 100
        dx = -2 * (a - x) - 4 * b * x * (y - x**2)
        dy = 2 * b * (y - x**2)
        return np.array([dx, dy])

    def himmelblau(coords):
        x, y = coords[0], coords[1]
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    def himmelblau_grad(coords):
        x, y = coords[0], coords[1]
        dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dx, dy])

    f = himmelblau
    df = himmelblau_grad
    start_pos = np.array([2., 2.])
    stepsize = 0.001
    max_iter = 50000

    # Standard GD
    pos_GD, val_GD = minimisers.gradient_desecent(
        f, df, start_pos, stepsize, max_iter, detail=True)

    #  stochastic GD
    pos_SGD, val_SGD = minimisers.stochastic_GD(
        f=f, df=df, x_0=start_pos, stepsize=stepsize, noise=0.3, max_iter=max_iter, detail=True)

    # RMSProp
    pos_RMS, val_RMS = minimisers.RMSProp_GD(
        f, df, start_pos, stepsize=0.01, forgetting=0.9, max_iter=max_iter, detail=True)

    # Quasi-Newton (BFGS)
    pos_QN, val_QN = minimisers.quasi_newton(
        f, df, x_0=np.array([4, 5]), stepsize=0.01, method="DFP", max_iter=max_iter, detail=True)

    print(f"GD End: {pos_GD}, Val: {val_GD:.4f}")
    print(f"SGD End: {pos_SGD}, Val: {val_SGD:.4f}")
    print(f"RMS End:{pos_RMS}, Val: {val_RMS:.4f}")
    print(f"QN End: {pos_QN}, Val: {val_QN:.4f}")

    endpoint_grad = np.linalg.norm(df(pos_QN))
    print(f"\nQN endpoint derivative (norm): {endpoint_grad:.4f}")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           alpha=0.6, edgecolor='none')

    ax.scatter(start_pos[0], start_pos[1], f(start_pos),
               color='green', s=100, label='Start')

    ax.scatter(pos_GD[0], pos_GD[1], val_GD, color='red', s=50, label='GD')
    ax.scatter(pos_RMS[0], pos_RMS[1], val_RMS,
               color='cyan', s=50, label='RMSProp')
    ax.scatter(pos_QN[0], pos_QN[1], val_QN,
               color='black', s=50, label='Quasi-Newton')
    ax.scatter(pos_SGD[0], pos_SGD[1], val_SGD,
               color='red', s=50, label='Stochastic Gradient Descent')

    ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)

    ax.set_title("Function Optimization")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Energy)')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # test_minimiser()
    test_minimiser_3d()
