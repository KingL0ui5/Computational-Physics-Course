"""
A module to test minimizer functions 
Louis Liu 25/11
"""

import numpy as np
import seaborn as sns
import Project.modules.minimisers as minimisers

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


def test_minimiser(minimisers):
    x_start = np.array([5.0])
    stepsize = 0.1
    max_iterations = 50

    def testf(x_arr):
        x = x_arr[0]
        return (x - 4)**2

    def testdf(x_arr):
        x = x_arr[0]
        return np.array([2 * (x - 4)])

    final_x_arr, final_value = minimisers.gradient_desecent(
        testf,
        testdf,
        x_start,
        stepsize,
        max_iter=max_iterations
    )

    final_x = final_x_arr[0]

    assert np.isclose(
        final_x, 4.0, atol=1e-2), f"Minimizer failed to find x=4.0. Found x={final_x:.4f}"
    print(f"✅ Test Passed. Final x: {final_x:.4f}, Value: {final_value:.4e}")

    x_range = np.linspace(2, 6, 100)
    y_range = np.array([(x - 4)**2 for x in x_range])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, y_range, label='$f(x) = (x-4)^2$', color='blue')

    plt.plot(x_start[0], testf(x_start), 'go',
             markersize=8, label='Start ($x=5$)')
    plt.plot(final_x, final_value, 'r*', markersize=10,
             label=f'End ($x={final_x:.3f}$)')

    plt.title('Gradient Descent Convergence (1D Paraboloid)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    test_minimiser
