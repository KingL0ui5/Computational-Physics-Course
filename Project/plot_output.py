import pandas as pd


def plot_energies(filename="/Users/louis/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Computer Science/Computational Physics/output.txt"):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    from modules.helpers import hydrogen_molecule_helpers as hlp

    data = np.loadtxt(filename)
    r_0 = np.linspace(0.2, 5., len(data))

    plt.figure(figsize=(8, 6))
    plt.plot(r_0, data, label="Simulated Annealing Energies",
             marker='o', linestyle='None', markersize=4)

    # f = hlp.V_morse
    # fit, cov = curve_fit(f, r_0, data, p0=[-1.0, 1.0, 1.0, 0.0])
    # r_fit = np.linspace(0.1, 10., 200)
    # plt.plot(r_fit, f(r_fit, *fit), label="Morse Potential Fit", color='red')

    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Energy")
    plt.ylim([-1.5, 0])
    plt.title("Hydrogen Molecule Energy vs Interatomic Distance")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_energies("output.txt")
