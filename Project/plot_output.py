def plot_energies(filename="/Users/louis/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Computer Science/Computational Physics/output_10000.txt"):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    from modules.helpers import hydrogen_molecule_helpers as hlp

    data = np.loadtxt(filename)
    energies, r_0 = data[0], data[1]
    mask = energies < -0.9
    energies_filt = energies[mask]
    r_0_filt = r_0[mask]

    plt.figure(figsize=(8, 6))
    plt.plot(r_0_filt, energies_filt, label="Simulated Annealing Energies",
             marker='o', linestyle='None', markersize=4)

    f = hlp.V_morse
    p_0 = [0.17, 1.3, 1.4]
    fit, cov = curve_fit(f, r_0_filt, energies_filt, p0=p_0)
    plt.plot(r_0_filt, f(r_0_filt, *fit),
             label="Morse Potential Fit", color='red')

    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Energy")
    plt.title("Hydrogen Molecule Energy vs Interatomic Distance")
    plt.legend()
    plt.grid()
    plt.show()
    print(f"Fitted parameters: {fit}")


if __name__ == '__main__':
    plot_energies()
