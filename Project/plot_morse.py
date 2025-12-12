def plot_energies():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.optimize import curve_fit
    from modules.helpers import hydrogen_molecule_helpers as hlp

    # --- 1. Load Data ---
    energy_filename = "/Users/louis/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Computer Science/Computational Physics/Project/morse_data/SR_Energy_Curve_100000.txt"
    thetas_filename = "/Users/louis/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Computer Science/Computational Physics/Project/morse_data/SR_Parameters_100000.txt"

    data = pd.read_csv(energy_filename, sep='\s+', comment='#', names=[
                       'Distance(a.u.)', 'Energy(Hartree)', 'Energy_Error(Hartree)'])

    r_0 = data['Distance(a.u.)'].to_numpy()
    energies = data['Energy(Hartree)'].to_numpy()
    energy_err = data['Energy_Error(Hartree)'].to_numpy()

    filt = energies < 0.
    r_0_filt = r_0[filt]
    energies_filt = energies[filt]

    # --- 2. Plot 1: Raw Energies with Error Bars (Marker = 'x') ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(r_0, energies, yerr=energy_err, label="Simulated Annealing Energies",
                 # Added capsize for error bars
                 marker='x', linestyle='None', markersize=6, capsize=3)
    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Energy")
    plt.title("Hydrogen Molecule Energy vs Interatomic Distance")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()

    # --- 3. Plot 2: Morse Potential Fit ---
    plt.figure(figsize=(8, 6))
    plt.plot(r_0_filt, energies_filt, label="Simulated Annealing Energies",
             marker='x', linestyle='None', markersize=6)  # Changed to 'x'

    f = hlp.V_morse
    p_0 = [0.17, 1.3, 1.4]
    fit, cov = curve_fit(f, r_0_filt, energies_filt, p0=p_0)
    plt.plot(r_0_filt, f(r_0_filt, *fit),
             label="Morse Potential Fit", color='red')

    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Energy")
    plt.title("Hydrogen Molecule Energy vs Interatomic Distance")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()

    print(
        f"Fitted Parameters:\n De (Depth) = {fit[0]:.4f}\n a  (Width) = {fit[1]:.4f}\n re (Eq. Pos)= {fit[2]:.4f}")
    print(f"Parameter Errors: {np.sqrt(np.diag(cov))}")

    # --- 4. Plot 3: All Thetas in ONE Subplot ---
    thetas = pd.read_csv(thetas_filename, sep='\s+', comment='#', names=[
                         'Distance', 'Theta1', 'Theta2', 'Theta3'])
    r_0_thetas = thetas['Distance'].to_numpy()
    theta1 = thetas['Theta1'].to_numpy()
    theta2 = thetas['Theta2'].to_numpy()
    theta3 = thetas['Theta3'].to_numpy()

    plt.figure(figsize=(8, 6))

    # Plot all three on the same axes
    plt.plot(r_0_thetas, theta1, label=r"$\theta_1$ (Orbital)",
             marker='x', linestyle='--', alpha=0.8)
    plt.plot(r_0_thetas, theta2, label=r"$\theta_2$ (Jastrow Num)",
             marker='x', linestyle='--', alpha=0.8)
    plt.plot(r_0_thetas, theta3, label=r"$\theta_3$ (Jastrow Denom)",
             marker='x', linestyle='--', alpha=0.8)

    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Parameter Value")
    plt.title("Evolution of Variational Parameters vs Distance")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_energies()
