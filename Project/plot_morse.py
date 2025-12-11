def plot_energies():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.optimize import curve_fit
    from modules.helpers import hydrogen_molecule_helpers as hlp

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

    plt.errorbar(r_0, energies, yerr=energy_err, label="Simulated Annealing Energies",
                 marker='o', linestyle='None', markersize=4)
    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Energy")
    plt.title("Hydrogen Molecule Energy vs Interatomic Distance")
    plt.legend()
    plt.grid()
    plt.show()

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
    print(
        f"Fitted Parameters:\n De (Depth) = {fit[0]:.4f}\n a  (Width) = {fit[1]:.4f}\n re (Eq. Pos)= {fit[2]:.4f}")
    print(f"Parameter Errors: {np.sqrt(np.diag(cov))}")

    thetas = pd.read_csv(thetas_filename, sep='\s+', comment='#', names=[
                         'Distance', 'Theta1', 'Theta2', 'Theta3'])
    r_0_thetas = thetas['Distance'].to_numpy()
    theta1 = thetas['Theta1'].to_numpy()
    theta2 = thetas['Theta2'].to_numpy()
    theta3 = thetas['Theta3'].to_numpy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0, 0].plot(r_0_thetas, theta1, label="Theta 1", marker='o',
                   linestyle='None', markersize=4)
    axs[0, 0].set_xlabel("$r_0$ (a.u.)")
    axs[0, 0].set_ylabel("Theta 1")
    axs[0, 0].set_title("Theta 1 vs Interatomic Distance")
    axs[0, 0].grid()
    axs[0, 1].plot(r_0_thetas, theta2, label="Theta 2", marker='o',
                   linestyle='None', markersize=4)
    axs[0, 1].set_xlabel("$r_0$ (a.u.)")
    axs[0, 1].set_ylabel("Theta 2")
    axs[0, 1].set_title("Theta 2 vs Interatomic Distance")
    axs[0, 1].grid()
    axs[1, 0].plot(r_0_thetas, theta3, label="Theta 3", marker='o',
                   linestyle='None', markersize=4)
    axs[1, 0].set_xlabel("$r_0$ (a.u.)")
    axs[1, 0].set_ylabel("Theta 3")
    axs[1, 0].set_title("Theta 3 vs Interatomic Distance")
    axs[1, 0].grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_energies()
