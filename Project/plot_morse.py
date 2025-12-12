import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from modules.helpers import hydrogen_molecule_helpers as hlp


def plot_energies():
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
    energy_err_filt = energy_err[filt]

    plt.figure(figsize=(8, 6))
    plt.errorbar(r_0, energies, yerr=energy_err, label="Simulated Annealing Energies",
                 marker='x', linestyle='None', markersize=6, capsize=3)
    plt.xlabel("$r_0$ (a.u.)")
    plt.ylabel("Energy")
    plt.title("Hydrogen Molecule Energy vs Interatomic Distance")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].errorbar(r_0_filt, energies_filt, yerr=energy_err_filt, label="Stochastic Reconfiguraton Energies",
                   marker='x', linestyle='None', markersize=6, capsize=3)

    f = hlp.V_morse
    p_0 = [0.17, 1.3, 1.4]
    fit, cov = curve_fit(f, r_0_filt, energies_filt,
                         p0=p_0, sigma=energy_err_filt, absolute_sigma=False)

    r_smooth = np.linspace(min(r_0_filt), max(r_0_filt), 200)
    ax[0].plot(r_smooth, f(r_smooth, *fit),
               label="Morse Potential Fit", color='red')

    ax[0].set_xlabel("$r_0$ (a.u.)")
    ax[0].set_ylabel("Energy")
    ax[0].set_title("Hydrogen Molecule Energy vs Interatomic Distance")
    ax[0].legend()
    ax[0].grid(True, which='both', linestyle='--', alpha=0.7)

    print(
        f"Fitted Parameters:\n De (Depth) = {fit[0]:.4f}\n a  (Width) = {fit[1]:.4f}\n re (Eq. Pos)= {fit[2]:.4f}")
    print(f"Parameter Errors: {np.sqrt(np.diag(cov))}")

    thetas = pd.read_csv(thetas_filename, sep='\s+', comment='#', names=[
                         'Distance', 'Theta1', 'Theta2', 'Theta3'])

    thetas = thetas.sort_values(by='Distance')

    r_0_thetas = thetas['Distance'].to_numpy()
    theta1 = thetas['Theta1'].to_numpy()
    theta2 = thetas['Theta2'].to_numpy()
    theta3 = thetas['Theta3'].to_numpy()

    ax[1].plot(r_0_thetas, theta1, label=r"$\theta_1$ (Orbital)",
               marker='x', linestyle='--', alpha=0.8)
    ax[1].plot(r_0_thetas, theta2, label=r"$\theta_2$ (Jastrow Num)",
               marker='x', linestyle='--', alpha=0.8)
    ax[1].plot(r_0_thetas, theta3, label=r"$\theta_3$ (Jastrow Denom)",
               marker='x', linestyle='--', alpha=0.8)

    ax[1].set_xlabel("$r_0$ (a.u.)")
    ax[1].set_ylabel("Parameter Value")
    ax[1].set_title("Evolution of Variational Parameters vs Distance")
    ax[1].legend()
    ax[1].grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_energies()
