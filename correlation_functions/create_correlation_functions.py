import numpy as np
import matplotlib.pyplot as plt
from correlation_function import *

if __name__ == "__main__":
    s_bins = np.geomspace(30, 100, 100)

    print(f"Calculating correlation functions for constant number density sample...")
    z_bins = [(0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]
    for low_z, high_z in z_bins:
        print(f"Starting on bin {low_z}<z<{high_z}...")
        xi_A, sigma_A = correlation_function("const_number_density_A.hdf5", f"{low_z}<z<{high_z}", s_bins)
        xi_B, sigma_B = correlation_function("const_number_density_B.hdf5", f"{low_z}<z<{high_z}", s_bins)
        xi = np.mean([xi_A, xi_B], axis=0)
        sigma = np.mean([sigma_A, sigma_B], axis=0)

        with h5py.File("correlation_functions/corrfunc_const_number_density.hdf5", "a") as corrfunc:
            save = corrfunc.create_group(f"{low_z}<z<{high_z}")
            save.create_dataset("s", data=s_bins[:-1])
            save.create_dataset("xi_0", data=xi)
            save.create_dataset("sigma", data=sigma)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f"Correlation Function in Real Space {low_z}<z<{high_z}")

        ax.plot(s_bins[:-1], xi)
        ax.fill_between(s_bins[:-1], xi+sigma, xi-sigma, alpha=0.5)
        ax.set_xlabel("$r$ [cMpc]")
        ax.set_ylabel("$\\xi(r)$")

        fig.savefig(f"correlation_functions/corrfunc_const_number_density_{low_z}_z_{high_z}.png")

    print(f"Calculating correlation functions for magnitude limited sample...")
    z_bins = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6)]
    for low_z, high_z in z_bins:
        print(f"Starting on bin {low_z}<z<{high_z}...")
        xi_A, sigma_A = correlation_function("magnitude_limited_A.hdf5", f"{low_z}<z<{high_z}", s_bins)
        xi_B, sigma_B = correlation_function("magnitude_limited_B.hdf5", f"{low_z}<z<{high_z}", s_bins)
        xi = np.mean([xi_A, xi_B], axis=0)
        sigma = np.mean([sigma_A, sigma_B], axis=0)

        with h5py.File("correlation_functions/corrfunc_magnitude_limited.hdf5", "a") as corrfunc:
            save = corrfunc.create_group(f"{low_z}<z<{high_z}")
            save.create_dataset("s", data=s_bins[:-1])
            save.create_dataset("xi_0", data=xi)
            save.create_dataset("sigma", data=sigma)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f"Correlation Function in Real Space {low_z}<z<{high_z}")

        ax.plot(s_bins[:-1], xi)
        ax.fill_between(s_bins[:-1], xi+sigma, xi-sigma, alpha=0.5)
        ax.set_xlabel("$r$ [cMpc]")
        ax.set_ylabel("$\\xi(r)$")

        fig.savefig(f"correlation_functions/corrfunc_magnitude_limited_{low_z}_z_{high_z}.png")

    print(f"Calculating correlation functions for constant stellar mass sample...")
    z_bins = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
    mass_bins = [(11, np.inf), (10, 11), (9, 10)]
    for low_mass, high_mass in mass_bins:
        for low_z, high_z in z_bins:
            print(f"Starting on bin {low_mass}<m<{high_mass}/{low_z}<z<{high_z}...")
            xi_A, sigma_A = correlation_function("const_stellar_mass_A.hdf5", f"{low_mass}<m<{high_mass}/{low_z}<z<{high_z}", s_bins)
            xi_B, sigma_B = correlation_function("const_stellar_mass_B.hdf5", f"{low_mass}<m<{high_mass}/{low_z}<z<{high_z}", s_bins)
            xi = np.mean([xi_A, xi_B], axis=0)
            sigma = np.mean([sigma_A, sigma_B], axis=0)

            with h5py.File("correlation_functions/corrfunc_const_stellar_mass.hdf5", "a") as corrfunc:
                save = corrfunc.create_group(f"{low_mass}<m<{high_mass}/{low_z}<z<{high_z}")
                save.create_dataset("s", data=s_bins[:-1])
                save.create_dataset("xi_0", data=xi)
                save.create_dataset("sigma", data=sigma)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            fig.suptitle(f"Correlation Function in Real Space {low_mass}<m<{high_mass}, {low_z}<z<{high_z}")

            ax.plot(s_bins[:-1], xi)
            ax.fill_between(s_bins[:-1], xi+sigma, xi-sigma, alpha=0.5)
            ax.set_xlabel("$r$ [cMpc]")
            ax.set_ylabel("$\\xi(r)$")
            
            fig.savefig(f"correlation_functions/corrfunc_const_stellar_mass_{low_mass}_m_{high_mass}_{low_z}_z_{high_z}.png")
