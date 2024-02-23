import numpy as np
import matplotlib.pyplot as plt
from correlation_function import *

if __name__ == "__main__":
    mag_limits = [19, 20, 21]
    mass_limits = [9.5, 10, 10.5, 11]
    mag_limit_cfs = []
    mass_limit_cfs = []
    s = np.linspace(0.1, 300, 201)

    for lim in mag_limits:
        print(f"Calculating correlation function for catalogue with magnitude limit {lim}...")
        data_catalogue = f"data_catalogue_r={lim}_m=0_z=0.2-0.5_A_full.hdf5"
        save_name = f"correlation_function_r={lim}_m=0_z=0.2-0.5_nmu_bins=1.hdf5"
        xi = correlation_function(data_catalogue, "random_catalogue_20_000_000.hdf5", s, 1, 1, save_name=save_name)
        mag_limit_cfs.append(xi)

    nmu_bins = 100
    for lim in mass_limits:
        print(f"Calculating correlation function for catalogue with mass limit {lim}...")
        data_catalogue = f"data_catalogue_r=19.5_m={lim}_z=0.2-0.5_A_full.hdf5"
        save_name = f"correlation_function_r={lim}_m=0_z=0.2-0.5_nmu_bins={nmu_bins}.hdf5"
        xi = correlation_function(data_catalogue, "random_catalogue_20_000_000.hdf5", s, 1, nmu_bins, save_name=save_name)
        mass_limit_cfs.append(xi)

    # plot correlation function for mag limited catalogues
    fig, ax = plt.subplots(1, 1)

    ax.plot(s[:-1], xi)
    ax.set_xlabel("s [cMpc/h]")
    ax.set_ylabel("$\\xi$")
    # ax.set_ylim(0, 150)
    # ax.set_xlim(0, 300)

    fig.savefig("correlation_functions/mag_limited.png")

    # plot multipole moments of correlation function for mass limited catalogues
    fig, ax = plt.subplots(1, 1)

    ax.plot(s[:-1], xi_0, label="Monopole")
    ax.plot(s[:-1], xi_1, label="Dipole")
    ax.plot(s[:-1], xi_2, label="Quadrupole")
    ax.plot(s[:-1], xi_3, label="Octupole")
    ax.plot(s[:-1], xi_4, label="Hexadecapole")
    ax2.set_xlabel("s [cMpc/h]")
    ax2.set_ylabel("Multipole moments of $\\xi$")
    # ax.set_ylim(-100, 300)
    # ax.set_xlim(0, 300)
    # ax2.set_xscale("log")
    ax2.legend()

    fig.savefig("correlation_functions/mass_limited.png")
