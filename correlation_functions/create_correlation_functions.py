import numpy as np
from correlation_function import *


def create_correlation_functions(filename, z_bins, base_catalogue=None):
    s_bins = np.linspace(30, 100, 100)
    base = f"{base_catalogue}/" if base_catalogue is not None else ""
    for low_z, high_z in z_bins:
        print(f"Starting on bin {low_z}<z<{high_z}...")
        xi_A, sigma_A, median_z_A = correlation_function(f"{filename}_A.hdf5", f"{base}{low_z}<z<{high_z}", s_bins)
        xi_B, sigma_B, median_z_B = correlation_function(f"{filename}_B.hdf5", f"{base}{low_z}<z<{high_z}", s_bins)
        xi = np.mean([xi_A, xi_B], axis=0)
        sigma = np.mean([sigma_A, sigma_B], axis=0)
        median_z = (median_z_A + median_z_B) / 2

        with h5py.File(f"correlation_functions/corrfunc_{filename}.hdf5", "a") as corrfunc:
            if f"{base}{low_z}<z<{high_z}" in corrfunc:
                save = corrfunc[f"{base}{low_z}<z<{high_z}"]
                del save["s"]
                del save["xi_0"]
                del save["sigma"]
            else:
                save = corrfunc.create_group(f"{base}{low_z}<z<{high_z}")
            save.create_dataset("s", data=s_bins[:-1])
            save.create_dataset("xi_0", data=xi)
            save.create_dataset("sigma", data=sigma)
            save.attrs["median_z_cos"] = median_z


if __name__ == "__main__":
    print(f"Calculating correlation functions for constant number density sample...")
    z_bins = [(0.45, 0.5), (0.4, 0.45), (0.35, 0.4), (0.3, 0.35), (0.25, 0.3), (0.2, 0.25), (0.15, 0.2), (0.1, 0.15), (0.05, 0.1), (0.0, 0.05)]
    create_correlation_functions("const_number_density", z_bins)

    print(f"Calculating correlation functions for magnitude limited sample...")
    z_bins = [(0.74, 0.8), (0.68, 0.74), (0.62, 0.68), (0.56, 0.62), (0.5, 0.56), (0.44, 0.5), (0.38, 0.44), (0.32, 0.38), (0.26, 0.32), (0.2, 0.26)]
    create_correlation_functions("magnitude_limited", z_bins)

    print(f"Calculating correlation functions for constant stellar mass sample...")
    z_bins = [(0.74, 0.8), (0.68, 0.74), (0.62, 0.68), (0.56, 0.62), (0.5, 0.56), (0.44, 0.5), (0.38, 0.44), (0.32, 0.38), (0.26, 0.32), (0.2, 0.26)]
    mass_bins = [(11.5, np.inf), (11, 11.5), (10.5, 11)]
    for low_mass, high_mass in mass_bins:
        create_correlation_functions("const_stellar_mass", z_bins, f"{low_mass}<m<{high_mass}")
