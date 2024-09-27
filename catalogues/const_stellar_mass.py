import numpy as np
from datetime import datetime
from catalogue import select_galaxies


def create_catalogue(lightcone_dir, files, filename, rsd=False):
    z_bins = [(0.74, 0.8), (0.68, 0.74), (0.62, 0.68), (0.56, 0.62), (0.5, 0.56), (0.44, 0.5), (0.38, 0.44), (0.32, 0.38), (0.26, 0.32), (0.2, 0.26)]
    mass_bins = [(11.5, np.inf), (11, 11.5), (10.5, 11)]

    for low_mass, high_mass in mass_bins:
        for low_z, high_z in z_bins:
            catalogue = f"{low_mass}<m<{high_mass}/{low_z}<z<{high_z}"
            total_galaxies, number_density, median_z = select_galaxies(lightcone_dir, files, filename, catalogue, z_lims=(low_z, high_z), mass_lims=(low_mass, high_mass), rsd=rsd)
            print(f"Number of galaxies in the range {low_mass}<m<{high_mass}, {low_z}<z<{high_z}: {total_galaxies}")
            print(f"Galaxy number density: {number_density}/cMpc^3")
            print(f"Median redshift: {median_z}")


if __name__ == "__main__":
    files = 155

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    print("Creating constant stellar mass galaxy sample for the A realisation (no RSDs)...")
    create_catalogue(lightcone_dir, files, "const_stellar_mass_A")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    print("Creating constant stellar mass galaxy sample for the B realisation (no RSDs)...")
    create_catalogue(lightcone_dir, files, "const_stellar_mass_B")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    print("Creating constant stellar mass galaxy sample for the A realisation (with RSDs)...")
    create_catalogue(lightcone_dir, files, "const_stellar_mass_rsd_A", rsd=True)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    print("Creating constant stellar mass galaxy sample for the B realisation (with RSDs)...")
    create_catalogue(lightcone_dir, files, "const_stellar_mass_rsd_B", rsd=True)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
