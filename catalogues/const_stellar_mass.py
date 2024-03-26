import numpy as np
from datetime import datetime
from catalogue import *

if __name__ == "__main__":
    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    files = 155
    multiplier = 50  # how many times bigger should the random catalogue be
    filename = "const_stellar_mass.hdf5"

    z_bins = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
    mass_bins = [(11, np.inf), (10, 11), (9, 10)]
    flags = [9, 10, 15]

    print("Creating constant stellar mass galaxy sample...")
    flag = 1
    for low_mass, high_mass in mass_bins:
        for low_z, high_z in z_bins:
            catalogue = f"{low_mass}<m<{high_mass}/{low_z}<z<{high_z}"
            if flag in flags:
                ra_lims = (30, 50)
                dec_lims = (30, 50)
            else:
                ra_lims = (0, 90)
                dec_lims = (0, 90)
            number = select_galaxies(lightcone_dir, files, filename, catalogue, z_lims=(low_z, high_z), mass_lims=(low_mass, high_mass), dec_lims=dec_lims, ra_lims=ra_lims)
            create_random_catalogue(multiplier, filename, catalogue)
            print(f"Number of galaxies in the range {low_mass}<m<{high_mass}, {low_z}<z<{high_z}: {number}")
            flag += 1

    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
