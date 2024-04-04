from datetime import datetime
from catalogue import *


def create_catalogue(lightcone_dir, files, filename, multiplier):
    z_bins = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]
    flags = [5, 6]
    flags_2 = [7, 6, 9, 10]

    flag = 1
    for low_z, high_z in z_bins:
        if flag in flags:
            ra_lims = (30, 60)
            dec_lims = (30, 60)
        elif flag in flags_2:
            ra_lims = (40, 60)
            dec_lims = (40, 60)
        else:
            ra_lims = (0, 90)
            dec_lims = (0, 90)
        number = select_galaxies(lightcone_dir, files, filename, f"{low_z}<z<{high_z}", z_lims=(low_z, high_z), dec_lims=dec_lims, ra_lims=ra_lims)
        create_random_catalogue(multiplier, filename, f"{low_z}<z<{high_z}")
        print(f"Number of galaxies in the range {low_z}<z<{high_z}{' (cut sky)' if flag in flags else ' (super cut sky)' if flag in flags_2 else ''}: {number}")
        flag += 1


if __name__ == "__main__":
    files = 155
    multiplier = 50  # how many times bigger should the random catalogue be
    
    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    filename = "magnitude_limited_A.hdf5"
    print("Creating magnitude limited galaxy sample for the A realisation...")
    create_catalogue(lightcone_dir, files, filename, multiplier)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    filename = "magnitude_limited_B.hdf5"
    print("Creating magnitude limited galaxy sample for the B realisation...")
    create_catalogue(lightcone_dir, files, filename, multiplier)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
