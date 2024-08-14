from datetime import datetime
from catalogue import select_galaxies


def create_catalogue(lightcone_dir, files, filename):
    z_bins = [(0.74, 0.8), (0.68, 0.74), (0.62, 0.68), (0.56, 0.62), (0.5, 0.56), (0.44, 0.5), (0.38, 0.44), (0.32, 0.38), (0.26, 0.32), (0.2, 0.26)]

    for low_z, high_z in z_bins:
        total_galaxies, number_density, median_z = select_galaxies(lightcone_dir, files, filename, f"{low_z}<z<{high_z}", z_lims=(low_z, high_z))
        print(f"Number of galaxies in the range {low_z}<z<{high_z}: {total_galaxies}")
        print(f"Galaxy number density: {number_density}/cMpc^3")
        print(f"Median cosmological redshift: {median_z}")


if __name__ == "__main__":
    files = 155
    
    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    filename = "magnitude_limited_A.hdf5"
    print("Creating magnitude limited galaxy sample for the A realisation...")
    create_catalogue(lightcone_dir, files, filename)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    filename = "magnitude_limited_B.hdf5"
    print("Creating magnitude limited galaxy sample for the B realisation...")
    create_catalogue(lightcone_dir, files, filename)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
