from datetime import datetime
from catalogue import *

if __name__ == "__main__":
    start_time = datetime.now()
    files = 155
    z_bins = [0, 0.03, 0.06, 0.1, 0.2, 0.3, 1]
    mag_limits = [17, 18, 19]
    random_catalogue_size = 500000

    print("Creating catalogues from the A realisation...")
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    for lim in mag_limits:
        print(f"Creating data catalogue with magnitude limit r<{lim}")
        create_data_catalogue(lightcone_dir, files, z_bins, mag_lim=lim, save_name=f"data_r<{lim}_A.hdf5")
        print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
        print(f"Creating random catalogue of size {random_catalogue_size}...")
        create_random_catalogue(random_catalogue_size, f"data_r<{lim}_A.hdf5", f"random_r<{lim}_A.hdf5")
        print(f"Random catalogue created, elapsed time: {datetime.now() - start_time}")
    print("Finished creating catalogues from the A realisation.")

    print("Creating catalogues from the B realisation...")
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01/"
    for lim in mag_limits:
        print(f"Creating data catalogue with magnitude limit r<{lim}")
        create_data_catalogue(lightcone_dir, files, z_bins, mag_lim=lim, save_name=f"data_r<{lim}_B.hdf5")
        print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
        print(f"Creating random catalogue of size {random_catalogue_size}...")
        create_random_catalogue(random_catalogue_size, f"data_r<{lim}_B.hdf5", f"random_r<{lim}_B.hdf5")
        print(f"Random catalogue created, elapsed time: {datetime.now() - start_time}")
    print("Finished creating catalogues from the B realisation.")
