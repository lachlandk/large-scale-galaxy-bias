import numpy as np
from datetime import datetime
from catalogue import *

if __name__ == "__main__":
    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    files = 155

    mag_limits = [19, 20, 21]
    mass_limits = [9.5, 10, 10.5, 11]
    z_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(len(z_bins) - 1):
        print(f"Creating catalouges in redshift range z={z_bins[i]}-{z_bins[i+1]}...")
        for lim in mag_limits:
            print(f"Creating catalogue with magnitude limit {lim}... ")
            catalogue = create_data_catalogue(lightcone_dir, files, mag_lim=lim, z_lims=[z_bins[i], z_bins[i+1]])
            print(f"Created catalogue of size {np.sum(catalogue)}. Elapsed time: {datetime.now() - start_time}")
        for lim in mass_limits:
            print(f"Creating catalogue with mass limit {lim} log_10(M_sol/h)... ")
            catalogue = create_data_catalogue(lightcone_dir, files, mass_lim=lim, z_lims=[z_bins[i], z_bins[i+1]])
            print(f"Created catalogue of size {np.sum(catalogue)}. Elapsed time: {datetime.now() - start_time}")
        print(f"Creating random catalogue for redshift range z={z_bins[i]}-{z_bins[i+1]}...")
        create_random_catalogue(1000000, f"data_catalogue_r=19_m=0_z={z_bins[i]}-{z_bins[i+1]}.hdf5", f"random_catalogue_1_000_000_z={z_bins[i]}-{z_bins[i+1]}.hdf5")
        print(f"Created catalouges in redshift range z={z_bins[i]}-{z_bins[i+1]}. Elapsed time: {datetime.now() - start_time}")
