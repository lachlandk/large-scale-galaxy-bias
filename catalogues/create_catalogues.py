import numpy as np
from datetime import datetime
from catalogue import *

if __name__ == "__main__":
    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    files = 155

    mag_limits = [19, 20, 21]
    mass_limits = [9.5, 10, 10.5, 11]
    for lim in mag_limits:
        print(f"Creating catalogue with magnitude limit {lim}... ")
        catalogue = create_data_catalogue(lightcone_dir, files, mag_lim=lim, z_lims=[0.2, 0.5])
        print(f"Created catalogue of size {np.sum(catalogue)}. Elapsed time: {datetime.now() - start_time}")
        print("Plotting map...")
        plot_catalogue(f"catalogues/data_catalogue_r={lim}_m=0_z=0.2-0.5.hdf5", f"maps/data_catalogue_r={lim}_m=0_z=0.2-0.5.png")
        print(f"Map plotted, elapsed time: {datetime.now() - start_time}")
    for lim in mass_limits:
        print(f"Creating catalogue with mass limit {lim} log_10(M_sol/h)... ")
        catalogue = create_data_catalogue(lightcone_dir, files, mass_lim=lim, z_lims=[0.2, 0.5])
        print(f"Created catalogue of size {np.sum(catalogue)}. Elapsed time: {datetime.now() - start_time}")
        print("Plotting map...")
        plot_catalogue(f"catalogues/data_catalogue_r=19.5_m={lim}_z=0.2-0.5.hdf5", f"maps/data_catalogue_r=19.5_m={lim}_z=0.2-0.5.png")
        print(f"Map plotted, elapsed time: {datetime.now() - start_time}")
