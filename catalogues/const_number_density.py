import h5py
import numpy as np
from datetime import datetime
from catalogue import *


def create_catalogue(lightcone_dir, files, filename, multiplier):
    # calculate number density from largest redshift bin
    total_galaxies, fixed_number_density, median_z = select_galaxies(lightcone_dir, files, filename, "0.45<z<0.5", z_lims=(0.45, 0.5), mass_lims=(11, np.inf))
    create_random_catalogue(multiplier, filename, "0.45<z<0.5")
    print(f"Number of galaxies in the range 0.45<z<0.5: {total_galaxies}")
    print(f"Galaxy number density: {fixed_number_density}/Mpc^3")
    print(f"Median cosmological redshift: {median_z}")

    # collect samples in other bins and cut them down to size
    z_bins = [(0.4, 0.45), (0.35, 0.4), (0.3, 0.35) (0.25, 0.3), (0.2, 0.25), (0.15, 0.2), (0.1, 0.15), (0.05, 0.1), (0.0, 0.05)]
    for low_z, high_z in z_bins: 
        total_galaxies, number_density, median_z = select_galaxies(lightcone_dir, files, filename, f"{low_z}<z<{high_z}", z_lims=(low_z, high_z), mass_lims=(11, np.inf))
        with h5py.File(f"catalogues/{filename}", "r+") as file:
            catalogue = file[f"{low_z}<z<{high_z}"]
            volume = total_galaxies / number_density
            limit_number = np.round(fixed_number_density * volume).astype("int")
            selected = np.sort(np.argsort(catalogue["StellarMass"])[:limit_number])
            catalogue["Pos"][:limit_number] = catalogue["Pos"][selected]
            catalogue["Pos"].resize(limit_number, axis=0)
            catalogue["ObsDist"][:limit_number] = catalogue["ObsDist"][selected]
            catalogue["ObsDist"].resize(limit_number, axis=0)
            catalogue["CosZ"][:limit_number] = catalogue["CosZ"][selected]
            catalogue["CosZ"].resize(limit_number, axis=0)
            catalogue["ObsZ"][:limit_number] = catalogue["ObsZ"][selected]
            catalogue["ObsZ"].resize(limit_number, axis=0)
            catalogue["ObsMag"][:limit_number] = catalogue["ObsMag"][selected]
            catalogue["ObsMag"].resize(limit_number, axis=0)
            catalogue["StellarMass"][:limit_number] = catalogue["StellarMass"][selected]
            catalogue["StellarMass"].resize(limit_number, axis=0)
        
        create_random_catalogue(multiplier, filename, f"{low_z}<z<{high_z}")
        print(f"Number of galaxies in the range {low_z}<z<{high_z}: {limit_number}")
        print(f"Galaxy number density: {limit_number / volume}/Mpc^3")
        print(f"Median cosmological redshift: {median_z}")


if __name__ == "__main__":
    files = 155
    multiplier = 50  # how many times bigger should the random catalogue be

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    filename = "const_number_density_A.hdf5"
    print("Creating constant number density galaxy sample for the A realisation...")
    create_catalogue(lightcone_dir, files, filename, multiplier)    
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    filename = "const_number_density_B.hdf5"
    print("Creating constant number density galaxy sample for the B realisation...")
    create_catalogue(lightcone_dir, files, filename, multiplier)    
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
