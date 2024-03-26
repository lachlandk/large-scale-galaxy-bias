import h5py
import numpy as np
from datetime import datetime
from catalogue import *


def create_catalogue(lightcone_dir, files, filename, multiplier):
    select_galaxies(lightcone_dir, files, filename, "0.4<z<0.5", z_lims=(0.4, 0.5), mass_lims=(11, np.inf))
    create_random_catalogue(multiplier, filename, "0.4<z<0.5")

    # calculate number density
    with h5py.File(f"catalogues/{filename}", "r") as file:
        volume = np.pi/6 * (comoving_distance(0.5)**3 - comoving_distance(0.4)**3)
        number = file["0.4<z<0.5"]["Pos"].shape[0]
        number_density = number/volume
    print(f"Number of galaxies in the range 0.4<z<0.5: {number}")
    print(f"Galaxy number density: {number_density}/Mpc^3")

    # collect samples in other bins and cut them down to size
    z_bins = [(0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]
    for low_z, high_z in z_bins: 
        select_galaxies(lightcone_dir, files, filename, f"{low_z}<z<{high_z}", z_lims=(low_z, high_z), mass_lims=(11, np.inf))
        volume = np.pi/6 * (comoving_distance(high_z)**3 - comoving_distance(low_z)**3)
        number = np.round(number_density * volume).astype("int")
        with h5py.File(f"catalogues/{filename}", "r+") as file:
            catalogue = file[f"{low_z}<z<{high_z}"]
            selected = np.sort(np.argsort(catalogue["StellarMass"])[:number])
            catalogue["Pos"][:number] = catalogue["Pos"][selected]
            catalogue["Pos"].resize(number, axis=0)
            catalogue["ObsDist"][:number] = catalogue["ObsDist"][selected]
            catalogue["ObsDist"].resize(number, axis=0)
            catalogue["SpecZ"][:number] = catalogue["SpecZ"][selected]
            catalogue["SpecZ"].resize(number, axis=0)
            catalogue["ObsZ"][:number] = catalogue["ObsZ"][selected]
            catalogue["ObsZ"].resize(number, axis=0)
            catalogue["ObsMagDust"][:number] = catalogue["ObsMagDust"][selected]
            catalogue["ObsMagDust"].resize(number, axis=0)
            catalogue["StellarMass"][:number] = catalogue["StellarMass"][selected]
            catalogue["StellarMass"].resize(number, axis=0)
        
        create_random_catalogue(multiplier, filename, f"{low_z}<z<{high_z}")
        print(f"Number of galaxies in the range {low_z}<z<{high_z}: {number}")


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
