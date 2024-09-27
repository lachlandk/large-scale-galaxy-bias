import h5py
import numpy as np
from datetime import datetime
from catalogue import select_galaxies


def create_catalogue(lightcone_dir, files, filename, rsd=False):
    # calculate number density from largest redshift bin
    total_galaxies, fixed_number_density, median_z = select_galaxies(lightcone_dir, files, filename, "0.45<z<0.5", z_lims=(0.45, 0.5), mass_lims=(11, np.inf), rsd=rsd)
    print(f"Number of galaxies in the range 0.45<z<0.5: {total_galaxies}")
    print(f"Galaxy number density: {fixed_number_density}/cMpc^3")
    print(f"Median redshift: {median_z}")

    # collect samples in other bins and cut them down to size
    z_bins = [(0.4, 0.45), (0.35, 0.4), (0.3, 0.35), (0.25, 0.3), (0.2, 0.25), (0.15, 0.2), (0.1, 0.15), (0.05, 0.1), (0.0, 0.05)]
    for low_z, high_z in z_bins:
        total_galaxies, number_density, _ = select_galaxies(lightcone_dir, files, filename, f"{low_z}<z<{high_z}", z_lims=(low_z, high_z), mass_lims=(11, np.inf), rsd=rsd)
        with h5py.File(f"catalogues/{filename}.hdf5", "r+") as file:
            catalogue = file[f"{low_z}<z<{high_z}"]
            volume = total_galaxies / number_density
            limit_number = np.round(fixed_number_density * volume).astype("int")
            selected = np.sort(np.argsort(catalogue["StellarMass"])[:limit_number])
            catalogue["Pos"][:limit_number] = catalogue["Pos"][selected]
            catalogue["Pos"].resize(limit_number, axis=0)
            catalogue["z"][:limit_number] = catalogue["z"][selected]
            catalogue["z"].resize(limit_number, axis=0)
            catalogue["ObsMag"][:limit_number] = catalogue["ObsMag"][selected]
            catalogue["ObsMag"].resize(limit_number, axis=0)
            catalogue["StellarMass"][:limit_number] = catalogue["StellarMass"][selected]
            catalogue["StellarMass"].resize(limit_number, axis=0)

            median_z = np.median(catalogue["z"])
            catalogue.attrs["median_z"] = median_z

        print(f"Number of galaxies in the range {low_z}<z<{high_z}: {limit_number}")
        print(f"Galaxy number density: {limit_number / volume}/cMpc^3")
        print(f"Median redshift: {median_z}")


if __name__ == "__main__":
    files = 155

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    print("Creating constant number density galaxy sample for the A realisation (no RSDS)...")
    create_catalogue(lightcone_dir, files, "const_number_density_A")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    print("Creating constant number density galaxy sample for the B realisation (no RSDs)...")
    create_catalogue(lightcone_dir, files, "const_number_density_B")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    print("Creating constant number density galaxy sample for the A realisation (with RSDS)...")
    create_catalogue(lightcone_dir, files, "const_number_density_rsd_A", rsd=True)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-B/SAM/galaxies_lightcone_01"
    print("Creating constant number density galaxy sample for the B realisation (with RSDs)...")
    create_catalogue(lightcone_dir, files, "const_number_density_rsd_B", rsd=True)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
