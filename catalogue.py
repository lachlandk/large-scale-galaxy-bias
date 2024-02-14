import h5py
import numpy as np
from tqdm import trange
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18 as cosmo, z_at_value


def create_catalogue(dir, num_files):
    with h5py.File("catalogue.hdf5", "w") as catalogue:
        cat_pos = catalogue.create_dataset("Pos", (0, 3), maxshape=(None, 3), dtype="f4")
        cat_vel = catalogue.create_dataset("RadialVel", (0,), maxshape=(None,), dtype="f4")
        cat_mass = catalogue.create_dataset("StellarMass", (0,), maxshape=(None,), dtype="f4")
        cat_num = np.zeros(num_files, dtype="int")

        for index in trange(num_files):
            with h5py.File(dir + f"gal_cone_01.{index}.hdf5", "r") as data:
                galaxies = data["Galaxies"]

                mag_limit = 19.5  # r band, get reference for this
                mag = np.array(galaxies["ObsMagDust"])  # u g r i z, magnitude with k-correction and corrected for dust extinction
                mag_filter = mag[:,2] < mag_limit
                cat_num[index] = np.count_nonzero(mag_filter)

                pos = np.array(galaxies["Pos"])[mag_filter]  # cMpc/h
                vel = np.array(galaxies["Vel"])[mag_filter]  # km/s
                stellar_mass = np.array(galaxies["StellarMass"])[mag_filter]  # 10^10 M_sol/h
            
                R = np.sqrt(pos[:,0]**2 + pos[:,1]**2)  # cMpc/h
                r = np.sqrt(R**2 + pos[:,2]**2)  # cMpc/h
                dec = 90 - 180/np.pi * np.arctan2(R, pos[:,2])  # degrees
                ra = 180/np.pi * np.arctan2(pos[:,1], pos[:,0])  # degrees

                v_r = (vel[:,0]*pos[:,0] + vel[:,1]*pos[:,1] + vel[:,2]*pos[:,2]) / r  # km/s

            total_galaxies = np.sum(cat_num)
            cat_pos.resize(total_galaxies, axis=0)
            cat_pos[cat_num[index - 1]:] = np.transpose([ra, dec, r])
            cat_vel.resize(total_galaxies, axis=0)
            cat_vel[cat_num[index - 1]:] = v_r
            cat_mass.resize(total_galaxies, axis=0)
            cat_mass[cat_num[index - 1]:] = stellar_mass
    return cat_num


def calculate_cosmological_redshift(distances):
    distances = distances * u.Mpc / cu.littleh
    distances = distances.to(u.Mpc, cu.with_H0())
    redshifts = z_at_value(cosmo.comoving_distance, distances, zmin=0, zmax=1.5, ztol=0.001)
    return redshifts


if __name__ == "__main__":
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    files = 155

    galaxy_numbers = create_catalogue(lightcone_dir, files)
    print(galaxy_numbers)
    print(np.sum(galaxy_numbers))

    with h5py.File("catalogue.hdf5", "r") as catalogue:
        pos = catalogue["Pos"]
        z = calculate_cosmological_redshift(pos[:,2])
        print(np.max(pos[:,2]), np.min(pos[:,2]))
        print(z)
        print(np.max(z), np.min(z))
