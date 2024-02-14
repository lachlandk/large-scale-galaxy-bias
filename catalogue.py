import h5py
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18 as cosmo, z_at_value
from mpl_toolkits.axisartist.floating_axes import FloatingSubplot, GridHelperCurveLinear
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D

plt.switch_backend("agg")


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


def calculate_cosmological_redshift(distance):
    distance = distance * u.Mpc / cu.littleh
    distance = distance.to(u.Mpc, cu.with_H0())
    return z_at_value(cosmo.comoving_distance, distance, zmin=0, zmax=1.5, ztol=0.001)


def calculate_peculiar_redshift(v_r):
    c = 3e5  # km/s
    return np.sqrt((1 + v_r/c) / (1 - v_r/c)) - 1


def plot_catalogue(ra, z, mass, save_name):
    fig = plt.figure(figsize=(20, 20))

    tr_scale = Affine2D().scale(np.pi/180.0, 1.0)
    transform = tr_scale + PolarAxes.PolarTransform()
    grid_locator1 = angle_helper.LocatorHMS(8)
    tick_formatter1 = angle_helper.FormatterHMS()
    grid_locator2 = MaxNLocator(3)
    grid_helper = GridHelperCurveLinear(transform, extremes=(90, 0, np.max(z), 0), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)

    ax = FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax)

    # distance axis ticks and label
    ax.axis["left"].toggle(ticklabels=False)
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].set_axis_direction("bottom")
    ax.axis["right"].label.set_visible(True)
    ax.axis["right"].label.set_text("z")

    # angle axis ticks and label
    ax.axis["bottom"].major_ticklabels.set_axis_direction("top")
    ax.axis["bottom"].label.set_axis_direction("top")
    ax.axis["bottom"].label.set_text("RA")

    ax.axis["top"].set_visible(False)

    aux_ax = ax.get_aux_axes(transform)
    aux_ax.patch = ax.patch
    ax.patch.zorder = 0

    scatter = aux_ax.scatter(ra, z, c=mass, cmap="spring", s=0.1, marker=".")

    ax.set_facecolor("black")

    fig.colorbar(scatter, ax=ax, label="Stellar Mass [$\\log_{10}10^{10}M_\\odot/h$]")

    plt.savefig(save_name)


def plot_real_space_catalogue(filename):
    with h5py.File(filename, "r") as catalogue:
        pos = np.array(catalogue["Pos"])
        cosmo_z = np.array(catalogue["z"])
        mass = np.log10(np.array(catalogue["StellarMass"]))
        plot_catalogue(pos[:,0], cosmo_z, mass, "real_space_catalogue.png")


def plot_redshift_space_catalogue(filename):
    with h5py.File(filename, "r") as catalogue:
        pos = np.array(catalogue["Pos"])
        cosmo_z = np.array(catalogue["zSpec"])
        mass = np.log10(np.array(catalogue["StellarMass"]))
        plot_catalogue(pos[:,0], cosmo_z, mass, "redshift_space_catalogue.png")        


if __name__ == "__main__":
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    files = 155

    galaxy_numbers = create_catalogue(lightcone_dir, files)
    print("Selected galaxies in each file: ", galaxy_numbers)
    print("Total galaxy number in catalogue: ", np.sum(galaxy_numbers))

    with h5py.File("catalogue.hdf5", "r+") as catalogue:
        pos = np.array(catalogue["Pos"])
        v_r = np.array(catalogue["RadialVel"])

        cosmo_z = calculate_cosmological_redshift(pos[:,2])
        spec_z = cosmo_z + calculate_peculiar_redshift(v_r)

        catalogue.create_dataset("z", data=cosmo_z, dtype="f4")
        catalogue.create_dataset("zSpec", data=spec_z, dtype="f4")

    plot_real_space_catalogue("catalogue.hdf5")
    plot_redshift_space_catalogue("catalogue.hdf5")
