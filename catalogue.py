import h5py
import numpy as np
from tqdm import trange
from datetime import datetime
from scipy import optimize, interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.floating_axes import FloatingSubplot, GridHelperCurveLinear
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.basemap import Basemap

plt.switch_backend("agg")

# cosmological parameters
c = 3e5  # km/s
Omega_m_0 = 0.30966
Omega_Lambda_0 = 0.6888463055445441

# cosmological functions
def comoving_distance(z):
    integration_range = np.linspace(0, z, 100)
    return np.trapz(c / (100*np.sqrt(Omega_m_0 * (1 + integration_range)**3 + Omega_Lambda_0)), integration_range)


def create_data_catalogue(dir, num_files):
    with h5py.File("catalogue.hdf5", "w") as catalogue:
        cat_pos = catalogue.create_dataset("Pos", (0, 3), maxshape=(None, 3), dtype="f8")
        cat_vel = catalogue.create_dataset("RadialVel", (0,), maxshape=(None,), dtype="f8")
        cat_mass = catalogue.create_dataset("StellarMass", (0,), maxshape=(None,), dtype="f4")
        cat_mag = catalogue.create_dataset("ObsMagDust", (0, 5), maxshape=(None, 5), dtype="f4")
        cat_num = np.zeros(num_files, dtype="int")

        for index in trange(num_files):
            with h5py.File(dir + f"gal_cone_01.{index}.hdf5", "r") as data:
                galaxies = data["Galaxies"]

                mag_limit = 19.5  # r band, same as DESI Bright Galaxy Sample
                mag = np.array(galaxies["ObsMagDust"])  # u g r i z, magnitude with k-correction and corrected for dust extinction
                mag_filter = mag[:,2] < mag_limit
                cat_num[index] = np.count_nonzero(mag_filter)

                pos = np.array(galaxies["Pos"])[mag_filter]  # cMpc/h
                vel = np.array(galaxies["Vel"])[mag_filter]  # km/s
                stellar_mass = np.array(galaxies["StellarMass"])[mag_filter]  # 10^10 M_sol/h
                filtered_mag = mag[mag_filter]  # mag
            
                R = np.sqrt(pos[:,0]**2 + pos[:,1]**2)  # cMpc/h
                r = np.sqrt(R**2 + pos[:,2]**2)  # cMpc/h
                dec = 90 - 180/np.pi * np.arctan2(R, pos[:,2])  # degrees
                ra = 180/np.pi * np.arctan2(pos[:,1], pos[:,0])  # degrees

                v_r = (vel[:,0]*pos[:,0] + vel[:,1]*pos[:,1] + vel[:,2]*pos[:,2]) / r  # km/s

            total_galaxies = np.sum(cat_num)
            start_index = np.sum(cat_num[:index])
            cat_pos.resize(total_galaxies, axis=0)
            cat_pos[start_index:] = np.transpose([ra, dec, r])
            cat_vel.resize(total_galaxies, axis=0)
            cat_vel[start_index:] = v_r
            cat_mass.resize(total_galaxies, axis=0)
            cat_mass[start_index:] = stellar_mass
            cat_mag.resize(total_galaxies, axis=0)
            cat_mag[start_index:] = filtered_mag
    return cat_num


def create_random_catalogue(size):
    with h5py.File("random_catalogue.hdf5", "w") as catalogue:
        ra = np.random.default_rng().uniform(0 , 90, size)
        dec = 90 - 180 / np.pi * np.arccos(np.random.default_rng().uniform(0, 1, size))

        with h5py.File("catalogue.hdf5", "r") as data:
            data_mass = np.array(data["StellarMass"])
            data_mag = np.array(data["ObsMagDust"])
            data_z = np.array(data["z"])

            z = np.random.default_rng().choice(data_z, size)
            mass = np.random.default_rng().choice(data_mass, size)
            mag = np.random.default_rng().choice(data_mag, size)

            dist = np.ndarray(size)
            for i in range(size):
                dist[i] = comoving_distance(z[i])

        catalogue.create_dataset("Pos", (size, 3), data=np.transpose([ra, dec, dist]), dtype="f8")
        catalogue.create_dataset("z", (size,), data=z, dtype="f8")
        catalogue.create_dataset("StellarMass", (size,), data=mass, dtype="f4")
        catalogue.create_dataset("ObsMagDust", (size, 5), data=mag, dtype="f4")


def plot_catalogue(filename, save_name):
    with h5py.File(filename, "r") as catalogue:
        pos = np.array(catalogue["Pos"])
        ra = pos[:,0]
        dist = pos[:,2]
        mass = np.log10(np.array(catalogue["StellarMass"]))

    fig = plt.figure(figsize=(30, 30))

    tr_scale = Affine2D().scale(np.pi/180.0, 1.0)
    transform = tr_scale + PolarAxes.PolarTransform()
    grid_locator1 = angle_helper.LocatorHMS(8)
    tick_formatter1 = angle_helper.FormatterHMS()
    grid_locator2 = MaxNLocator(3)
    grid_helper = GridHelperCurveLinear(transform, extremes=(90, 0, np.max(dist), 0), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)

    ax = FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax)

    # distance axis ticks and label
    ax.axis["left"].toggle(ticklabels=False)
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].set_axis_direction("bottom")
    ax.axis["right"].label.set_visible(True)
    ax.axis["right"].label.set_text("Distance [cMpc/h]")

    # angle axis ticks and label
    ax.axis["bottom"].major_ticklabels.set_axis_direction("top")
    ax.axis["bottom"].label.set_axis_direction("top")
    ax.axis["bottom"].label.set_text("RA")

    ax.axis["top"].set_visible(False)

    aux_ax = ax.get_aux_axes(transform)
    aux_ax.patch = ax.patch
    ax.patch.zorder = 0

    scatter = aux_ax.scatter(ra, dist, c=mass, cmap="spring", s=0.1, marker=".")

    ax.set_facecolor("black")

    fig.colorbar(scatter, ax=ax, label="Stellar Mass [$\\log_{10}10^{10}M_\\odot/h$]")

    plt.savefig(save_name)


def plot_catalogue_map(filename, save_name):
    with h5py.File(filename, "r") as catalogue:
        pos = np.array(catalogue["Pos"])
        mass = np.log10(np.array(catalogue["StellarMass"]))

        fig, ax = plt.subplots(1, 1, figsize=(30, 30))
        fig.suptitle("Catalogue Map", fontsize=40)

        map = Basemap(projection="ortho", lat_0=45, lon_0=45, ax=ax)

        map.drawmeridians(np.arange(0, 360, 30), color="white", textcolor="white", dashes=(None, None), latmax=90)
        map.drawparallels(np.arange(-90, 90, 30), color="white", textcolor="white", dashes=(None, None))
        map.drawmapboundary(fill_color="black")

        scatter = map.scatter(pos[:,0], pos[:,1], latlon=True, c=mass, cmap="spring", s=0.1, marker=".")

        map.colorbar(scatter, ax=ax, label="Stellar Mass [$\\log_{10}10^{10}M_\\odot/h$]")

        plt.savefig(save_name)      


if __name__ == "__main__":
    start_time = datetime.now()
    # lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    # files = 155
    lightcone_dir = "./test/"
    files = 2

    print("Creating galaxy catalogue...")
    galaxy_numbers = create_data_catalogue(lightcone_dir, files)
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
    print("Selected galaxies in each file: ", galaxy_numbers)
    print("Total galaxy number in catalogue: ", np.sum(galaxy_numbers))

    with h5py.File("catalogue.hdf5", "r+") as catalogue:
        pos = np.array(catalogue["Pos"])
        v_r = np.array(catalogue["RadialVel"])

        print("Calculating cosmological redshifts...")
        cosmo_z = np.ndarray(pos.shape[0])

        num_interp_points = 100
        interp_points_r = np.ndarray(num_interp_points)
        interp_points_z = np.linspace(0, 1.5, num_interp_points)
        for i in range(num_interp_points):
            interp_points_r[i] = comoving_distance(interp_points_z[i])  # [cMpc/h]

        interp = interpolate.CubicSpline(interp_points_z, interp_points_r, extrapolate=False)

        for i in trange(pos.shape[0]):
            cosmo_z[i] = optimize.root_scalar(lambda z: pos[i,2] - interp(z), bracket=[0, 1.5]).root
        print(f"Cosmological redshifts calculated, elapsed time: {datetime.now() - start_time}")

        print("Calculating spectroscopic redshifts...")
        spec_z = (1 + cosmo_z)*(1 + v_r/c) - 1  # correct to linear order
        print(f"Spectroscopic redshifts calculated, elapsed time: {datetime.now() - start_time}")

        print("Calculating redshift-space positions...")
        rsd_dist = np.ndarray(pos.shape[0])
        for i in trange(pos.shape[0]):
            rsd_dist[i] = comoving_distance(spec_z[i])  # [cMpc/h]
        print(f"Redshift-space positions calculated, elapsed time: {datetime.now() - start_time}")

        catalogue["Pos"][:,2] = rsd_dist
        catalogue.create_dataset("z", data=spec_z, dtype="f8")

    random_catalogue_size = 100000
    print(f"Creating random catalogue of size {random_catalogue_size}...")
    create_random_catalogue(random_catalogue_size)
    print(f"Random catalogue created, elapsed time: {datetime.now() - start_time}")

    print("Plotting catalogue maps...")
    plot_catalogue_map("catalogue.hdf5", "maps/catalogue_map.png")
    plot_catalogue_map("random_catalogue.hdf5", "maps/random_catalogue_map.png")
    plot_catalogue("catalogue.hdf5", "maps/catalogue.png")
    plot_catalogue("random_catalogue.hdf5", "maps/random_catalogue.png")
    print(f"Catalogue maps plotted, elapsed time: {datetime.now() - start_time}")
