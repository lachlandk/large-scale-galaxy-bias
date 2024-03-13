import h5py
import numpy as np
from tqdm import trange
from datetime import datetime
from scipy import optimize, interpolate
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.floating_axes import FloatingAxes, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.basemap import Basemap

plt.switch_backend("agg")
plt.rcParams["axes.titlesize"] = 40
plt.rcParams["axes.labelsize"] = 40
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30

# cosmological parameters
c = 3e5  # km/s
Omega_m_0 = 0.3089
Omega_Lambda_0 = 0.6911

# cosmological functions
# create interpolating function for comoving distance
num_interp_points = 100
interp_points_r = np.ndarray(num_interp_points)
interp_points_z = np.linspace(0, 1.5, num_interp_points)
for i in range(num_interp_points):
    integration_range = np.linspace(0, interp_points_z[i], 100)
    interp_points_r[i] = np.trapz(c / (100*np.sqrt(Omega_m_0 * (1 + integration_range)**3 + Omega_Lambda_0)), integration_range)  # [cMpc/h]
comoving_distance_interp = interpolate.CubicSpline(interp_points_z, interp_points_r, extrapolate=False)  # [cMpc/h]


def comoving_distance(z):
    if isinstance(z, np.ndarray):
        return comoving_distance_interp(z.clip(0, None))
    else:
        return comoving_distance_interp(z) if z > 0 else 0

def z_at_comoving_distance(r):
    if isinstance(r, np.ndarray):
        z = np.ndarray(r.shape[0])
        for i in range(r.shape[0]):
            z[i] = optimize.root_scalar(lambda z: r[i] - comoving_distance(z), bracket=[0, 1.5]).root
    else:
        z = optimize.root_scalar(lambda z: r - comoving_distance(z), bracket=[0, 1.5]).root
    return z


def create_data_catalogue(dir, num_files, z_bins, mag_lim=19.5, mass_lim=0, save_name=None):
    filename = f"data_catalogue_r<{mag_lim}_m>{mass_lim}_z={z_bins[0]}-{z_bins[-1]}.hdf5" if save_name is None else save_name
    
    with h5py.File(f"catalogues/{filename}", "w") as file:
        # create catalogues
        for i in range(len(z_bins) - 1):
            catalogue = file.create_group(f"{z_bins[i]}<z<{z_bins[i+1]}")
            cat_pos = catalogue.create_dataset("Pos", (0, 3), maxshape=(None, 3), dtype="f8")
            cat_dist = catalogue.create_dataset("ObsDist", (0,), maxshape=(None,), dtype="f8")
            cat_spec_z = catalogue.create_dataset("SpecZ", (0,), maxshape=(None,), dtype="f8")
            cat_obs_z = catalogue.create_dataset("ObsZ", (0,), maxshape=(None,), dtype="f8")
            cat_mag = catalogue.create_dataset("ObsMagDust", (0, 5), maxshape=(None, 5), dtype="f4")

        # generate catalogues from data
        for index in trange(num_files):
            with h5py.File(f"{dir}gal_cone_01.{index}.hdf5", "r") as data:
                galaxies = data["Galaxies"]
                mag = np.array(galaxies["ObsMagDust"])  # u g r i z, magnitude with k-correction and corrected for dust extinction
                mass = np.log10(1e10 * np.array(galaxies["StellarMass"])) # [log_10(M_sol/h)]

                pos = np.array(galaxies["Pos"])  # [cMpc/h]
                vel = np.array(galaxies["Vel"])  # [km/s]

                # calculate positions in spherical coordinates
                R = np.sqrt(pos[:,0]**2 + pos[:,1]**2)  # [cMpc/h]
                r = np.sqrt(R**2 + pos[:,2]**2)  # [cMpc/h]

                # calculate radial peculiar velocity
                v_r = (vel[:,0]*pos[:,0] + vel[:,1]*pos[:,1] + vel[:,2]*pos[:,2]) / r  # [km/s]

                for i in range(len(z_bins) - 1):
                    # calculate an upper bound on the size of the bin in real space
                    lower_dist_bound = comoving_distance((1 + z_bins[i])/(1 + np.max(v_r)/c) - 1)
                    upper_dist_bound = comoving_distance((1 + z_bins[i+1])/(1 + np.min(v_r)/c) - 1) 

                    mag_filter = mag[:,2] < mag_lim  # magnitude limit in r band, 19.5 is the limit for DESI Bright Galaxy Sample
                    mass_filter = mass > mass_lim  # lower mass limit
                    data_filter = mag_filter & mass_filter & (r > lower_dist_bound) & (r < upper_dist_bound)

                    # calculate redshift space position by correcting for peculiar velocity
                    cosmo_z = z_at_comoving_distance(r[data_filter])
                    obs_z = (1 + cosmo_z)*(1 + v_r[data_filter]/c) - 1  # correct to linear order
                    
                    # apply redshift bin filter
                    z_filter = (obs_z > z_bins[i]) & (obs_z < z_bins[i+1])
                    obs_r = comoving_distance(obs_z[z_filter])  # [cMpc/h]
                    ra = 180/np.pi * np.arctan2(pos[data_filter][z_filter][:,1], pos[data_filter][z_filter][:,0])  # [degrees]
                    dec = 90 - 180/np.pi * np.arctan2(R[data_filter][z_filter], pos[data_filter][z_filter][:,2])  # [degrees]

                    # add data to catalogue
                    catalogue = file[f"{z_bins[i]}<z<{z_bins[i+1]}"]
                    cat_pos = catalogue["Pos"]
                    cat_dist = catalogue["ObsDist"]
                    cat_spec_z = catalogue["SpecZ"]
                    cat_obs_z = catalogue["ObsZ"]
                    cat_mag = catalogue["ObsMagDust"]

                    start_index = cat_pos.shape[0]
                    total_galaxies = start_index + np.count_nonzero(z_filter)
                    cat_pos.resize(total_galaxies, axis=0)
                    cat_pos[start_index:] = np.transpose([ra, dec, r[data_filter][z_filter]])
                    cat_dist.resize(total_galaxies, axis=0)
                    cat_dist[start_index:] = obs_r
                    cat_spec_z.resize(total_galaxies, axis=0)
                    cat_spec_z[start_index:] = cosmo_z[z_filter]
                    cat_obs_z.resize(total_galaxies, axis=0)
                    cat_obs_z[start_index:] = obs_z[z_filter]
                    cat_mag.resize(total_galaxies, axis=0)
                    cat_mag[start_index:] = mag[data_filter][z_filter]


def create_random_catalogue(size, data_catalogue, save_name):
    with h5py.File(f"catalogues/{save_name}", "w") as file:
        with h5py.File(f"catalogues/{data_catalogue}", "r") as data:
            for z_bin in data:
                random_catalogue = file.create_group(z_bin)
                data_catalogue = data[f"{z_bin}"]

                data_spec_z = np.array(data_catalogue["SpecZ"])
                data_obs_z = np.array(data_catalogue["ObsZ"])
                data_mag = np.array(data[f"{z_bin}"]["ObsMagDust"])

                size = size if data_spec_z.shape[0] > 0 else 0
                ra = np.random.default_rng().uniform(0 , 90, size)
                dec = 90 - 180 / np.pi * np.arccos(np.random.default_rng().uniform(0, 1, size))
                spec_z = np.random.default_rng().choice(data_spec_z, size)
                obs_z = np.random.default_rng().choice(data_obs_z, size)
                mag = np.random.default_rng().choice(data_mag, size) 
                
                random_catalogue.create_dataset("Pos", (size, 3), data=np.transpose([ra, dec, comoving_distance(spec_z)]), dtype="f8")
                random_catalogue.create_dataset("ObsDist", (size,), data=comoving_distance(obs_z), dtype="f8")
                random_catalogue.create_dataset("SpecZ", (size,), data=spec_z, dtype="f8")
                random_catalogue.create_dataset("ObsZ", (size,), data=obs_z, dtype="f8")
                random_catalogue.create_dataset("ObsMagDust", (size, 5), data=mag, dtype="f4")


def plot_catalogue(filename, save_name):
    with h5py.File(f"catalogues/{filename}", "r") as catalogue:
        pos = np.array(catalogue["Pos"])
        mag = np.array(catalogue["ObsMagDust"])
        colour = mag[:,1] - mag[:,2]

    fig = plt.figure(figsize=(50, 25))
    fig.suptitle("Catalogue Map", fontsize=40)

    # sky projection
    ax1 = fig.add_subplot(121)
    ax1.set_title("Light cone projected onto the sky")

    map = Basemap(projection="ortho", lat_0=45, lon_0=45, ax=ax1)
    map.drawmeridians(np.arange(0, 360, 30), color="white", dashes=(None, None), latmax=90)
    map.drawparallels(np.arange(-90, 90, 30), color="white", dashes=(None, None))
    map.drawmapboundary(fill_color="black")

    scatter = map.scatter(pos[:,0], pos[:,1], latlon=True, c=colour, cmap="spring", s=0.1, marker=".")

    # plane projection
    tr_scale = Affine2D().scale(np.pi/180.0, 1.0)
    transform = tr_scale + PolarAxes.PolarTransform()
    grid_locator1 = angle_helper.LocatorHMS(8)
    tick_formatter1 = angle_helper.FormatterHMS()
    grid_locator2 = MaxNLocator(3)
    grid_helper = GridHelperCurveLinear(transform, extremes=(90, 0, np.max(pos[:,2]), 0), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)
    ax2 = fig.add_subplot(122, axes_class=FloatingAxes, grid_helper=grid_helper)
    ax2.set_title("Light cone projected into 2D")
    ax2.set_facecolor("black")

    # distance axis ticks and label
    ax2.axis["left"].toggle(ticklabels=False)
    ax2.axis["right"].toggle(ticklabels=True)
    ax2.axis["right"].set_axis_direction("bottom")
    ax2.axis["right"].label.set_visible(True)
    ax2.axis["right"].label.set_text("Distance [cMpc/h]")

    # angle axis ticks and label
    ax2.axis["bottom"].major_ticklabels.set_axis_direction("top")
    ax2.axis["bottom"].label.set_axis_direction("top")
    ax2.axis["bottom"].label.set_text("RA")

    ax2.axis["top"].set_visible(False)

    aux_ax = ax2.get_aux_axes(transform)
    aux_ax.patch = ax2.patch
    ax2.patch.zorder = 0

    scatter = aux_ax.scatter(pos[:,0], pos[:,2], c=colour, cmap="spring", s=0.01, marker=".")

    # colorbar
    fig.colorbar(scatter, ax=ax2, label="g-r Colour (Observer Frame)")

    plt.savefig(f"catalogues/{save_name}")


if __name__ == "__main__":
    start_time = datetime.now()
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01/"
    files = 155

    print("Creating galaxy catalogue...")
    create_data_catalogue(lightcone_dir, files, [0.2, 0.5], mag_lim=19.5, save_name="data_catalogue.hdf5")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")

    random_catalogue_size = 1000000
    print(f"Creating random catalogue of size {random_catalogue_size}...")
    create_random_catalogue(random_catalogue_size, "data_catalogue.hdf5", "random_catalogue.hdf5")
    print(f"Random catalogue created, elapsed time: {datetime.now() - start_time}")

    # print("Plotting catalogue maps...")
    # plot_catalogue("data_catalogue.hdf5", "data_catalogue_map.png")
    # plot_catalogue("random_catalogue.hdf5", "random_catalogue_map.png")
    # print(f"Catalogue maps plotted, elapsed time: {datetime.now() - start_time}")
