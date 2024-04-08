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
H_0 = 67.74  # km/s/Mpc
h = H_0/100

# cosmological functions
# create interpolating function for comoving distance
num_interp_points = 100
interp_points_r = np.ndarray(num_interp_points)
interp_points_z = np.linspace(0, 1.5, num_interp_points)
for i in range(num_interp_points):
    integration_range = np.linspace(0, interp_points_z[i], 100)
    interp_points_r[i] = np.trapz(c / (H_0*np.sqrt(Omega_m_0 * (1 + integration_range)**3 + Omega_Lambda_0)), integration_range)  # [cMpc]
comoving_distance_interp = interpolate.CubicSpline(interp_points_z, interp_points_r, extrapolate=False)  # [cMpc]


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


def observed_redshift(cosmo_z, v_r):
    return (1 + cosmo_z)*(1 + v_r/c) - 1  # correct to linear order


def cosmological_redshift(obs_z, v_r):
    return (1 + obs_z)/(1 + v_r/c) - 1  # correct to linear order


def select_galaxies(dir, num_files, save_file, save_catalogue, z_lims=(0, 1.5), mag_lims=(-np.inf, 19.5), mass_lims=(-np.inf, np.inf), dec_lims=(0, 90), ra_lims=(0, 90)):
    with h5py.File(f"catalogues/{save_file}", "a") as file:
        catalogue = file.create_group(save_catalogue)
        cat_pos = catalogue.create_dataset("Pos", (0, 3), maxshape=(None, 3), dtype="f8")
        cat_dist = catalogue.create_dataset("ObsDist", (0,), maxshape=(None,), dtype="f8")
        cat_cos_z = catalogue.create_dataset("CosZ", (0,), maxshape=(None,), dtype="f8")
        cat_obs_z = catalogue.create_dataset("ObsZ", (0,), maxshape=(None,), dtype="f8")
        cat_mag = catalogue.create_dataset("ObsMag", (0, 5), maxshape=(None, 5), dtype="f4")
        cat_mass = catalogue.create_dataset("StellarMass", (0,), maxshape=(None,), dtype="f4")

        # generate catalogues from data
        for index in trange(num_files):
            with h5py.File(f"{dir}/gal_cone_01.{index}.hdf5", "r") as data:
                galaxies = data["Galaxies"]
                mag = np.array(galaxies["ObsMag"])  # u g r i z, magnitude with k-correction and corrected for dust extinction
                mass = np.log10(1e10 * np.array(galaxies["StellarMass"])) - np.log10(h) # [log_10(M_sol)]

                pos = np.array(galaxies["Pos"])/h  # [cMpc]
                vel = np.array(galaxies["Vel"])  # [km/s]

                # calculate positions in spherical coordinates
                R = np.sqrt(pos[:,0]**2 + pos[:,1]**2)  # [cMpc]
                r = np.sqrt(R**2 + pos[:,2]**2)  # [cMpc]
                ra = 180/np.pi * np.arctan2(pos[:,1], pos[:,0])  # [degrees]
                dec = 90 - 180/np.pi * np.arctan2(R, pos[:,2])  # [degrees]

                # calculate radial peculiar velocity
                v_r = (vel[:,0]*pos[:,0] + vel[:,1]*pos[:,1] + vel[:,2]*pos[:,2]) / r  # [km/s]

                # calculate an upper bound on the size of the bin in real space
                lower_dist_bound = comoving_distance(cosmological_redshift(z_lims[0], np.max(v_r)))
                upper_dist_bound = comoving_distance(cosmological_redshift(z_lims[1], np.min(v_r)))

                approx_dist_filter = (r > lower_dist_bound) & (r < upper_dist_bound)
                mag_filter = (mag[:,2] > mag_lims[0]) & (mag[:,2] < mag_lims[1])  # magnitude limit in r band
                mass_filter = (mass > mass_lims[0]) & (mass < mass_lims[1])  # mass limits in log10(M_sol)
                dec_filter = (dec > dec_lims[0]) & (dec < dec_lims[1])
                ra_filter = (ra > ra_lims[0]) & (ra < ra_lims[1])
                data_filter = approx_dist_filter & mag_filter & mass_filter & dec_filter & ra_filter

                # calculate redshift space position by correcting for peculiar velocity
                cosmo_z = z_at_comoving_distance(r[data_filter])
                obs_z = observed_redshift(cosmo_z, v_r[data_filter])
                
                # apply redshift bin filter
                z_filter = (obs_z > z_lims[0]) & (obs_z < z_lims[1])
                obs_r = comoving_distance(obs_z[z_filter])  # [cMpc]

                # add data to catalogue
                start_index = cat_pos.shape[0]
                total_galaxies = start_index + np.count_nonzero(z_filter)
                cat_pos.resize(total_galaxies, axis=0)
                cat_pos[start_index:] = np.transpose([ra[data_filter][z_filter], dec[data_filter][z_filter], r[data_filter][z_filter]])
                cat_dist.resize(total_galaxies, axis=0)
                cat_dist[start_index:] = obs_r
                cat_cos_z.resize(total_galaxies, axis=0)
                cat_cos_z[start_index:] = cosmo_z[z_filter]
                cat_obs_z.resize(total_galaxies, axis=0)
                cat_obs_z[start_index:] = obs_z[z_filter]
                cat_mag.resize(total_galaxies, axis=0)
                cat_mag[start_index:] = mag[data_filter][z_filter]
                cat_mass.resize(total_galaxies, axis=0)
                cat_mass[start_index:] = mass[data_filter][z_filter]

        r_lims = (comoving_distance(z_lims[0]), comoving_distance(z_lims[1]))
        volume = np.pi/6 * (r_lims[1]**3 - r_lims[0]**3)
        median_z = np.median(cat_cos_z)
        catalogue.attrs["r_lims"] = r_lims
        catalogue.attrs["ra_lims"] = ra_lims
        catalogue.attrs["dec_lims"] = dec_lims
        catalogue.attrs["median_z_cos"] = median_z
        catalogue.attrs["total_galaxies"] = total_galaxies
        catalogue.attrs["number_density"] = total_galaxies / volume
    return total_galaxies, total_galaxies / volume, median_z


def create_random_catalogue(multiplier, data_file, data_catalogue):
    with h5py.File(f"catalogues/{data_file}", "r+") as file:
        random_catalogue = file[data_catalogue].create_group("random")
            
        data_cos_z = np.array(file[data_catalogue]["CosZ"])
        data_obs_z = np.array(file[data_catalogue]["ObsZ"])
        ra_lims = file[data_catalogue].attrs["ra_lims"]
        dec_lims = file[data_catalogue].attrs["dec_lims"]

        size = multiplier * data_cos_z.shape[0] if data_cos_z.shape[0] > 0 else 0
        ra = np.random.default_rng().uniform(ra_lims[0], ra_lims[1], size)
        dec = 90 - 180 / np.pi * np.arccos(np.random.default_rng().uniform(np.cos(dec_lims[1]), np.cos(dec_lims[0]), size))
        cos_z = np.random.default_rng().choice(data_cos_z, size)
        obs_z = np.random.default_rng().choice(data_obs_z, size)
        
        random_catalogue.create_dataset("Pos", (size, 3), data=np.transpose([ra, dec, comoving_distance(cos_z)]), dtype="f8")
        random_catalogue.create_dataset("ObsDist", (size,), data=comoving_distance(obs_z), dtype="f8")
        random_catalogue.create_dataset("CosZ", (size,), data=cos_z, dtype="f8")
        random_catalogue.create_dataset("ObsZ", (size,), data=obs_z, dtype="f8")


def plot_catalogue(filename):
    cat_pos = []
    cat_dist = []
    cat_mag = []
    with h5py.File(f"catalogues/{filename}", "r") as data_file:
        for catalogue in data_file:
            cat_pos.append(np.array(data_file[catalogue]["Pos"]))
            cat_dist.append(np.array(data_file[catalogue]["ObsDist"]))
            cat_mag.append(np.array(data_file[catalogue]["ObsMag"]))
    pos = np.concatenate(cat_pos)
    dist = np.concatenate(cat_dist)
    mag = np.concatenate(cat_mag)
    colour = np.clip(mag[:,1] - mag[:,2], 0.5, 1.2)

    fig = plt.figure(figsize=(50, 50))
    fig.set_layout_engine("constrained")
    fig.suptitle("Galaxy Catalogue", fontsize=50)

    # plane projection
    tr_scale = Affine2D().scale(np.pi/180.0, 1.0)
    transform = tr_scale + PolarAxes.PolarTransform()
    grid_locator1 = angle_helper.LocatorDMS(8)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_locator2 = MaxNLocator(3)
    grid_helper = GridHelperCurveLinear(transform, extremes=(90, 0, np.max(pos[:,2]), 0), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)
    
    ax1 = fig.add_subplot(221, axes_class=FloatingAxes, grid_helper=grid_helper)
    ax1.set_title("Light Cone in Real Space")
    ax1.set_facecolor("black")
    ax2 = fig.add_subplot(222, axes_class=FloatingAxes, grid_helper=grid_helper)
    ax2.set_title("Light Cone in Redshift Space")
    ax2.set_facecolor("black")

    # distance axis ticks and label
    ax1.axis["left"].toggle(ticklabels=False)
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].set_axis_direction("bottom")
    ax1.axis["right"].label.set_visible(True)
    ax1.axis["right"].label.set_text("True Distance [cMpc/h]")
    ax2.axis["left"].toggle(ticklabels=False)
    ax2.axis["right"].toggle(ticklabels=True)
    ax2.axis["right"].set_axis_direction("bottom")
    ax2.axis["right"].label.set_visible(True)
    ax2.axis["right"].label.set_text("Observed Distance [cMpc/h]")

    # angle axis ticks and label
    ax1.axis["bottom"].major_ticklabels.set_axis_direction("top")
    ax1.axis["bottom"].label.set_axis_direction("top")
    ax1.axis["bottom"].label.set_text("Right Ascension [deg]")
    ax1.axis["top"].set_visible(False)
    ax2.axis["bottom"].major_ticklabels.set_axis_direction("top")
    ax2.axis["bottom"].label.set_axis_direction("top")
    ax2.axis["bottom"].label.set_text("Right Ascension [deg]")
    ax2.axis["top"].set_visible(False)

    aux_ax1 = ax1.get_aux_axes(transform)
    aux_ax1.patch = ax1.patch
    ax1.patch.zorder = 0
    aux_ax2 = ax2.get_aux_axes(transform)
    aux_ax2.patch = ax2.patch
    ax2.patch.zorder = 0
    aux_ax1.scatter(pos[:,0], pos[:,2], c=colour, cmap="bwr", s=0.01, marker=".")
    scatter = aux_ax2.scatter(pos[:,0], dist, c=colour, cmap="bwr", s=0.01, marker=".")

    # colourbar
    colourbar = fig.colorbar(scatter, ax=ax2)
    colourbar.set_label("g-r Colour (Rest Frame)", labelpad=15)

    # sky projection
    ax3 = fig.add_subplot(223)
    ax3.set_title("Light Cone Projected onto the Sky")

    map = Basemap(projection="ortho", lat_0=35, lon_0=45, ax=ax3)
    map.drawmeridians(np.arange(0, 360, 30), color="white", dashes=(None, None), latmax=90)
    map.drawparallels(np.arange(-90, 90, 30), color="white", dashes=(None, None))
    for meridian in np.arange(0, 120, 30):
        ax3.annotate(f"{meridian}$^\\circ$", map(meridian + 2, -6), size=30, color="white")
    for parallel in np.arange(0, 90, 30):
        ax3.annotate(f"{parallel}$^\\circ$", map(92, parallel + 3), size=30, color="white")
    ax3.annotate("Right Ascension [deg]", map(32, -13), size=30, color="white")
    ax3.annotate("Declination [deg]", map(78, 44), size=30, rotation=-57, color="white")
    map.drawmapboundary(fill_color="black")

    map.scatter(pos[:,0], pos[:,1], latlon=True, c=colour, cmap="bwr", s=0.1, alpha=0.2, marker=".")

    fig.savefig(f"catalogues/{filename.split('.')[0]}.png")


if __name__ == "__main__":
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    files = 155

    start_time = datetime.now()
    print("Creating magnitude limited sample for mapping...")  
    total_galaxies = select_galaxies(lightcone_dir, files, "map_0_z_1.hdf5", "map", z_lims=(0, 0.5))
    print(f"Galaxies in catalogue: {total_galaxies}")
    plot_catalogue("map_0_z_1.hdf5")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")    
