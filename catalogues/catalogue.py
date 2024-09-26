import h5py
import numpy as np
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist.floating_axes import FloatingAxes, GridHelperCurveLinear

from cosmology import h, comoving_distance, z_at_comoving_distance, cosmological_redshift, observed_redshift, volume


# catalogue creation functions
def select_galaxies(dir, num_files, catalogue_save, bin_name, z_lims=(0, 1.5), mag_lims=(-np.inf, 19.5), mass_lims=(-np.inf, np.inf), dec_lims=(0, 90), ra_lims=(0, 90), rsd=False):
    with h5py.File(f"catalogues/{catalogue_save}.hdf5", "a") as file:
        catalogue = file.create_group(bin_name)
        cat_pos = catalogue.create_dataset("Pos", (0, 3), maxshape=(None, 3), dtype="f8")
        cat_z = catalogue.create_dataset("z", (0,), maxshape=(None,), dtype="f8")
        cat_mag = catalogue.create_dataset("ObsMag", (0, 5), maxshape=(None, 5), dtype="f4")
        cat_mass = catalogue.create_dataset("StellarMass", (0,), maxshape=(None,), dtype="f4")

        # generate catalogues from data
        print("Grabbing properties from raw data...")
        for index in trange(num_files):
            with h5py.File(f"{dir}/gal_cone_01.{index}.hdf5", "r") as data:
                galaxies = data["Galaxies"]
                mag = np.array(galaxies["ObsMag"])  # u g r i z, magnitude with k-correction and corrected for dust extinction
                mass = np.log10(1e10 * np.array(galaxies["StellarMass"])) - np.log10(h) # [log_10(M_sol)]

                pos = np.array(galaxies["Pos"]) / h  # [cMpc]
                vel = np.array(galaxies["Vel"])  # [km/s]

            # calculate positions in spherical coordinates
            R = np.sqrt(pos[:,0]**2 + pos[:,1]**2)  # [cMpc]
            r = np.sqrt(R**2 + pos[:,2]**2)  # [cMpc]
            ra = 180/np.pi * np.arctan2(pos[:,1], pos[:,0])  # [degrees]
            dec = 90 - 180/np.pi * np.arctan2(R, pos[:,2])  # [degrees]

            mag_filter = (mag[:,2] > mag_lims[0]) & (mag[:,2] < mag_lims[1])  # magnitude limit in r band
            mass_filter = (mass > mass_lims[0]) & (mass < mass_lims[1])  # mass limits in log10(M_sol)
            dec_filter = (dec > dec_lims[0]) & (dec < dec_lims[1])
            ra_filter = (ra > ra_lims[0]) & (ra < ra_lims[1])
            if rsd:
                # create catalogue with RSDs
                # calculate radial peculiar velocity
                v_r = (vel[:,0]*pos[:,0] + vel[:,1]*pos[:,1] + vel[:,2]*pos[:,2]) / r  # [km/s]

                # calculate an upper bound on the size of the bin in real space
                lower_dist_bound = comoving_distance(cosmological_redshift(z_lims[0], np.max(v_r)))
                upper_dist_bound = comoving_distance(cosmological_redshift(z_lims[1], np.min(v_r)))
                approx_dist_filter = (r > lower_dist_bound) & (r < upper_dist_bound)
                data_filter = approx_dist_filter & mag_filter & mass_filter & dec_filter & ra_filter

                # calculate redshift space position by correcting for peculiar velocity
                obs_z = observed_redshift(z_at_comoving_distance[r[data_filter]], v_r[data_filter])

                # apply redshift bin filter
                z_filter = (obs_z > z_lims[0]) & (obs_z < z_lims[1])
                obs_r = comoving_distance(obs_z[z_filter])  # [cMpc]

                # add data to catalogue
                start_index = cat_pos.shape[0]
                total_galaxies = start_index + np.count_nonzero(z_filter)
                cat_pos.resize(total_galaxies, axis=0)
                cat_pos[start_index:] = np.transpose([ra[data_filter][z_filter], dec[data_filter][z_filter], obs_r])
                cat_z.resize(total_galaxies, axis=0)
                cat_z[start_index:] = obs_z[z_filter]
                cat_mag.resize(total_galaxies, axis=0)
                cat_mag[start_index:] = mag[data_filter][z_filter]
                cat_mass.resize(total_galaxies, axis=0)
                cat_mass[start_index:] = mass[data_filter][z_filter]

                median_z = np.median(obs_z[z_filter])
            else:
                # create catalogue without RSDs
                dist_filter = (r > comoving_distance(z_lims[0])) & (r < comoving_distance(z_lims[1]))
                data_filter = dist_filter & mag_filter & mass_filter & dec_filter & ra_filter
                
                cosmo_z = z_at_comoving_distance(r[data_filter])
            
                # add data to catalogue
                start_index = cat_pos.shape[0]
                total_galaxies = start_index + np.count_nonzero(data_filter)

                cat_pos.resize(total_galaxies, axis=0)
                cat_pos[start_index:] = np.transpose([ra[data_filter], dec[data_filter], r[data_filter]])
                cat_z.resize(total_galaxies, axis=0)
                cat_z[start_index:] = cosmo_z
                cat_mag.resize(total_galaxies, axis=0)
                cat_mag[start_index:] = mag[data_filter]
                cat_mass.resize(total_galaxies, axis=0)
                cat_mass[start_index:] = mass[data_filter]

                median_z = np.median(cosmo_z)

        r_lims = (comoving_distance(z_lims[0]), comoving_distance(z_lims[1]))
        n_g = 3 * total_galaxies / volume(*r_lims, *ra_lims, *dec_lims)

        catalogue.attrs["r_lims"] = r_lims
        catalogue.attrs["ra_lims"] = ra_lims
        catalogue.attrs["dec_lims"] = dec_lims
        catalogue.attrs["median_z"] = median_z
        catalogue.attrs["total_galaxies"] = total_galaxies
        catalogue.attrs["n_g"] = n_g
    return total_galaxies, n_g, median_z


def create_random_catalogue(multiplier, data_file, data_catalogue):
    with h5py.File(f"catalogues/{data_file}", "r+") as file:
        random_catalogue = file[data_catalogue].create_group("random")
        
        data_z = np.array(file[data_catalogue]["z"])
        ra_lims = file[data_catalogue].attrs["ra_lims"]
        dec_lims = file[data_catalogue].attrs["dec_lims"]

        size = multiplier * data_z.shape[0] if data_z.shape[0] > 0 else 0
        ra = np.random.default_rng().uniform(ra_lims[0], ra_lims[1], size)
        dec = 90 - 180 / np.pi * np.arccos(np.random.default_rng().uniform(np.cos(np.pi/180 * (90 - dec_lims[1])), np.cos(np.pi/180 * (90 - dec_lims[0])), size))
        rand_z = np.random.default_rng().choice(data_z, size)
        
        random_catalogue.create_dataset("Pos", (size, 3), data=np.transpose([ra, dec, comoving_distance(rand_z)]), dtype="f8")
        random_catalogue.create_dataset("z", (size,), data=rand_z, dtype="f8")


# plot a map of a whole sample
def plot_slice(file, catalogue):
    plt.switch_backend("agg")
    plt.rcParams["axes.titlesize"] = 50
    plt.rcParams["axes.labelsize"] = 50
    plt.rcParams["xtick.labelsize"] = 40
    plt.rcParams["ytick.labelsize"] = 40

    # load data from catalogue
    cat_pos = []
    cat_mag = []
    with h5py.File(f"catalogues/{file}.hdf5", "r") as catalogue_save:
        for z_bin in catalogue_save[catalogue]:
            cat_pos.append(np.array(catalogue_save[z_bin]["Pos"]))
            cat_mag.append(np.array(catalogue_save[z_bin]["ObsMag"]))
    pos = np.concatenate(cat_pos)
    mag = np.concatenate(cat_mag)
    colour = np.clip(mag[:,1] - mag[:,2], 0.5, 1.2)

    # plane projection (project onto the x-y plane)
    fig = plt.figure(figsize=(30, 30), layout="constrained")
    tr_scale = Affine2D().scale(np.pi/180.0, 1.0)
    transform = tr_scale + PolarAxes.PolarTransform()
    grid_locator1 = angle_helper.LocatorDMS(8)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_locator2 = MaxNLocator(3)
    grid_helper = GridHelperCurveLinear(transform, extremes=(90, 0, np.max(pos[:,2]), 0), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=None)
    
    ax = fig.add_subplot(111, axes_class=FloatingAxes, grid_helper=grid_helper)
    ax.set_title("Light cone projected into to a plane")
    ax.set_facecolor("black")

    # distance axis ticks and label
    ax.axis["left"].toggle(ticklabels=False)
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].set_axis_direction("bottom")
    ax.axis["right"].label.set_visible(True)
    ax.axis["right"].label.set_text("Real-Space Distance $r^{(r)}$ [cMpc]")
    # ax.axis["right"].label.set_text("Redshift-Space Distance $r^{(s)}$ [cMpc]")

    # angle axis ticks and label
    ax.axis["bottom"].major_ticklabels.set_axis_direction("top")
    ax.axis["bottom"].label.set_axis_direction("top")
    ax.axis["bottom"].label.set_text("Right Ascension $\\alpha$ [deg]")
    ax.axis["top"].set_visible(False)

    aux_ax1 = ax.get_aux_axes(transform)
    aux_ax1.patch = ax.patch
    ax.patch.zorder = 0
    scatter = aux_ax1.scatter(pos[:,0], pos[:,2], c=colour, cmap="bwr", s=0.01, marker=".", rasterized=True)

    # colourbar
    colourbar = fig.colorbar(scatter, ax=ax)
    colourbar.set_label("$g-r$ Colour (Observer Frame)", labelpad=15)

    if catalogue == "/":
        fig.savefig(f"catalogues/map_{file}_slice.pdf")
    else:
        fig.savefig(f"catalogues/map_{file}_{catalogue.replace('<', '_lt_').replace('.', '_')}_slice.pdf")


def plot_sky(file, catalogue):
    plt.switch_backend("agg")
    plt.rcParams["axes.titlesize"] = 50
    plt.rcParams["axes.labelsize"] = 50
    plt.rcParams["xtick.labelsize"] = 40
    plt.rcParams["ytick.labelsize"] = 40

    # load data from catalogue
    cat_pos = []
    cat_mag = []
    with h5py.File(f"catalogues/{file}.hdf5", "r") as catalogue_save:
        for z_bin in catalogue_save[catalogue]:
            cat_pos.append(np.array(catalogue_save[z_bin]["Pos"]))
            cat_mag.append(np.array(catalogue_save[z_bin]["ObsMag"]))
    pos = np.concatenate(cat_pos)
    mag = np.concatenate(cat_mag)
    colour = np.clip(mag[:,1] - mag[:,2], 0.5, 1.2)

    # sky projection (project onto the surface of a sphere)
    fig = plt.figure(figsize=(25, 25), layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_title("Light cone projected onto the sky")

    map = Basemap(projection="ortho", lat_0=35, lon_0=45, ax=ax)
    map.drawmeridians(np.arange(0, 360, 30), color="white", dashes=(None, None), latmax=90)
    map.drawparallels(np.arange(-90, 90, 30), color="white", dashes=(None, None))
    for meridian in np.arange(0, 120, 30):
        ax.annotate(f"{meridian}$^\\circ$", map(meridian + 2, -6), size=30, color="white")
    for parallel in np.arange(0, 90, 30):
        ax.annotate(f"{parallel}$^\\circ$", map(92, parallel + 3), size=30, color="white")
    ax.annotate("Right ascension $\\alpha$ [deg]", map(30.5, -11), size=30, color="white")
    ax.annotate("Declination $\\theta$ [deg]", map(78, 42), size=30, rotation=-57, color="white")
    map.drawmapboundary(fill_color="black")

    scatter = map.scatter(pos[:,0], pos[:,1], latlon=True, c=colour, cmap="bwr", s=0.1, alpha=0.2, marker=".", rasterized=True)

    # colourbar
    colourbar = fig.colorbar(scatter, ax=ax)
    colourbar.set_label("$g-r$ Colour (Observer Frame)", labelpad=15)

    if catalogue == "/":
        fig.savefig(f"catalogues/map_{file}_sky.pdf")
    else:
        fig.savefig(f"catalogues/map_{file}_{catalogue.replace('<', '_lt_').replace('.', '_')}_sky.pdf")


if __name__ == "__main__":
    lightcone_dir = "/freya/ptmp/mpa/vrs/TestRuns/MTNG/MTNG-L500-2160-A/SAM/galaxies_lightcone_01"
    files = 155

    start_time = datetime.now()
    print("Creating magnitude limited sample for mapping...")  
    total_galaxies, _, _ = select_galaxies(lightcone_dir, files, "magnitude_limited_0_lt_z_lt_0_5", "map", z_lims=(0, 0.5))
    print(f"Galaxies in catalogue: {total_galaxies}")
    plot_slice("magnitude_limited_0_lt_z_lt_0_5", "/")
    plot_sky("magnitude_limited_0_lt_z_lt_0_5", "/")
    print(f"Galaxy catalogue created, elapsed time: {datetime.now() - start_time}")
