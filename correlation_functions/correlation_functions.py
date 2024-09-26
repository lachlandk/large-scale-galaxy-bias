import h5py
import numpy as np
import healpy as hp
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt

from pair_counting import count_pairs


# measure the galaxy correlation function and estimate errors for a single catalogue
def measure_correlation_functions_single_realisation(file, catalogue, s_bins, resamples=1000):
    # nmu_bins = 100
    xi_bins = []
    sigma_xi_bins = []
    median_z_bins = []
    start_time = datetime.now()
    with h5py.File(f"catalogues/{file}.hdf5", "r") as catalogue_save, h5py.File(f"correlation_functions/{file}.hdf5", "a") as corrfunc_save:
        for z_bin in catalogue_save[catalogue]:
            try:
                # load corrfunc
                xi_bins.append(np.array(corrfunc_save[catalogue]["xi"]))
                sigma_xi_bins.append(np.array(corrfunc_save[catalogue]["sigma_xi"]))
                median_z_bins.append(corrfunc_save[catalogue].attrs["median_z_cos"])
                print(f"Correlation function loaded, elapsed time: {datetime.now() - start_time}")
            except (ValueError, KeyError):
                # if corrfunc is not saved, calculate it
                save = corrfunc_save[catalogue].create_group(z_bin)
                xi_bin = []
                positions = np.array(catalogue_save[catalogue][z_bin]["Pos"])
                r_lims = catalogue_save[catalogue][z_bin].attrs["r_lims"]
                ra_lims = catalogue_save[catalogue][z_bin].attrs["ra_lims"]
                dec_lims = catalogue_save[catalogue][z_bin].attrs["dec_lims"]
                median_z = catalogue_save[catalogue][z_bin].attrs["median_z_cos"]
                print(f"Galaxies in bin {z_bin}: {positions.shape[0]}")

                # get a high resolution pixelation for masking masks
                mask_nside = 2**10
                pixel_centres = hp.pix2ang(mask_nside, np.arange(hp.nside2npix(mask_nside)))

                print("Calculating correlation function...")
                # make a high resolution mask of the bin
                ra, dec = hp.pix2ang(mask_nside, np.arange(hp.nside2npix(mask_nside)), lonlat=True)
                bin_mask = ((ra > ra_lims[0]) & (ra < ra_lims[1]) & (dec > dec_lims[0]) & (dec < dec_lims[1])).astype("float")
                
                # count pairs
                DD, RR = count_pairs(positions, r_lims, bin_mask, s_bins, start_time, ra_lims=ra_lims, dec_lims=dec_lims)
                xi_bin.append(DD)
                xi_bin.append(RR)
                xi_bin.append(DD/RR - 1)
                print(f"Correlation function calculated, elapsed time: {datetime.now() - start_time}")

                # get subregions inside catalogue
                print("Estimating variance...")
                subregion_nside = 2**2
                subregion_centres = hp.pix2ang(subregion_nside, np.arange(hp.nside2npix(subregion_nside)), lonlat=True)
                subregions = np.nonzero(((subregion_centres[1] > ra_lims[0]) & (subregion_centres[1] < ra_lims[1]) & (subregion_centres[0] > dec_lims[0]) & (subregion_centres[0] < dec_lims[1])))[0]  # indices of pixels representing the subregions
                print(f"Number of subregions in catalogue: {subregions.shape[0]}")
                print(f"Approximate resolution: {hp.nside2resol(subregion_nside)*r_lims[0]} cMpc")

                # measure correlation function in each subregion
                xi_subregions = np.ndarray((subregions.shape[0], 3, s_bins.shape[0] - 1))
                for i, subregion in enumerate(subregions):
                    print(f"Starting on region with centre (ra, dec)=({hp.pix2ang(subregion_nside, subregion, lonlat=True)}) [deg]")
                    # get galaxies inside the subregion
                    galaxies_in_subregion = positions[hp.ang2pix(subregion_nside, positions[:,1], positions[:,0], lonlat=True) == subregion]
                    print(f"Galaxies in subregion: {galaxies_in_subregion.shape[0]}")

                    # make a high resolution mask of the subregion
                    subregion_mask = (hp.ang2pix(subregion_nside, pixel_centres[0], pixel_centres[1]) == subregion).astype("float")

                    # count pairs in subregion
                    DD, RR = count_pairs(galaxies_in_subregion, r_lims, subregion_mask, s_bins, start_time, nside=subregion_nside)
                    xi_subregions[i, 0] = DD
                    xi_subregions[i, 1] = RR                    
                    xi_subregions[i, 2] = DD/RR - 1

                # bootstrap to get an estimate of the error
                resampled_xi = np.ndarray((resamples, 3, s_bins.shape[0] - 1))
                for i in trange(resamples):
                    indices = np.random.default_rng().choice(np.arange(subregions.shape[0]), subregions.shape[0])
                    resampled_xi[i] = np.mean(xi_subregions[indices], axis=0)
                std_xi = np.std(resampled_xi, axis=0)

                save.create_dataset("DD", data=xi_bin[0])
                save.create_dataset("RR", data=xi_bin[1])
                save.create_dataset("xi", data=xi_bin[2])
                save.create_dataset("DD_subregions", data=xi_subregions[:, 0])
                save.create_dataset("RR_subregions", data=xi_subregions[:, 1])
                save.create_dataset("xi_subregions", data=xi_subregions[:, 2])
                save.create_dataset("sigma_DD", data=std_xi[0])
                save.create_dataset("sigma_RR", data=std_xi[1])
                save.create_dataset("sigma_xi", data=std_xi[2])
                save.create_dataset("s", data=s_bins)
                save.attrs["median_z"] = median_z

                xi_bins.append(xi_bin[2])
                sigma_xi_bins.append(std_xi[2])
                median_z_bins.append(median_z)
                
    return xi_bins, sigma_xi_bins, median_z_bins


def measure_correlation_functions(file, catalogue, s_bins):
    xi_bins = []
    sigma_xi_bins = []
    median_z_bins = []
    with h5py.File(f"catalogues/{file}.hdf5", "r") as catalogue_save, h5py.File(f"correlation_functions/{file}.hdf5", "a") as corrfunc_save:
        try:
            # load corrfunc
            for z_bin in catalogue_save[catalogue]:   
                xi_bins.append(np.array(corrfunc_save[catalogue][z_bin]["xi"]))
                sigma_xi_bins.append(np.array(corrfunc_save[catalogue][z_bin]["sigma_xi"]))
                median_z_bins.append(corrfunc_save[catalogue][z_bin].attrs["median_z_cos"])
        except (ValueError, KeyError):
            # if corrfunc is not saved, calculate it
            xi_A, sigma_A, median_z_A = measure_correlation_functions_single_realisation(f"{file}_A", "/", s_bins)
            xi_B, sigma_B, median_z_B = measure_correlation_functions_single_realisation(f"{file}_B", "/", s_bins)
            
            xi_bins = np.mean([xi_A, xi_B], axis=0)
            sigma_xi_bins = np.mean([sigma_A, sigma_B], axis=0)
            median_z_bins = np.mean([median_z_A, median_z_B], axis=0)

            for i, z_bin in enumerate(catalogue_save[catalogue]):
                save = corrfunc_save[catalogue].create_group(z_bin)
                save.create_dataset("xi", data=xi_bins[i])
                save.create_dataset("sigma_xi", data=sigma_xi_bins[i])
                save.create_dataset("s", data=s_bins)
                save.attrs["median_z"] = median_z_bins[i]


# plot the measured correlation function with subregions used for bootstrapping for a single realisation
def plot_correlation_functions(file, catalogue):
    # plot of correlation function with subsamples
    with h5py.File(f"correlation_functions/corrfunc_{file}.hdf5", "r") as corrfunc_save:
         s = np.array([corrfunc_save[catalogue][z_bin]["s"] for z_bin in corrfunc_save[catalogue]])
         xi = np.array([corrfunc_save[catalogue][z_bin]["xi"] for z_bin in corrfunc_save[catalogue]])
         sigma_xi = np.array([corrfunc_save[catalogue][z_bin]["sigma_xi"] for z_bin in corrfunc_save[catalogue]])
         xi_subregions = np.array([corrfunc_save][catalogue][z_bin]["xi_subregions"] for z_bin in corrfunc_save[catalogue])
         z = [corrfunc_save[catalogue][z_bin].attrs["median_z"] for z_bin in corrfunc_save[catalogue]]

    fig, axes = plt.subplots(2, 5, figsize=(25, 10), layout="constrained", sharex=True)
    for i, ax in enumerate(axes.flat):
        j = -1-i
        ax.plot(s[j], xi[j])
        ax.fill_between(s[j], xi[j] + sigma_xi[j], xi[j] - sigma_xi[j], alpha=0.3)
        for k in xi_subregions.shape[1]:
            ax.plot(s[j], xi_subregions[j,k], alpha=0.5)

        ax.annotate(f"$z={np.round(z[j], decimals=2)}$", (0.05, 0.9), xycoords="axes fraction", fontsize=15)

        axes.flat[0].set_ylabel("Correlation function $\\xi(r)$")
        axes.flat[5].set_ylabel("Correlation function $\\xi(r)$")
        axes.flat[5].set_xlabel("Separation $r$ [Mpc]")
        axes.flat[6].set_xlabel("Separation $r$ [Mpc]")
        axes.flat[7].set_xlabel("Separation $r$ [Mpc]")
        axes.flat[8].set_xlabel("Separation $r$ [Mpc]")
        axes.flat[9].set_xlabel("Separation $r$ [Mpc]")

    if catalogue == "/":
        fig.savefig(f"bias_evolution/bias_{file}.pdf")
    else:
        fig.savefig(f"bias_evolution/bias_{file}_{catalogue.replace('<', '_lt_').replace('.', '_')}.pdf")


if __name__ == "__main__":
    s = np.linspace(30, 100, 101)

    print(f"Calculating correlation functions for constant number density sample...")
    measure_correlation_functions("const_number_density", "/", s)

    print(f"Calculating correlation functions for constant stellar mass sample...")
    mass_bins = [(11.5, np.inf), (11, 11.5), (10.5, 11)]
    for low_mass, high_mass in mass_bins:
        measure_correlation_functions("const_stellar_mass", f"{low_mass}<m<{high_mass}", s)

    print(f"Calculating correlation functions for magnitude limited sample...")
    measure_correlation_functions("magnitude_limited", "/", s)
