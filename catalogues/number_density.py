import numpy as np
from scipy import integrate

from cosmology import volume, z_at_comoving_distance


def number_density(distances, r_lims, ra_lims, dec_lims, subregions=10, resamples=1000):
    subregion_bins = np.linspace(r_lims[0], r_lims[1], subregions + 1)
    subregion_n_g = np.ndarray((subregions, 3))
    for i in range(subregions):
        galaxies_in_subregion = (distances >= subregion_bins[i]) & (distances <= subregion_bins[i+1])
        subregion_volume = volume(subregion_bins[i], subregion_bins[i+1], ra_lims[0], ra_lims[1], dec_lims[0], dec_lims[1])
        subregion_n_g[i,0] = np.count_nonzero(galaxies_in_subregion) / subregion_volume  # number density
        subregion_n_g[i,1] = (subregion_bins[i+1] + subregion_bins[i]) / 2  # mean comoving distance
        subregion_n_g[i,2] = z_at_comoving_distance((subregion_bins[i+1] + subregion_bins[i]) / 2)  # mean redshift

    # bootstrap to get the variance in the mean
    sorted_indices = np.arange(subregions)
    resampled_n_g = np.ndarray(resamples)
    for i in range(resamples):
        indices = np.random.default_rng().choice(sorted_indices, subregions)
        resampled_n_g[i] = np.mean(subregion_n_g[:,0][indices])
    
    return np.mean(subregion_n_g[:,0]), np.std(resampled_n_g), subregion_n_g


# calculate normalisation constant for RR integral
def W(ra_min, ra_max, dec_min, dec_max):
    return 1/(4*np.pi) * integrate.dblquad(lambda _, theta: np.cos(theta), ra_min*np.pi/180, ra_max*np.pi/180, dec_min*np.pi/180, dec_max*np.pi/180)[0]


if __name__ == "__main__":
    import h5py

    number_density_full = []
    with h5py.File("catalogues/magnitude_limited_A_old.hdf5", "a") as catalogue_save:
        catalogue = catalogue_save["/"]
        for z_bin in catalogue:
            distances = np.array(catalogue[z_bin]["ObsDist"])
            r_lims = catalogue[z_bin].attrs["r_lims"]
            ra_lims = catalogue[z_bin].attrs["ra_lims"]
            dec_lims = catalogue[z_bin].attrs["dec_lims"]

            # TODO: check if mean number density is equal to attribute calculated before 
            _, _, subsamples = number_density(distances, r_lims, ra_lims, dec_lims, subregions=50)
            number_density_full.append(subsamples)

            # calculate W for each bin
            catalogue[z_bin].attrs["W"] = W(ra_lims[0], ra_lims[1], dec_lims[0], dec_lims[1])

            try:
                del catalogue[z_bin]["n_g"]
            except KeyError:
                pass
            finally:
                catalogue[z_bin].create_dataset("n_g", data=subsamples)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

    # for z_bin in number_density_full:
    #     dist = z_bin[:,1]
    #     n_g = z_bin[:,0]
    #     ax.plot(dist, n_g)

    # ax.set_ylabel("Comoving galaxy number density $n_g$ [cMpc$^{-3}$]")
    # ax.set_xlabel("Comoving distance [cMpc]")

    # fig.savefig("catalogues/number_density_evolution.pdf")
