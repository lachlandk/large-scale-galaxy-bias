import numpy as np

from cosmology import volume, z_at_comoving_distance


def number_density(distances, r_lims, ra_lims, dec_lims, subregions=10, resamples=1000):
    subregion_bins = np.linspace(r_lims[0], r_lims[1], subregions + 1)
    subregion_n_g = np.ndarray((subregions, 3))
    for i in range(subregions):
        galaxies_in_subregion = (distances >= subregion_bins[i]) & (distances <= subregion_bins[i+1])
        subregion_volume = volume(subregion_bins[i], subregion_bins[i+1], ra_lims[0], ra_lims[1], dec_lims[0], dec_lims[1])
        subregion_n_g[i,0] = np.count_nonzero(galaxies_in_subregion) / subregion_volume
        subregion_n_g[i,1] = (subregion_bins[i+1] + subregion_bins[i]) / 2
        subregion_n_g[i,2] = z_at_comoving_distance((subregion_bins[i+1] + subregion_bins[i]) / 2)

    # bootstrap to get the variance in the mean
    sorted_indices = np.arange(subregions)
    resampled_n_g = np.ndarray(resamples)
    for i in range(resamples):
        indices = np.random.default_rng().choice(sorted_indices, subregions)
        resampled_n_g[i] = np.mean(subregion_n_g[:,0][indices])
    
    return np.mean(subregion_n_g[:,0]), np.std(resampled_n_g), subregion_n_g


# def number_density_evolution(file, base, resamples):
#     with h5py.File(f"number_density/number_density_{file}.hdf5", "a") as save_file:
#         try:
#             n_g = np.array(save_file[base]["n_g"])
#             sigma_n_g = np.array(save_file[base]["sigma"])
#             median_z = np.array(save_file[base]["z"])
#         except (ValueError, KeyError):
#             # get redshift bins
#             with h5py.File(f"catalogues/{file}_A.hdf5", "r") as data_catalogue:
#                 z_bins = [z_bin for z_bin in data_catalogue[base]]
            
#             # calculate number density and variance in all redshift bins
#             n_g = np.ndarray((len(z_bins), 2))
#             sigma_n_g = np.ndarray((len(z_bins), 2))
#             median_z = np.ndarray((len(z_bins), 2))
#             for i, z_bin in enumerate(z_bins):
#                 n_g_A, sigma_n_g_A, median_z_A = number_density(f"catalogues/{file}_A.hdf5", f"{'' if base == '/' else base + '/'}{z_bin}", resamples)
#                 n_g_B, sigma_n_g_B, median_z_B = number_density(f"catalogues/{file}_B.hdf5", f"{'' if base == '/' else base + '/'}{z_bin}", resamples)
#                 n_g[i, 0] = n_g_A
#                 n_g[i, 1] = n_g_B
#                 sigma_n_g[i, 0] = sigma_n_g_A
#                 sigma_n_g[i, 1] = sigma_n_g_B
#                 median_z[i, 0] = median_z_A
#                 median_z[i, 1] = median_z_B

#             sorted_indices_A = np.argsort(median_z[:,0])
#             sorted_indices_B = np.argsort(median_z[:,1])
#             median_z = np.mean((median_z[sorted_indices_A, 0], median_z[sorted_indices_B, 1]), axis=0)
#             n_g = np.mean((n_g[sorted_indices_A, 0], n_g[sorted_indices_B, 1]), axis=0)
#             sigma_n_g = np.mean((sigma_n_g[sorted_indices_A, 0], sigma_n_g[sorted_indices_B, 1]), axis=0)

#             try:
#                 group = save_file.create_group(base)
#             except ValueError:
#                 group = save_file
#             group.create_dataset("z", data=median_z)
#             group.create_dataset("n_g", data=n_g)
#             group.create_dataset("sigma", data=sigma_n_g)                

#     return n_g, sigma_n_g, median_z


if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt

    number_density_full = []
    with h5py.File("catalogues/magnitude_limited_A.hdf5", "r") as catalogue:
        for z_bin in catalogue:
            distances = np.array(catalogue[z_bin]["ObsDist"])
            r_lims = catalogue[z_bin].attrs["r_lims"]
            ra_lims = catalogue[z_bin].attrs["ra_lims"]
            dec_lims = catalogue[z_bin].attrs["dec_lims"]

            _, _, subsamples = number_density(distances, r_lims, ra_lims, dec_lims, subregions=50)
            number_density_full.append(subsamples)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

    for z_bin in number_density_full:
        dist = z_bin[:,1]
        n_g = z_bin[:,0]
        ax.plot(dist, n_g)

    ax.set_ylabel("Comoving galaxy number density $n_g$ [cMpc$^{-3}$]")
    ax.set_xlabel("Comoving distance [cMpc]")

    fig.savefig("catalogues/number_density_evolution.pdf")
