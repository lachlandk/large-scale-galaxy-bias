import numpy as np
from scipy import integrate

from cosmology import volume, z_at_comoving_distance


# calculate number density as a function of radius and get a representative value for the whole sample
def number_density(distances, r_lims, ra_lims, dec_lims, radial_subdivisions=10, resamples=1000):
    # TODO: do this for different regions of the sky and average over
    subdivision_bins = np.linspace(r_lims[0], r_lims[1], radial_subdivisions + 1)  # bin edges
    subdivision_values_r = np.linspace(r_lims[0], ra_lims[1], radial_subdivisions)  # representative values for each bin
    subdivision_values_z = z_at_comoving_distance(subdivision_values_r)
    subdivision_n_g = np.zeros(radial_subdivisions)
    
    # find number of galaxies in each radial subdivision
    for i in range(radial_subdivisions):
        galaxies_in_subdivision = (distances >= subdivision_bins[i]) & (distances <= subdivision_bins[i+1])
        subregion_volume = volume(subdivision_bins[i], subdivision_bins[i+1], ra_lims[0], ra_lims[1], dec_lims[0], dec_lims[1])
        subdivision_n_g[i] = np.count_nonzero(galaxies_in_subdivision) / subregion_volume  # number density

    # bootstrap to get the variance in the mean
    sorted_indices = np.arange(radial_subdivisions)
    resampled_n_g = np.ndarray(resamples)
    for i in range(resamples):
        indices = np.random.default_rng().choice(sorted_indices, radial_subdivisions)
        resampled_n_g[i] = np.mean(subdivision_n_g[indices])
    
    # TODO: return a standard deviation for the radial distribution
    return np.mean(subdivision_n_g), np.std(resampled_n_g), (subdivision_n_g, subdivision_values_r, subdivision_values_z)

# calculate average value of the survey mask over the whole sky
def W(ra_min, ra_max, dec_min, dec_max):
    # assuming that the survey mask is just a top hat over the survey area
    return 1/(4*np.pi) * integrate.dblquad(lambda theta, _: np.cos(theta), ra_min*np.pi/180, ra_max*np.pi/180, dec_min*np.pi/180, dec_max*np.pi/180)[0]


def dNdr(n_g, r, W):
    # TODO: return an error on this distribution, should be just the same formula but with sigma_n_g
    return 4*np.pi*W*n_g*r**2
