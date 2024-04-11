import h5py
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from scipy import integrate, interpolate

# cosmological parameters
c = 3e5  # km/s
Omega_m_0 = 0.3089
Omega_Lambda_0 = 0.6911
H_0 = 67.74  # km/s/Mpc

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


def volume(r_min, r_max, ra_min, ra_max, dec_min, dec_max):
    fraction = integrate.dblquad(lambda _, theta: np.cos(theta), ra_min*np.pi/180, ra_max*np.pi/180, dec_min*np.pi/180, dec_max*np.pi/180)
    return fraction[0]*(r_max**3 - r_min**3)/3


def number_density(file, catalogue, resamples):
    subregions = 10

    with h5py.File(file, "r") as data_catalogue:
        ra_min, ra_max = data_catalogue[catalogue].attrs["ra_lims"]
        dec_min, dec_max = data_catalogue[catalogue].attrs["dec_lims"]
        median_z = data_catalogue[catalogue].attrs["median_z_cos"]
        r_max = comoving_distance(float(catalogue.split("<")[-1].split("/")[-1]))
        r_min = comoving_distance(float(catalogue.split("<")[-3].split("/")[-1]))

        r_lims = np.linspace(r_min, r_max, subregions + 1)
        subsampled_n_g = np.ndarray(subregions)
        for i in range(subregions):
            subsample_filter = (data_catalogue[catalogue]["ObsDist"] >= r_lims[i]) & (data_catalogue[catalogue]["ObsDist"] <= r_lims[i+1])
            subsample_volume = volume(r_lims[i], r_lims[i+1], ra_min, ra_max, dec_min, dec_max)
            subsampled_n_g[i] = np.count_nonzero(subsample_filter) / subsample_volume

    # bootstrap to get the variance
    sorted_indices = np.arange(subregions)
    resampled_n_g = np.ndarray(resamples)
    for i in trange(resamples):
        indices = np.random.default_rng().choice(sorted_indices, subregions)
        resampled_n_g[i] = np.mean(subsampled_n_g[indices])

    return np.mean(subsampled_n_g), np.std(resampled_n_g), median_z


def number_density_evolution(file, base, resamples):
    with h5py.File(f"number_density/number_density_{file}.hdf5", "a") as save_file:
        try:
            n_g = np.array(save_file[base]["n_g"])
            sigma_n_g = np.array(save_file[base]["sigma"])
            median_z = np.array(save_file[base]["z"])
        except (ValueError, KeyError):
            # get redshift bins
            with h5py.File(f"catalogues/{file}_A.hdf5", "r") as data_catalogue:
                z_bins = [z_bin for z_bin in data_catalogue[base]]
            
            # calculate number density and variance in all redshift bins
            n_g = np.ndarray((len(z_bins), 2))
            sigma_n_g = np.ndarray((len(z_bins), 2))
            median_z = np.ndarray((len(z_bins), 2))
            for i, z_bin in enumerate(z_bins):
                n_g_A, sigma_n_g_A, median_z_A = number_density(f"catalogues/{file}_A.hdf5", f"{'' if base == '/' else base + '/'}{z_bin}", resamples)
                n_g_B, sigma_n_g_B, median_z_B = number_density(f"catalogues/{file}_B.hdf5", f"{'' if base == '/' else base + '/'}{z_bin}", resamples)
                n_g[i, 0] = n_g_A
                n_g[i, 1] = n_g_B
                sigma_n_g[i, 0] = sigma_n_g_A
                sigma_n_g[i, 1] = sigma_n_g_B
                median_z[i, 0] = median_z_A
                median_z[i, 1] = median_z_B

            sorted_indices_A = np.argsort(median_z[:,0])
            sorted_indices_B = np.argsort(median_z[:,1])
            median_z = np.mean((median_z[sorted_indices_A, 0], median_z[sorted_indices_B, 1]), axis=0)
            n_g = np.mean((n_g[sorted_indices_A, 0], n_g[sorted_indices_B, 1]), axis=0)
            sigma_n_g = np.mean((sigma_n_g[sorted_indices_A, 0], sigma_n_g[sorted_indices_B, 1]), axis=0)

            try:
                group = save_file.create_group(base)
            except ValueError:
                group = save_file
            group.create_dataset("z", data=median_z)
            group.create_dataset("n_g", data=n_g)
            group.create_dataset("sigma", data=sigma_n_g)                

    return n_g, sigma_n_g, median_z


if __name__ == "__main__":
    n_g_const_num, sigma_n_g_const_num, z_const_num = number_density_evolution("const_number_density", "/", 1000)
    n_g_const_mass_A, sigma_n_g_const_mass_A, z_const_mass_A = number_density_evolution("const_stellar_mass", "11.5<m<inf", 1000)
    n_g_const_mass_B, sigma_n_g_const_mass_B, z_const_mass_B = number_density_evolution("const_stellar_mass", "11<m<11.5", 1000)
    n_g_const_mass_C, sigma_n_g_const_mass_C, z_const_mass_C = number_density_evolution("const_stellar_mass", "10.5<m<11", 1000)
    n_g_mag_lim, sigma_n_g_mag_lim, z_mag_lim = number_density_evolution("magnitude_limited", "/", 1000)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

    ax.plot(z_const_num, n_g_const_num, label="Sample 1: Constant Number Density")
    ax.fill_between(z_const_num, n_g_const_num+3*sigma_n_g_const_num, n_g_const_num-3*sigma_n_g_const_num, alpha=0.5)
    ax.plot(z_const_mass_A, n_g_const_mass_A, label="Sample 2A: Constant Stellar Mass $11.5<m<\\infty$")
    ax.fill_between(z_const_mass_A, n_g_const_mass_A+3*sigma_n_g_const_mass_A, n_g_const_mass_A-3*sigma_n_g_const_mass_A, alpha=0.5)
    ax.plot(z_const_mass_B, n_g_const_mass_B, label="Sample 2B: Constant Stellar Mass $11<m<11.5$")
    ax.fill_between(z_const_mass_B, n_g_const_mass_B+3*sigma_n_g_const_mass_B, n_g_const_mass_B-3*sigma_n_g_const_mass_B, alpha=0.5)
    ax.plot(z_const_mass_C, n_g_const_mass_C, label="Sample 2C: Constant Stellar Mass $10.5<m<11$")
    ax.fill_between(z_const_mass_C, n_g_const_mass_C+3*sigma_n_g_const_mass_C, n_g_const_mass_C-3*sigma_n_g_const_mass_C, alpha=0.5)
    ax.plot(z_mag_lim, n_g_mag_lim, label="Sample 3: Magnitude Limited")
    ax.fill_between(z_mag_lim, n_g_mag_lim+3*sigma_n_g_mag_lim, n_g_mag_lim-3*sigma_n_g_mag_lim, alpha=0.5)
    
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.legend(loc="lower right")

    ax.set_ylabel("Comoving Galaxy Number Density $n_g$ [cMpc$^{-3}$]")
    ax.set_xlabel("Redshift $z$")

    fig.savefig("number_density/number_density_evolution.pdf")
