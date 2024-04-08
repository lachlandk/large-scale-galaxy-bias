import h5py
import numpy as np
from tqdm import trange
from scipy import optimize
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology


# define cosmology for calculating the CDM correlation function
cosmo = cosmology.setCosmology("mtng", flat=True, H0=67.74, Om0=0.3089, Ob0=0.0486, sigma8=0.8159, ns=0.9667)


def chi_2(b_1, s, xi, sigma, z):
    return np.sum(norm_residuals(b_1, s, xi, sigma, z)**2)


def norm_residuals(b_1, s, xi, sigma, z):
    return (xi - b_1**2 * cosmo.correlationFunction(s, z=z))/sigma


def compute_bias(s, xi, sigma, z, resamples):
    bias_fit = optimize.minimize(chi_2, 2, args=(s, xi, sigma, z))

    # bootstrap to get the variance
    N = xi.shape[0]
    sorted_indices = np.arange(N)
    resampled_bias = np.ndarray(resamples)
    for i in trange(resamples):
        indices = np.random.default_rng().choice(sorted_indices, N)
        resampled_bias_fit = optimize.minimize(chi_2, 2, args=(s[indices], xi[indices], sigma[indices], z))
        resampled_bias[i] = resampled_bias_fit.x[0]

    return bias_fit.x[0], np.std(resampled_bias), norm_residuals(bias_fit.x[0], s, xi, sigma, z)


def bias_evolution(file, catalogue, resamples):
    with h5py.File(f"bias_evolution/bias_{file}.hdf5", "a") as bias_save:
        try:
            bias = np.array(bias_save[catalogue]["b_1"])
            z = np.array(bias_save[catalogue]["z"])
            sigma_b = np.array(bias_save[catalogue]["sigma"])
            residuals = np.array(bias_save[catalogue]["residuals"])
        except (ValueError, KeyError):
            z = []
            bias = []
            sigma_b = []
            residuals = []
            with h5py.File(f"correlation_functions/corrfunc_{file}.hdf5", "r") as corrfunc:
                for i, z_bin in enumerate(corrfunc[catalogue]):
                    low_z = float(z_bin.split("<")[0])
                    high_z = float(z_bin.split("<")[2])
                    s = np.array(corrfunc[catalogue][f"{low_z}<z<{high_z}"]["s"])
                    xi = np.array(corrfunc[catalogue][f"{low_z}<z<{high_z}"]["xi_0"])
                    sigma_xi = np.array(corrfunc[catalogue][f"{low_z}<z<{high_z}"]["sigma"])
                    median_z = corrfunc[catalogue][f"{low_z}<z<{high_z}"].attrs["median_z"]

                    b_1, sigma_b_1, xi_residuals = compute_bias(s, xi, sigma_xi, median_z, resamples)

                    z.append(median_z)
                    bias.append(b_1)
                    sigma_b.append(sigma_b_1)
                    residuals.append(xi_residuals)
                
                z = np.array(z)
                bias = np.array(bias)
                sigma_b = np.array(sigma_b)
                residuals = np.array(residuals)
            
            try:
                group = bias_save.create_group(catalogue)
            except ValueError:
                group = bias_save
            group.create_dataset("z", data=z)
            group.create_dataset("b_1", data=bias)
            group.create_dataset("sigma", data=sigma_b)
            group.create_dataset("residuals", data=residuals)

    # plot of linear matter power spectrum and correlation function
    with h5py.File(f"correlation_functions/corrfunc_{file}.hdf5", "r") as corrfunc_save:
         s = np.array([corrfunc_save[catalogue][z_bin]["s"] for z_bin in corrfunc_save[catalogue]])
         xi = np.array([corrfunc_save[catalogue][z_bin]["xi_0"] for z_bin in corrfunc_save[catalogue]])
         sigma_xi = np.array([corrfunc_save[catalogue][z_bin]["sigma"] for z_bin in corrfunc_save[catalogue]])

    fig, axes = plt.subplots(2, 5, figsize=(25, 10), layout="constrained")
    for i, ax in enumerate(axes.flat):
        j = -1-i
        ax.plot(s[j,], xi[j,])
        ax.plot(s[j,], bias[j]**2*cosmo.correlationFunction(s[j,], z[j]))
        ax.fill_between(s[j,], xi[j,] + sigma_xi[j,], xi[j,] - sigma_xi[j,], alpha=0.5)
        ax.annotate(f"$b_1={np.round(bias[j], decimals=2)}\\pm{np.round(sigma_b[j], decimals=2)}$", (0.05, 0.9), xycoords="axes fraction", fontsize=15)
        ax.annotate(f"$z={np.round(z[j], decimals=2)}$", (0.05, 0.8), xycoords="axes fraction", fontsize=15)

        inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        try:
            inset.hist(residuals[j,], bins=20, range=(np.nanmin(residuals[j,]), np.nanmax(residuals[j,])))
        except ValueError:
            pass

    if catalogue == "/":
        fig.savefig(f"bias_evolution/bias_{file}.png")
    else:
        fig.savefig(f"bias_evolution/bias_{file}_{catalogue.replace('<', '_lt_').replace('.', '_')}.png")

if __name__ == "__main__":
    # constant number density sample 
    bias_evolution("const_number_density", "/", 1000)

    # constant stellar mass sample
    bias_evolution("const_stellar_mass", "11<m<inf", 1000)
    bias_evolution("const_stellar_mass", "10.5<m<11", 1000)
    bias_evolution("const_stellar_mass", "10<m<10.5", 1000)

    # magnitude limited sample
    bias_evolution("magnitude_limited", "/", 1000)
        