import h5py
import numpy as np
from tqdm import trange
from scipy import optimize
import matplotlib.pyplot as plt
from bias_models import b_1_conserved_tracers, b_1_non_conserved_tracers, n_g as number_density


def chi_2_conserved(b_1_i, z, b_1, sigma):
    return np.sum(norm_residuals_conserved(b_1_i, z, b_1, sigma))


def norm_residuals_conserved(b_1_i, z, b_1, sigma):
    return (b_1 - b_1_conserved_tracers(z, b_1_i))/sigma


def fit_conserved(z, b_1, sigma, resamples):
    model_fit = optimize.minimize(chi_2_conserved, b_1[-1], args=(z, b_1, sigma))

    # bootstrap to get the variance
    N = b_1.shape[0]
    sorted_indices = np.arange(N)
    resampled_bias = np.ndarray(resamples)
    for i in trange(resamples):
        indices = np.random.default_rng().choice(sorted_indices, N)
        resampled_bias_fit = optimize.minimize(chi_2_conserved, 2, args=(z[indices], b_1[indices], sigma[indices]))
        resampled_bias[i] = resampled_bias_fit.x[0]

    return model_fit.x[0], np.std(resampled_bias), norm_residuals_conserved(model_fit.x[0], z, b_1, sigma)


def chi_2_number_density(params, z, n_g, sigma):
    z_star, sigma_0, alpha_1, alpha_2 = params
    return np.sum(norm_residuals_number_density(z_star, sigma_0, alpha_1, alpha_2, z, n_g, sigma))


def norm_residuals_number_density(z_star, sigma_0, alpha_1, alpha_2, z, n_g, sigma):
    return (n_g - number_density(z, n_g[-1], z_star, sigma_0, alpha_1, alpha_2))/sigma


def fit_number_density(z, n_g, sigma, resamples):
    model_fit = optimize.minimize(chi_2_number_density, (0.5, 0.2, 1, 1), args=(z, n_g, sigma), bounds=((0, None), (None, None), (None, None), (None, None)))

    # bootstrap to get the variance
    N = n_g.shape[0]
    sorted_indices = np.arange(N)
    resampled_params = np.ndarray((resamples, 4))
    for i in trange(resamples):
        indices = np.random.default_rng().choice(sorted_indices, N)
        resampled_fit = optimize.minimize(chi_2_number_density, (model_fit.x[0], model_fit.x[1], model_fit.x[2], model_fit.x[3]), args=(z[indices], n_g[indices], sigma[indices]), bounds=((0, None), (None, None), (None, None), (None, None)))
        resampled_params[i, 0] = resampled_fit.x[0]
        resampled_params[i, 1] = resampled_fit.x[1]
        resampled_params[i, 2] = resampled_fit.x[2]
        resampled_params[i, 3] = resampled_fit.x[3]

    return model_fit.x[0], model_fit.x[1], model_fit.x[2], model_fit.x[3], np.std(resampled_params[:,0]), np.std(resampled_params[:,1]), np.std(resampled_params[:,2]), np.std(resampled_params[:,3]), norm_residuals_number_density(model_fit.x[0], model_fit.x[1], model_fit.x[2], model_fit.x[3], z, n_g, sigma)


def chi_2_non_conserved(b_1_i, z, b_1, sigma, n_g_i, z_star, sigma_0, alpha_1, alpha_2):
    return np.sum(norm_residuals_non_conserved(b_1_i, z, b_1, sigma, n_g_i, z_star, sigma_0, alpha_1, alpha_2))


def norm_residuals_non_conserved(b_1_i, z, b_1, sigma, n_g_i, z_star, sigma_0, alpha_1, alpha_2):
    return (b_1 - b_1_non_conserved_tracers(z, b_1_i, n_g_i, z_star, sigma_0, alpha_1, alpha_2))/sigma


def fit_non_conserved(z, b_1, sigma, n_g_i, z_star, sigma_0, alpha_1, alpha_2, resamples):
    model_fit = optimize.minimize(chi_2_non_conserved, b_1[-1], args=(z, b_1, sigma, n_g_i, z_star, sigma_0, alpha_1, alpha_2))

    # bootstrap to get the variance
    N = b_1.shape[0]
    sorted_indices = np.arange(N)
    resampled_bias = np.ndarray(resamples)
    for i in trange(resamples):
        indices = np.random.default_rng().choice(sorted_indices, N)
        resampled_bias_fit = optimize.minimize(chi_2_non_conserved, 2, args=(z[indices], b_1[indices], sigma[indices], n_g_i, z_star, sigma_0, alpha_1, alpha_2))
        resampled_bias[i] = resampled_bias_fit.x[0]

    return model_fit.x[0], np.std(resampled_bias), norm_residuals_non_conserved(model_fit.x[0], z, b_1, sigma, n_g_i, z_star, sigma_0, alpha_1, alpha_2)


def create_fits(file, base, plot=False):
    with h5py.File(f"number_density/number_density_{file}.hdf5", "r") as number_density_save:
        z = np.array(number_density_save[base]["z"])
        n_g = np.array(number_density_save[base]["n_g"])
        sigma_n_g = np.array(number_density_save[base]["sigma"])

    # fit the number density
    z_star, sigma_0, alpha_1, alpha_2, sigma_z_star, sigma_sigma_0, sigma_alpha_1, sigma_alpha_2, residuals = fit_number_density(z, n_g, sigma_n_g, 1)
    print(z_star, sigma_0, alpha_1, alpha_2)

    # fit the bias
    with h5py.File(f"bias_evolution/bias_{file}.hdf5", "r") as measured_bias:
        z = np.array(measured_bias[base]["z"])
        bias = np.array(measured_bias[base]["b_1"])
        sigma_b = np.array(measured_bias[base]["sigma"])

    # b_1_i_conserved, b_1_i_sigma_conserved, residuals_conserved = fit_conserved(z, bias, sigma_b, 1000)
    # b_1_i_non_conserved, b_1_i_sigma_non_conserved, residuals_non_conserved = fit_non_conserved(z, bias, sigma_b, n_g[-1], z_star, sigma_0, alpha_1, alpha_2, 1000)

    if plot:
        # plot the number density fit
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

        ax.plot(z, n_g, label="Measured Number Density")
        ax.fill_between(z, n_g+sigma_n_g, n_g-sigma_n_g, alpha=0.5)
        ax.plot(np.linspace(z[-1], 0, 100), number_density(np.linspace(z[-1], 0, 100), n_g[-1], z_star, sigma_0, alpha_1, alpha_2), label="Predicted Number Density Evolution")
        
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.legend(loc="lower right")

        ax.set_title("Number Density Evolution")
        ax.set_ylabel("$n_g [\\log_{10}M_\\odot]$")
        ax.set_xlabel("$z$")

        fig.savefig(f"bias_evolution/fit_number_density_{file}{'_' + base.replace('<', '_lt_').replace('.', '_') if base != '/' else ''}.pdf")

        # plot the bias fit
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.plot(z, bias)
        ax.fill_between(z, bias+sigma_b, bias-sigma_b, alpha=0.5)
        ax.plot(np.linspace(z[-1], 0, 100), b_1_conserved_tracers(np.linspace(z[-1], 0, 100), bias[-1]))
        ax.plot(np.linspace(z[-1], 0, 100), b_1_non_conserved_tracers(np.linspace(z[-1], 0, 100), bias[-1], n_g[-1], z_star, sigma_0, alpha_1, alpha_2))

        ax.invert_xaxis()

        ax.set_ylabel("Linear bias $b_1$")
        ax.set_xlabel("Redshift $z$")

        fig.savefig(f"bias_evolution/fit_bias_{file}{'_' + base.replace('<', '_lt_').replace('.', '_') if base != '/' else ''}.pdf")

    return (z_star, sigma_0, alpha_1, alpha_2), (sigma_z_star, sigma_sigma_0, sigma_alpha_1, sigma_alpha_2)


if __name__ == "__main__":
    print("Constant Number Density")
    params_const_num, sigma_params_const_num = create_fits("const_number_density", "/")
    print("Constant Stellar Mass")
    create_fits("const_stellar_mass", "11.5<m<inf")
    create_fits("const_stellar_mass", "11<m<11.5")
    create_fits("const_stellar_mass", "10.5<m<11")
    print("Magnitude Limited")
    params_mag_lim, sigma_params_mag_lim = create_fits("const_number_density", "/")

    z = np.ndarray((10, 5))
    bias = np.ndarray((10, 5))
    sigma_b = np.ndarray((10, 5))
    with h5py.File("bias_evolution/bias_const_number_density.hdf5", "r") as file:
        z[:,0] = np.array(file["z"])
        bias[:,0] = np.array(file["b_1"])
        sigma_b[:,0] = np.array(file["sigma"])
    with h5py.File("number_density/number_density_const_number_density.hdf5", "r") as file:
        n_g_const_num = np.array(file["n_g"])
    for i, mass_bin in enumerate(("11.5<m<inf", "11<m<11.5", "10.5<m<11")):
        with h5py.File("bias_evolution/bias_const_stellar_mass.hdf5", "r") as file:
            z[:,1+i] = np.array(file[mass_bin]["z"])
            bias[:,1+i] = np.array(file[mass_bin]["b_1"])
            sigma_b[:,1+i] = np.array(file[mass_bin]["sigma"])
    with h5py.File("bias_evolution/bias_magnitude_limited.hdf5", "r") as file:
        z[:,4] = np.array(file["z"])
        bias[:,4] = np.array(file["b_1"])
        sigma_b[:,4] = np.array(file["sigma"])
    with h5py.File("number_density/number_density_magnitude_limited.hdf5", "r") as file:
        n_g_mag_lim = np.array(file["n_g"])

    # plot bias evolution with models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    model_z = np.linspace(0.5, 0, 100)
    ax1.set_title("Sample 1: Constant Number Density")
    ax1.set_ylabel("Linear bias $b_1$")
    ax1.set_xlabel("Redshift $z$")
    ax1.plot(z[:,0], bias[:,0], label="Measured")
    ax1.fill_between(z[:,0], bias[:,0]+sigma_b[:,0], bias[:,0]-sigma_b[:,0], alpha=0.5)
    ax1.plot(model_z, b_1_conserved_tracers(model_z, bias[-1,0]), label="Conserved Tracers")
    ax1.fill_between(model_z, b_1_conserved_tracers(model_z, bias[-1,0]+sigma_b[-1,0]), b_1_conserved_tracers(model_z, bias[-1,0]-sigma_b[-1,0]), alpha=0.5)
    ax1.plot(model_z, b_1_non_conserved_tracers(model_z, bias[-1,0], n_g_const_num[-1], *params_const_num), linestyle="dashed", label="Non-conserved Tracers")
    ax1.invert_xaxis()
    ax1.legend()

    model_z = np.linspace(0.8, 0, 100)
    ax2.set_title("Sample 3: Magnitude Limited")
    ax2.set_xlabel("Redshift $z$")
    ax2.plot(z[:,4], bias[:,4], label="Measured")
    ax2.fill_between(z[:,4], bias[:,4]+sigma_b[:,4], bias[:,4]-sigma_b[:,4], alpha=0.5)
    ax2.plot(model_z, b_1_conserved_tracers(model_z, bias[-1,4]), label="Conserved Tracers")
    ax2.fill_between(model_z, b_1_conserved_tracers(model_z, bias[-1,4]+sigma_b[-1,4]), b_1_conserved_tracers(model_z, bias[-1,4]-sigma_b[-1,4]), alpha=0.5)
    ax2.plot(model_z, b_1_non_conserved_tracers(model_z, bias[-1,4], n_g_const_num[-1], *params_const_num), label="Non-conserved Tracers")
    ax2.fill_between(model_z, b_1_non_conserved_tracers(model_z, bias[-1,4]+sigma_b[-1,4], n_g_const_num[-1], *params_mag_lim), b_1_non_conserved_tracers(model_z, bias[-1,4]-sigma_b[-1,4], n_g_const_num[-1], *params_mag_lim), alpha=0.5)
    ax2.invert_xaxis()
    ax2.legend()

    fig.savefig("bias_evolution/bias_fit.pdf")

    # plot measured bias evolution for all samples
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

    ax.plot(z[:,0], bias[:,0], label="Sample 1: Constant Number Density")
    ax.fill_between(z[:,0], bias[:,0]+sigma_b[:,0], bias[:,0]-sigma_b[:,0], alpha=0.5)
    ax.plot(z[:,1], bias[:,1], label="Sample 2A: Constant Stellar Mass $11.5<m<\\infty$")
    ax.fill_between(z[:,1], bias[:,1]+sigma_b[:,1], bias[:,1]-sigma_b[:,1], alpha=0.5)
    ax.plot(z[:,2], bias[:,2], label="Sample 2B: Constant Stellar Mass $11<m<11.5$")
    ax.fill_between(z[:,2], bias[:,2]+sigma_b[:,2], bias[:,2]-sigma_b[:,2], alpha=0.5)
    ax.plot(z[:,3], bias[:,3], label="Sample 2C: Constant Stellar Mass $10.5<m<11$")
    ax.fill_between(z[:,3], bias[:,3]+sigma_b[:,3], bias[:,3]-sigma_b[:,3], alpha=0.5)
    ax.plot(z[:,4], bias[:,4], label="Sample 3: Magnitude Limited")
    ax.fill_between(z[:,4], bias[:,4]+sigma_b[:,4], bias[:,4]-sigma_b[:,4], alpha=0.5)

    ax.invert_xaxis()
    ax.legend(loc="upper right")

    ax.set_ylabel("Linear bias $b_1$")
    ax.set_xlabel("Redshift $z$")

    fig.savefig(f"bias_evolution/bias_evolution.pdf")