import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from bias_models import b_1_conserved_tracers, b_1_non_conserved_tracers
from mcmc_fit import mcmc, plot_model, plot_posterior_1d, plot_chains, plot_corner


def conserved_bias_evolution(theta, args):
    b_1_i = theta[0]
    z, *_ = args
    return b_1_conserved_tracers(z, b_1_i)


def non_conserved_bias_evolution(theta, args):
    z_star, sigma_0, alpha_1, alpha_2 = theta
    z, b_1, _, n_g_f = args
    b_1_f = b_1[np.argmin(z)]
    return b_1_non_conserved_tracers(z, b_1_f, n_g_f, np.min(z), z_star, sigma_0, alpha_1, alpha_2)


def log_prior_conserved(theta):
    b_1_i = theta[0]
    if b_1_i > 1:
        return 0.0
    return -np.inf


def log_prior_non_conserved(theta):
    z_star, sigma_0, alpha_1, alpha_2 = theta
    if z_star > 0 and z_star < 3 and sigma_0 > 0 and sigma_0 < 3:
        return 0.0
    return -np.inf


def fit_bias_evolution_conserved(catalogue, subsample, nwalkers, total_steps, burn_in_steps):
    # load measured bias values
    with h5py.File(f"bias_evolution/bias_measurements.hdf5", "r") as measured_bias:
        z = np.array(measured_bias[catalogue][subsample]["z"])
        b_1 = np.array(measured_bias[catalogue][subsample]["b_1"])
        sigma_b_1 = np.array(measured_bias[catalogue][subsample]["sigma_b_1"])

    # open save file
    with h5py.File(f"bias_evolution/bias_evolution_conserved.hdf5", "a") as bias_save:
        # try accessing saved optimal values
        try:
            b_1_i = np.array(bias_save[catalogue][subsample]["b_1_i"])
            sigma_b_1_i = np.array(bias_save[catalogue][subsample]["sigma_b_1_i"])
            posterior = np.array(bias_save[catalogue][subsample]["posterior"])
            chains = np.array(bias_save[catalogue][subsample]["chains"])
        except (ValueError, KeyError):
            posterior, chains = mcmc(conserved_bias_evolution, log_prior_conserved, (z, b_1, sigma_b_1), 1, nwalkers, 1, total_steps, burn_in_steps)
            b_1_i = np.mean(posterior)
            sigma_b_1_i = np.std(posterior)

            # save optimal values
            group = f"{catalogue}/{subsample}" if subsample != "." else catalogue
            bias_save.create_dataset(f"{group}/b_1_i", data=b_1_i)
            bias_save.create_dataset(f"{group}/sigma_b_1_i", data=sigma_b_1_i)
            bias_save.create_dataset(f"{group}/posterior", data=posterior)
            bias_save.create_dataset(f"{group}/chains", data=chains)

    # plots
    model_fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    plot_model(ax, (b_1_i,), conserved_bias_evolution, (z, b_1, sigma_b_1), posterior, 50)
    ax.fill_between(z, b_1 - sigma_b_1, b_1 + sigma_b_1, alpha=0.2)
    ax.invert_xaxis()
    ax.set_xlabel("$z$")
    ax.set_ylabel("$b_1$")
    model_fig.suptitle(f"Bias evolution: {catalogue}/{subsample}")
    ax.annotate(f"$b_{{1,i}}={np.round(b_1_i, decimals=2)}\\pm{np.round(sigma_b_1_i, decimals=2)}$", (0.05, 0.05), xycoords="axes fraction", fontsize=15)

    # plot posterior and chains as insets
    posterior_inset = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
    plot_posterior_1d(posterior_inset, posterior, 20)
    posterior_inset.set_xlabel("$b_{1,i}$")
    posterior_inset.set_ylabel("$p(b_{1,i}|b_1(z))$")
    chains_inset = ax.inset_axes([0.1, 0.75, 0.4, 0.2])
    plot_chains(chains_inset, chains, burn_in_steps)
    chains_inset.set_xlabel("Steps")
    chains_inset.set_ylabel("$b_{1,i}$")

    if subsample == ".":
        model_fig.savefig(f"bias_evolution/bias_evolution_conserved_{catalogue}.pdf")
    else:
        model_fig.savefig(f"bias_evolution/bias_evolution_conserved_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}.pdf")


def fit_bias_evolution_non_conserved(catalogue, subsample, nwalkers, total_steps, burn_in_steps):
    # load measured bias values and number density evolution parameters
    with h5py.File(f"bias_evolution/bias_measurements.hdf5", "r") as measured_bias:
        z = np.array(measured_bias[catalogue][subsample]["z"])
        b_1 = np.array(measured_bias[catalogue][subsample]["b_1"])
        sigma_b_1 = np.array(measured_bias[catalogue][subsample]["sigma_b_1"])

    with h5py.File(f"number_density/number_density_{catalogue}.hdf5", "r") as measured_number_density:
        n_g_f = np.array(measured_number_density[subsample]["n_g"])[np.argmin(z)]

    # open save file
    with h5py.File(f"bias_evolution/bias_evolution_non_conserved.hdf5", "a") as bias_save:
        # try accessing saved optimal values
        try:
            params = np.array(bias_save[catalogue][subsample]["params"])
            sigma_params = np.array(bias_save[catalogue][subsample]["sigma_params"])
            posterior = np.array(bias_save[catalogue][subsample]["posterior"])
            chains = np.array(bias_save[catalogue][subsample]["chains"])
        except (ValueError, KeyError):
            initial = (0.3, 0.5, 0.01, -0.001)
            posterior, chains = mcmc(non_conserved_bias_evolution, log_prior_non_conserved, (z, b_1, sigma_b_1, n_g_f), initial, nwalkers, 4, total_steps, burn_in_steps)
            params = np.median(posterior, axis=0)
            sigma_params = stats.median_abs_deviation(posterior, axis=0)

            # save optimal values
            group = f"{catalogue}/{subsample}" if subsample != "." else catalogue
            bias_save.create_dataset(f"{group}/params", data=params)
            bias_save.create_dataset(f"{group}/sigma_params", data=sigma_params)
            bias_save.create_dataset(f"{group}/posterior", data=posterior)
            bias_save.create_dataset(f"{group}/chains", data=chains)

    # plots
    model_fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    plot_model(ax, params, non_conserved_bias_evolution, (z, b_1, sigma_b_1, n_g_f), posterior, 50)
    ax.fill_between(z, b_1 - sigma_b_1, b_1 + sigma_b_1, alpha=0.2)
    ax.invert_xaxis()
    ax.set_xlabel("$z$")
    ax.set_ylabel("$b_1$")
    model_fig.suptitle(f"Bias evolution: {catalogue}/{subsample}")
    ax.annotate(f"$z_\\ast={params[0]}\\pm{sigma_params[0]}$", (0.05, 0.95), xycoords="axes fraction", fontsize=15)
    ax.annotate(f"$\\sigma_0={params[1]}\\pm{sigma_params[1]}$", (0.05, 0.9), xycoords="axes fraction", fontsize=15)
    ax.annotate(f"$\\alpha_1={params[2]}\\pm{sigma_params[2]}$", (0.05, 0.85), xycoords="axes fraction", fontsize=15)
    ax.annotate(f"$\\alpha_2={params[3]}\\pm{sigma_params[3]}$", (0.05, 0.8), xycoords="axes fraction", fontsize=15)
    
    # clip posterior distribution to ignore outliers in corner plot
    clipped_posterior = np.clip(posterior, params - 5*sigma_params, params + 5*sigma_params)

    # plot posterior as corner plot
    corner_fig, axes = plt.subplots(4, 4, figsize=(10, 10), layout="constrained")
    plot_corner(axes, clipped_posterior, 20)
    axes[3, 0].set_xlabel("$z_\\ast$")
    axes[3, 1].set_xlabel("$\\sigma_0$")
    axes[3, 2].set_xlabel("$\\alpha_1$")
    axes[3, 3].set_xlabel("$\\alpha_2$")
    axes[1, 0].set_ylabel("$\\sigma_0$")
    axes[2, 0].set_ylabel("$\\alpha_1$")
    axes[3, 0].set_ylabel("$\\alpha_2$")

    # plot chains
    chains_fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    plot_chains(ax1, chains, burn_in_steps, param_index=0)
    ax1.set_ylabel("$z_\\ast$")
    plot_chains(ax2, chains, burn_in_steps, param_index=1)
    ax2.set_ylabel("$\\sigma_0$")
    plot_chains(ax3, chains, burn_in_steps, param_index=2)
    ax3.set_ylabel("$\\alpha_1$")
    plot_chains(ax4, chains, burn_in_steps, param_index=3)
    ax4.set_ylabel("$\\alpha_2$")
    ax4.set_xlabel("Steps")

    if subsample == ".":
        model_fig.savefig(f"bias_evolution/bias_evolution_non_conserved_{catalogue}.pdf")
        corner_fig.savefig(f"bias_evolution/bias_evolution_non_conserved_{catalogue}_corner.pdf")
        chains_fig.savefig(f"bias_evolution/bias_evolution_non_conserved_{catalogue}_chains.pdf")
    else:
        model_fig.savefig(f"bias_evolution/bias_evolution_non_conserved_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}.pdf")
        corner_fig.savefig(f"bias_evolution/bias_evolution_non_conserved_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}_corner.pdf")
        chains_fig.savefig(f"bias_evolution/bias_evolution_non_conserved_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}_chains.pdf")


if __name__ == "__main__":
    # print("Fitting bias evolution for constant number density sample")
    # fit_bias_evolution_conserved("const_number_density", ".", 32, 5000, 100)

    # print("Fitting bias evolution for constant stellar mass (high) sample")
    # fit_bias_evolution_conserved("const_stellar_mass", "11.5<m<inf", 32, 5000, 100)
    # print("Fitting bias evolution for constant stellar mass (medium) sample")
    # fit_bias_evolution_conserved("const_stellar_mass", "11<m<11.5", 32, 5000, 100)
    # print("Fitting bias evolution for constant stellar mass (low) sample")
    # fit_bias_evolution_conserved("const_stellar_mass", "10.5<m<11", 32, 5000, 100)

    # print("Fitting bias evolution for magnitude limited sample")
    # fit_bias_evolution_conserved("magnitude_limited", ".", 32, 5000, 100)

    print("Fitting bias evolution for constant stellar mass (high) sample")
    fit_bias_evolution_non_conserved("const_stellar_mass", "11.5<m<inf", 256, 10000, 1000)
    print("Fitting bias evolution for constant stellar mass (medium) sample")
    fit_bias_evolution_non_conserved("const_stellar_mass", "11<m<11.5", 256, 10000, 1000)
    print("Fitting bias evolution for constant stellar mass (low) sample")
    fit_bias_evolution_non_conserved("const_stellar_mass", "10.5<m<11", 256, 10000, 1000)

    print("Fitting bias evolution for magnitude limited sample")
    fit_bias_evolution_non_conserved("magnitude_limited", ".", 256, 10000, 1000)
