import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from mcmc_fit import mcmc, plot_model, plot_chains, plot_corner
from bias_models import n_g as number_density


def number_density_evolution(theta, args):
    z_star, sigma_0, alpha_1, alpha_2 = theta
    z, n_g, _ = args
    return number_density(z, n_g[np.argmax(z)], z_star, sigma_0, alpha_1, alpha_2)


def number_density_log_priors(theta):
    z_star, sigma_0, alpha_1, alpha_2 = theta
    if z_star > 0 and z_star < 3 and sigma_0 > 0 and sigma_0 < 1 and alpha_1 < 1 and alpha_2 < 1:
        return 0.0
    return -np.inf


def fit_number_density_evolution(catalogue, subsample, nwalkers, total_steps, burn_in_steps):
    # load measured number density values
    with h5py.File(f"number_density/number_density_{catalogue}.hdf5", "r") as measured_number_density:
        z = np.array(measured_number_density[subsample]["z"])
        n_g = np.array(measured_number_density[subsample]["n_g"])
        sigma_n_g = np.array(measured_number_density[subsample]["sigma"])

    # open save file
    with h5py.File(f"bias_evolution/number_density_evolution.hdf5", "a") as number_density_save:
        # try accessing saved optimal values
        try:
            params = np.array(number_density_save[catalogue][subsample]["params"])
            sigma_params = np.array(number_density_save[catalogue][subsample]["sigma_params"])
            posterior = np.array(number_density_save[catalogue][subsample]["posterior"])
            chains = np.array(number_density_save[catalogue][subsample]["chains"])
        except (ValueError, KeyError):
            initial = (0.3, 0.5, 0.01, -0.001)
            # initial = np.clip(np.random.default_rng().normal((0.3, 0.5, 0.01, -0.001), (0.01, 0.025, 0.0005, 0.00005), (nwalkers, 4)), (0, 0, -np.inf, -np.inf), (3, 1, 1, 1))
            posterior, chains = mcmc(number_density_evolution, number_density_log_priors, (z, n_g, sigma_n_g), initial, nwalkers, 4, total_steps, burn_in_steps)
            params = np.median(posterior, axis=0)
            sigma_params = stats.median_abs_deviation(posterior, axis=0)

            # save optimal values
            group = f"{catalogue}/{subsample}" if subsample != "." else catalogue
            number_density_save.create_dataset(f"{group}/params", data=params)
            number_density_save.create_dataset(f"{group}/sigma_params", data=sigma_params)
            number_density_save.create_dataset(f"{group}/posterior", data=posterior)
            number_density_save.create_dataset(f"{group}/chains", data=chains)

    # plots
    model_fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    plot_model(ax, params, number_density_evolution, (z, n_g, sigma_n_g), posterior, 50)
    ax.invert_xaxis()
    ax.set_xlabel("$z$")
    ax.set_ylabel("$n_g$")
    model_fig.suptitle(f"Number density evolution: {catalogue}/{subsample}")
    ax.annotate(f"$z_\\ast={params[0]}\\pm{sigma_params[0]}$", (0.05, 0.95), xycoords="axes fraction", fontsize=15)
    ax.annotate(f"$\\sigma_0={params[1]}\\pm{sigma_params[1]}$", (0.05, 0.9), xycoords="axes fraction", fontsize=15)
    ax.annotate(f"$\\alpha_1={params[2]}\\pm{sigma_params[2]}$", (0.05, 0.85), xycoords="axes fraction", fontsize=15)
    ax.annotate(f"$\\alpha_2={params[3]}\\pm{sigma_params[3]}$", (0.05, 0.8), xycoords="axes fraction", fontsize=15)

    # clip posterior distribution to ignore outliers
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
        model_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}.pdf")
        corner_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}_corner.pdf")
        chains_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}_chains.pdf")
    else:
        model_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}.pdf")
        corner_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}_corner.pdf")
        chains_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}_chains.pdf")


if __name__ == "__main__":
    # print("Fitting number density evolution for constant number density sample")
    # fit_number_density_evolution("const_number_density", ".", 128, 20000, 1000)

    print("Fitting number density evolution for constant stellar mass (high) sample")
    fit_number_density_evolution("const_stellar_mass", "11.5<m<inf", 256, 10000, 1000)
    print("Fitting number density evolution for constant stellar mass (medium) sample")
    fit_number_density_evolution("const_stellar_mass", "11<m<11.5", 256, 10000, 1000)
    print("Fitting number density evolution for constant stellar mass (low) sample")
    fit_number_density_evolution("const_stellar_mass", "10.5<m<11", 256, 10000, 1000)

    print("Fitting number density evolution for magnitude limited sample")
    fit_number_density_evolution("magnitude_limited", ".", 256, 10000, 1000)
