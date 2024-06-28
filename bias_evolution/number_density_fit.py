import h5py
import numpy as np
import matplotlib.pyplot as plt

from mcmc_fit import mcmc, plot_model, plot_posterior_1d, plot_chains
from bias_models import n_g as number_density


def number_density_evolution(theta, args):
    z_star, sigma_0, alpha_1, alpha_2 = theta
    z, n_g, _ = args
    return number_density(z, n_g[0], z_star, sigma_0, alpha_1, alpha_2)


def number_density_log_priors(theta):
    z_star, sigma_0, alpha_1, alpha_2 = theta
    if z_star > 0:
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
            initial = (0.5, 0.1, 1, 1)
            posterior, chains = mcmc(number_density_evolution, number_density_log_priors, (z, n_g, sigma_n_g), initial, nwalkers, 4, total_steps, burn_in_steps)
            params = np.mean(posterior, axis=0)
            sigma_params = np.std(posterior, axis=0)

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
    # ax.annotate(f"$b_{{1,i}}={np.round(b_1_i, decimals=2)}\\pm{np.round(sigma_b_1_i, decimals=2)}$", (0.05, 0.05), xycoords="axes fraction", fontsize=15)

    # # plot posterior and chains as insets
    # posterior_inset = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
    # plot_posterior_1d(posterior_inset, posterior, 20)
    # posterior_inset.set_xlabel("$b_{1,i}$")
    # posterior_inset.set_ylabel("$p(b_{1,i}|b_1(z))$")
    # chains_inset = ax.inset_axes([0.1, 0.75, 0.4, 0.2])
    # plot_chains(chains_inset, chains, burn_in_steps)
    # chains_inset.set_xlabel("Steps")
    # chains_inset.set_ylabel("$b_{1,i}$")

    if subsample == ".":
        model_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}.pdf")
    else:
        model_fig.savefig(f"bias_evolution/number_density_evolution_{catalogue}_{subsample.replace('<', '_lt_').replace('.', '_')}.pdf")


if __name__ == "__main__":
    print("Fitting number density evolution for constant number density sample")
    fit_number_density_evolution("const_number_density", ".", 64, 5000, 100)

    print("Fitting number density evolution for constant stellar mass (high) sample")
    fit_number_density_evolution("const_stellar_mass", "11.5<m<inf", 64, 5000, 100)
    print("Fitting number density evolution for constant stellar mass (medium) sample")
    fit_number_density_evolution("const_stellar_mass", "11<m<11.5", 64, 5000, 100)
    print("Fitting number density evolution for constant stellar mass (low) sample")
    fit_number_density_evolution("const_stellar_mass", "10.5<m<11", 64, 5000, 100)

    print("Fitting number density evolution for magnitude limited sample")
    fit_number_density_evolution("magnitude_limited", ".", 64, 5000, 100)
