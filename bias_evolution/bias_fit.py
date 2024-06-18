import h5py
import numpy as np
import matplotlib.pyplot as plt

from mcmc_fit import mcmc, plot_model, plot_posterior_1d, plot_chains
from bias_models import b_1_conserved_tracers


def conserved_bias_evolution(theta, args):
    b_1_i = theta[0]
    z, *_ = args
    return b_1_conserved_tracers(z, b_1_i)


def conserved_log_prior(theta):
    b_1_i = theta[0]
    if b_1_i > 1:
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
            posterior, chains = mcmc(conserved_bias_evolution, conserved_log_prior, (z, b_1, sigma_b_1), 1, nwalkers, 1, total_steps, burn_in_steps)
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


if __name__ == "__main__":
    print("Fitting bias evolution for constant number density sample")
    fit_bias_evolution_conserved("const_number_density", ".", 32, 5000, 100)

    print("Fitting bias evolution for constant stellar mass (high) sample")
    fit_bias_evolution_conserved("const_stellar_mass", "11.5<m<inf", 32, 5000, 100)
    print("Fitting bias evolution for constant stellar mass (medium) sample")
    fit_bias_evolution_conserved("const_stellar_mass", "11<m<11.5", 32, 5000, 100)
    print("Fitting bias evolution for constant stellar mass (low) sample")
    fit_bias_evolution_conserved("const_stellar_mass", "10.5<m<11", 32, 5000, 100)

    print("Fitting bias evolution for magnitude limited sample")
    fit_bias_evolution_conserved("magnitude_limited", ".", 32, 5000, 100)
