import h5py
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

from mcmc_fit import mcmc, plot_model, plot_posterior_1d, plot_chains

plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 15

# define cosmology for calculating the CDM correlation function
cosmo = cosmology.setCosmology("mtng", flat=True, H0=67.74, Om0=0.3089, Ob0=0.0486, sigma8=0.8159, ns=0.9667)


# linear bias model for correlation function
def linear_bias(theta, args):
    b_1 = theta[0]
    s, _, _, z = args
    return b_1**2 * cosmo.correlationFunction(s, z=z)


# assume bias must be greater than one
def log_prior(theta):
    b_1 = theta[0]
    if b_1 > 1:
        return 0.0
    return -np.inf


def bias_evolution(file, catalogue, nwalkers, total_steps, burn_in_steps):
    # load correlation functions
    with h5py.File(f"correlation_functions/corrfunc_{file}.hdf5", "r") as corrfunc_save:
        s = np.array([corrfunc_save[catalogue][z_bin]["s"] for z_bin in corrfunc_save[catalogue]])
        xi = np.array([corrfunc_save[catalogue][z_bin]["xi_0"] for z_bin in corrfunc_save[catalogue]])
        sigma_xi = np.array([corrfunc_save[catalogue][z_bin]["sigma"] for z_bin in corrfunc_save[catalogue]])
        z = np.array([corrfunc_save[catalogue][z_bin].attrs["median_z_cos"] for z_bin in corrfunc_save[catalogue]])

    # open save file
    with h5py.File(f"bias_evolution/bias_{file}.hdf5", "a") as bias_save:
        # try accessing saved optimal values
        try:
            b_1 = np.array(bias_save[catalogue]["b_1"])
            sigma_b_1 = np.array(bias_save[catalogue]["sigma_b_1"])
            posteriors_all_bins = np.array(bias_save[catalogue]["posterior"])
            chains_all_bins = np.array(bias_save[catalogue]["chains"])
        # otherwise calculate them if they aren't saved
        except (ValueError, KeyError):
            posteriors_all_bins = []
            chains_all_bins = []

            # sample posterior distribution for each redshift bin
            for i in range(z.shape[0]):
                posterior, chains = mcmc(linear_bias, log_prior, (s[i], xi[i], sigma_xi[i], z[i]), 1, nwalkers, 1, total_steps, burn_in_steps)
                posteriors_all_bins.append(posterior)
                chains_all_bins.append(chains)

            # put all redshift bins in order
            sorted_indices = np.argsort(z)
            z = np.array(z)[sorted_indices]
            posteriors_all_bins = np.array(posteriors_all_bins)[sorted_indices]
            chains_all_bins = np.array(chains_all_bins)[sorted_indices]

            # calculate optimal values
            b_1 = np.mean(posteriors_all_bins, axis=1).squeeze()
            sigma_b_1 = np.std(posteriors_all_bins, axis=1).squeeze()

            # save optimal values
            # try creating a group with the catalogue name
            try:
                group = bias_save.create_group(catalogue)
            # if catalogue is "/", save to root
            except ValueError:
                group = bias_save
            group.create_dataset("z", data=z)
            group.create_dataset("b_1", data=b_1)
            group.create_dataset("sigma_b_1", data=sigma_b_1)
            group.create_dataset("posterior", data=posteriors_all_bins)
            group.create_dataset("chains", data=chains_all_bins)

    # plots
    with h5py.File(f"correlation_functions/corrfunc_{file}.hdf5", "r") as corrfunc_save:
         s = np.array([corrfunc_save[catalogue][z_bin]["s"] for z_bin in corrfunc_save[catalogue]])
         xi = np.array([corrfunc_save[catalogue][z_bin]["xi_0"] for z_bin in corrfunc_save[catalogue]])
         sigma_xi = np.array([corrfunc_save[catalogue][z_bin]["sigma"] for z_bin in corrfunc_save[catalogue]])

     # plot model with samples from posterior 
    model_fig, axes = plt.subplots(2, 5, figsize=(25, 10), layout="constrained", sharex=True)
    for i, ax in enumerate(axes.flat):
        # iterate backwards through the ax list
        j = - 1 - i

        plot_model(ax, (b_1[j],), linear_bias, (s[j], xi[j], sigma_xi[j], z[j]), posteriors_all_bins[j], samples=50)
        ax.annotate(f"$b_1={np.round(b_1[j], decimals=2)}\\pm{np.round(sigma_b_1[j], decimals=2)}$", (0.05, 0.9), xycoords="axes fraction", fontsize=15)
        ax.annotate(f"$z={np.round(z[j], decimals=2)}$", (0.05, 0.8), xycoords="axes fraction", fontsize=15)

        # plot posterior distribution in an inset
        inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        plot_posterior_1d(inset, posteriors_all_bins[j], bins=20)
        inset.set_ylabel("$p(b_1|\\xi(r))$")
        inset.set_xlabel("$b_1$")

    axes.flat[0].set_ylabel("Correlation function $\\xi(r)$")
    axes.flat[5].set_ylabel("Correlation function $\\xi(r)$")
    axes.flat[5].set_xlabel("Separation $r$ [Mpc]")
    axes.flat[6].set_xlabel("Separation $r$ [Mpc]")
    axes.flat[7].set_xlabel("Separation $r$ [Mpc]")
    axes.flat[8].set_xlabel("Separation $r$ [Mpc]")
    axes.flat[9].set_xlabel("Separation $r$ [Mpc]")

    # plot of chains
    chains_fig, axes = plt.subplots(2, 5, figsize=(25, 10), layout="constrained", sharex=True)
    for i, ax in enumerate(axes.flat):
        # iterate backwards through the ax list
        j = - 1 - i

        plot_chains(ax, chains_all_bins[j], burn_in_steps)
        ax.annotate(f"$b_1={np.round(b_1[j], decimals=2)}\\pm{np.round(sigma_b_1[j], decimals=2)}$", (0.05, 0.9), xycoords="axes fraction", fontsize=15)
        ax.annotate(f"$z={np.round(z[j], decimals=2)}$", (0.05, 0.8), xycoords="axes fraction", fontsize=15)

    axes.flat[0].set_ylabel("$b_1$")
    axes.flat[5].set_ylabel("$b_1$")
    axes.flat[5].set_xlabel("Steps")
    axes.flat[6].set_xlabel("Steps")
    axes.flat[7].set_xlabel("Steps")
    axes.flat[8].set_xlabel("Steps")
    axes.flat[9].set_xlabel("Steps")

    if catalogue == "/":
        model_fig.savefig(f"bias_evolution/bias_measurements_{file}.pdf")
        chains_fig.savefig(f"bias_evolution/bias_chains_{file}.pdf")
    else:
        model_fig.savefig(f"bias_evolution/bias_measurements_{file}_{catalogue.replace('<', '_lt_').replace('.', '_')}.pdf")
        chains_fig.savefig(f"bias_evolution/bias_chains_{file}_{catalogue.replace('<', '_lt_').replace('.', '_')}.pdf")


if __name__ == "__main__":
    # constant number density sample 
    bias_evolution("const_number_density", "/", 32, 5000, 100)

    # constant stellar mass sample
    bias_evolution("const_stellar_mass", "11.5<m<inf", 32, 5000, 100)
    bias_evolution("const_stellar_mass", "11<m<11.5", 32, 5000, 100)
    bias_evolution("const_stellar_mass", "10.5<m<11", 32, 5000, 100)

    # magnitude limited sample
    bias_evolution("magnitude_limited", "/", 32, 5000, 100)
        