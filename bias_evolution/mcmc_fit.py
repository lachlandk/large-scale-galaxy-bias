import emcee
import numpy as np


# log of the likelihood function
def log_likelihood(theta, model, args):
    _, y, sigma_y, *_ = args
    return -0.5 * np.nansum((y - model(theta, args))**2 / sigma_y**2 + np.log(2*np.pi*sigma_y**2)) 


# log of the posterior function
def log_posterior(theta, model, log_prior, args):
    return log_prior(theta) + log_likelihood(theta, model, args)


# sample posterior distribution and return samples and chains
def mcmc(model, log_prior, args, start, nwalkers, ndim, total_steps, burn_in_steps):
    initial = start + 0.1 * np.random.default_rng().random((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(model, log_prior, args))

    sampler.run_mcmc(initial, total_steps, progress=True)

    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)
    print(f"Autocorrelation times: {sampler.get_autocorr_time()}")

    return flat_samples, samples


# plot the data with the optimal model and samples from the posterior distribution overlaid
def plot_model(ax, theta, model, args, posterior, samples):
    x, y, *_ = args
    ax.plot(x, y)
    ax.plot(x, model(theta, args))

    indices = np.random.default_rng().integers(posterior.shape[0], size=samples)
    for i in indices:
        ax.plot(x, model(posterior[i], args), alpha=0.1)


# plot a 1D projection of the posterior distribution
def plot_posterior_1d(ax, posterior, bins, axis=0):
    mean = np.mean(posterior, axis=axis)
    std = np.std(posterior, axis=axis)

    ax.hist(posterior, bins=bins, density=True)
    ax.axvline(mean, linestyle="dashed", color="tab:orange")
    ax.axvline(mean + std, linestyle="dashed", color="tab:red")
    ax.axvline(mean - std, linestyle="dashed", color="tab:red")


# plot chains
def plot_chains(ax, chains, burn_in):
    # for now assume chains are 1D
    for i in range(chains.shape[1]):
        ax.plot(chains[:,i], alpha=0.5, rasterized=True)
    ax.axvline(burn_in, linestyle="dashed", color="black")


def plot_corner():
    pass
