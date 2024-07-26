import os
import emcee
import numpy as np
import multiprocessing

os.environ["OMP_NUM_THREADS"] = "1"


# log of the likelihood function
def log_likelihood(theta, model, args):
    _, y, sigma_y, *_ = args
    return -0.5 * np.nansum((y - model(theta, args))**2 / sigma_y**2 + np.log(2*np.pi*sigma_y**2)) 


# log of the posterior function
def log_posterior(theta, model, log_prior, args):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, args)


# sample posterior distribution and return samples and chains
def mcmc(model, log_prior, args, start, nwalkers, ndim, total_steps, burn_in_steps):
    with multiprocessing.Pool() as pool:
        initial = start + 0.1 * np.array(start) * np.random.default_rng().random((nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(model, log_prior, args), moves=emcee.moves.DEMove(), pool=pool)

        sampler.run_mcmc(initial, total_steps, progress=True)

    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)

    try:
        print(f"Autocorrelation times: {sampler.get_autocorr_time()}")
    except emcee.autocorr.AutocorrError as error:
        print(error)
    return flat_samples, samples


# plot the data with the optimal model and samples from the posterior distribution overlaid
def plot_model(ax, theta, model, args, posterior, samples):
    x, y, *_ = args
    x_smooth = np.linspace(x[0], x[-1], 100)
    ax.plot(x, y)
    ax.plot(x_smooth, model(theta, (x_smooth, *args[1:])))

    indices = np.random.default_rng().integers(posterior.shape[0], size=samples)
    for i in indices:
        ax.plot(x_smooth, model(posterior[i], (x_smooth, *args[1:])), alpha=0.1)


# plot a 1D projection of the posterior distribution
def plot_posterior_1d(ax, posterior, bins, param_index=0):
    mean = np.mean(posterior[:,param_index])
    std = np.std(posterior[:,param_index])

    ax.hist(posterior[:,param_index], bins=bins, density=True)
    ax.axvline(mean, linestyle="dashed", color="tab:orange")
    ax.axvline(mean + std, linestyle="dashed", color="tab:red")
    ax.axvline(mean - std, linestyle="dashed", color="tab:red")


# plot a 2D projection of the posterior distribution
def plot_posterior_2d(ax, posterior, bins, param_index_1, param_index_2):
    height, x_edges, y_edges, _ = ax.hist2d(posterior[:,param_index_1], posterior[:,param_index_2], bins=bins, density=True, cmap="Greys")
    
    # contours at 0.5, 1, 1.5, and 2 sigma
    # std = np.sqrt(np.std(posterior[:,param_index_1])**2 + np.std(posterior[:,param_index_2])**2)
    # ax.contour(height.transpose(), 20 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2), extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()])
    # ax.contour(height.transpose(), (0.5, 1, 1.5, 2), extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()])
    
    # print(np.sum(np.sum(height[:,] * np.diff(y_edges)) * np.diff(x_edges)))

    # scatter plot outside 2 sigma
    # ax.scatter(posterior[:,param_index_1], posterior[:,param_index_2], s=1, alpha=0.5, rasterized=True)


# plot chains
def plot_chains(ax, chains, burn_in, param_index=0):
    for i in range(chains.shape[1]):
        ax.plot(chains[:,i,param_index], alpha=0.5, rasterized=True)
    ax.axvline(burn_in, linestyle="dashed", color="black")


# plot the whole posterior distribution as a corner plot
def plot_corner(axes, posterior, bins):
    for i in range(posterior.shape[1]):
        # diagonals
        plot_posterior_1d(axes[i, i], posterior, bins, param_index=i)

        # off-diagonals
        for j in range(posterior.shape[1]):
            # plot nothing above the diagonal
            if j < i:
                axes[j, i].set_axis_off()
            elif j > i:
                plot_posterior_2d(axes[j, i], posterior, 20, i, j)
