import os
import sys
import ctypes
import subprocess
import numpy as np
import numba as nb
import healpy as hp
from tqdm import trange
from ispice import ispice
from scipy import integrate
from datetime import datetime
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks


# create a linear interpolation between a set of points
@nb.cfunc(nb.types.double(nb.types.double, nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double)))
def linear_interp(x, x_data, y_data):
    # if x < x_data[0] or x > x_data[9]:
    #     print(x)
    #     print(x_data[0], x_data[9])
        # raise Exception
    
    # find which subinterval contains x
    i = 1
    if x > x_data[0]:
        while x_data[i] < x:
            i += 1

    # linearly interpolate to approximate the y value
    return (y_data[i-1]*(x_data[i] - x) + y_data[i]*(x - x_data[i-1]))/(x_data[i] - x_data[i-1])


# helper function for RR integrand
@nb.cfunc(nb.types.double(nb.types.double, nb.types.double, nb.types.double))
def mu_star(r_1, s, mu):
    return (-s + s*mu**2 + mu*np.sqrt(s**2 * mu**2 - s**2 + 4*r_1**2))/(2*r_1)


# helper function for RR integrand
@nb.cfunc(nb.types.UniTuple(nb.types.double, 2)(nb.types.double, nb.types.double, nb.types.double, nb.types.double, nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double)))
def mu_lim(s, r_1, mu_min, mu_max, r_data, dNdr_data, theta_data, w_data):
    return (mu_star(r_1, s, mu_min), mu_star(r_1, s, mu_max))


# helper function for RR integrand
@nb.cfunc(nb.types.double(nb.types.double, nb.types.double, nb.types.double))
def r_2(r_1, s, mu):
    return r_1*np.sqrt(1 + 2*mu*s/r_1 + s**2/r_1**2)


# helper function for RR integrand
@nb.cfunc(nb.types.double(nb.types.double, nb.types.double, nb.types.double))
def theta(r_1, r_2, s):
    return np.arccos((r_1**2 + r_1**2 - s**2)/(2*r_1*r_2))


# integrand for calculating RR
@nb.cfunc(nb.types.double(nb.types.double, nb.types.double, nb.types.double, nb.types.double, nb.types.double, nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double)))
def integrand(mu, s, r_1, mu_min, mu_max, r_data, dNdr_data, theta_data, w_data):
    _r_2 = r_2(r_1, s, mu)
    _theta = theta(r_1, _r_2, s)
    dNdr_1 = linear_interp(r_1, r_data, dNdr_data)
    dNdr_2 = linear_interp(_r_2, r_data, dNdr_data)
    w = linear_interp(_theta, theta_data, w_data)
    return s**2/_r_2**2 * w * dNdr_1 * dNdr_2


# find RR for a set of separations
def RR_integral(r, dNdr, theta, ang_corr, s_bins, W):
    # mu_bins = np.linspace(0, 1, 2)
    r = np.copy(r)
    dNdr = np.copy(dNdr)
    theta = np.copy(theta)
    ang_corr = np.copy(ang_corr)

    # integrals = np.ndarray((len(s_bins) - 1, len(mu_bins) - 1))
    integrals = np.ndarray(len(s_bins) - 1)
    
    for i in trange(len(s_bins) - 1):
        integral = integrate.nquad(integrand, (mu_lim, (s_bins[i], s_bins[i+1]), (np.min(r), np.max(r))), args=(0, 1, r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), dNdr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ang_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))#, opts={"epsabs": 1e-3, "epsrel": 1e-3})
        integrals[i] = integral[0]
        # for j in range(len(mu_bins) - 1):
            # mu_limits = lambda s, r_1: (mu_lim(r_1, s, mu_bins[j]), mu_lim(r_1, s, mu_bins[j+1]))

            # integrals[i, j] = nquad(integrand, ((-1, 1), (s_bins[i], s_bins[i+1]), (r_min, r_max)), opts={"epsabs": 1e-3, "epsrel": 1e-3})[0]

    integrals *= 0.5 / W**2
    return integrals


# calculate average value of the survey mask over the whole sky
def W_lims(ra_min, ra_max, dec_min, dec_max):
    # assuming that the survey mask is just a top hat over the survey area
    return 1/(4*np.pi) * integrate.dblquad(lambda dec, _: np.cos(dec), ra_min*np.pi/180, ra_max*np.pi/180, np.pi/2 - dec_max*np.pi/180, np.pi/2 - dec_min*np.pi/180)[0]


# calculate average value of the top hat of a healpix pixel over the whole sky
def W_nside(nside):
    # this is just the fraction of the area of the sphere taken up by the pixel
    return hp.nside2pixarea(nside) / (4*np.pi)


# calculate dNdr as a function of radius
def radial_distribution(distances, r_lims, radial_subdivisions=10):
    subdivision_bins = np.linspace(r_lims[0], r_lims[1], radial_subdivisions + 1)  # bin edges
    subdivision_values = np.linspace(r_lims[0], r_lims[1], radial_subdivisions)  # representative values for each bin
    subdivision_dNdr = np.zeros(radial_subdivisions)
    
    # find number of galaxies in each radial subdivision
    for i in range(radial_subdivisions):
        galaxies_in_subdivision = (distances >= subdivision_bins[i]) & (distances <= subdivision_bins[i+1])
        volume = (subdivision_bins[i+1]**3 - subdivision_bins[i]**3)/3  # radial part only
        number_density = np.count_nonzero(galaxies_in_subdivision) / volume
        subdivision_dNdr[i] = number_density * subdivision_values[i]

    return subdivision_values, subdivision_dNdr


# measure the galaxy correlation function of galaxies in a given region of the sky
def count_pairs(positions, r_lims, mask, s_bins, start_time, nside=0, ra_lims=(0, 90), dec_lims=(0, 90)):
    # assume that positions are within the region
    print("Counting data pairs...")
    DD = 1/(positions.shape[0]**2) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=1, binfile=s_bins, RA1=positions[:,0], DEC1=positions[:,1], CZ1=positions[:,2], is_comoving_dist=True)["npairs"]
    print(f"Data pairs counted, elapsed time: {datetime.now() - start_time}")

    print("Calculating random pairs...")
    # calculate angular correlation function of the mask
    hp.write_map("correlation_functions/mask.fits", mask, overwrite=True, partial=True, dtype="float64")
    ispice("correlation_functions/mask.fits", "NO", corfile="correlation_functions/ang_corr_mask.txt", fits_out="NO", binpath=f"{'/'.join(sys.executable.split('/')[:-1])}/spice", verbosity=0)

    # calculate radial distribution
    r, dNdr = radial_distribution(positions[:,2], r_lims)
    np.savetxt("correlation_functions/dNdr.txt", np.transpose((r, dNdr)))
    if nside != 0:
        W = W_nside(nside)
    else:
        W = W_lims(*ra_lims, *dec_lims)
    with open("correlation_functions/param.txt", "w") as param_file:
        param_file.writelines(["\n".join([
            "dndrfile1 /home/lachlan/analytic-pair-counts/correlation_functions/dNdr.txt",
            "dndrfile2 /home/lachlan/analytic-pair-counts/correlation_functions/dNdr.txt",
            "wformat polspice",
            "wfile /home/lachlan/analytic-pair-counts/correlation_functions/ang_corr_mask.txt",
            f"W0_1 {W}",
            f"W0_2 {W}",
            f"W1W2 {W}",
            f"smin {np.min(s_bins)}",
            f"smax {np.max(s_bins)}",
            f"ns {s_bins.shape[0] - 1}",
            "nmu 1",
            "angle mid",
            "integration suave",
            "eps_rel 1e-5",
            "output_base /home/lachlan/analytic-pair-counts/correlation_functions/corrfunc"
        ]), "\n"])
    subprocess.call(["../RR_code/compute_RR", "correlation_functions/param.txt"])
    results = np.loadtxt("correlation_functions/corrfunc_mid_suave_1e-5.txt")
    RR = results[::2,5]

    # # python code
    # spice_ang_corr = np.loadtxt("correlation_functions/ang_corr_mask.txt")
    # theta = np.flip(spice_ang_corr[:,0])
    # ang_corr = np.flip(spice_ang_corr[:,2])
    # r = np.array(catalogue_save[catalogue][z_bin]["n_g"])[:,1]
    # dNdr = np.array(catalogue_save[catalogue][z_bin]["dNdr"])
    # W = catalogue_save[catalogue][z_bin].attrs["W"]
    # RR = RR_integral(r, dNdr, theta, ang_corr, s_bins, W)
    
    os.remove("correlation_functions/mask.fits")
    os.remove("correlation_functions/ang_corr_mask.txt")
    os.remove("correlation_functions/dNdr.txt")
    os.remove("correlation_functions/corrfunc_mid_suave_1e-5.txt")
    os.remove("correlation_functions/param.txt")
    print(f"Random pairs calculated, elapsed time: {datetime.now() - start_time}")

    return DD, RR
