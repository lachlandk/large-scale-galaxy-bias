# %%
import sys
import h5py
import numpy as np
import healpy as hp
import numba as nb
from tqdm import trange
from ispice import ispice
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import LowLevelCallable
from scipy.interpolate import CubicSpline
from scipy.integrate import tplquad, nquad
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks

# cosmological parameters
h = 0.6774

# helper functions
def mu_lim(r_1, s, mu):
    return (-s + s*mu**2 + mu*np.sqrt(s**2 * mu**2 - s**2 + 4*r_1**2))/(2*r_1)


@nb.cfunc(nb.types.double(nb.types.double, nb.types.CPointer(nb.types.double), nb.types.CPointer(nb.types.double), nb.types.intc))
def linear_interp(x, _points, _values, n):
    points = nb.carray(_points, n)
    values = nb.carray(_values, n)
    i = 0
    while x < points[i]: i += 1
    return values[i] + (x - points[i]) * (values[i+1] - values[i])/(points[i+1] - points[i])


def make_integrand(w_theta, p_r):
    @nb.cfunc()
    def integrand(mu, s, r_1):
            w_theta = CubicSpline(np.flip(w_theta_data[:,0]), np.flip(w_theta_data[:,2]))
            p_1 = CubicSpline(p_r_data[:,0], p_r_data[:,1])
            p_2 = CubicSpline(p_r_data[:,0], p_r_data[:,1])

            def phi(r_1, r_2, s):
                return np.arccos((r_1**2 + r_1**2 - s**2)/(2*r_1*r_2))

            def r_2(r_1, s, mu):
                return r_1*np.sqrt(1 + 2*mu*s/r_1 + s**2/r_1**2)
            
            return s**2/r_2(r_1, s, mu)**2 * w_theta(phi(r_1, r_2(r_1, s, mu), s)) * p_1(r_1) * p_2(r_2(r_1, s, mu))
    
    return integrand



def RR_integral(s_bins):
    mu_bins = np.linspace(0, 1, 2)

    w_theta_data = np.loadtxt("correlation_functions/corr_R.txt")
    
    p_r_data = np.loadtxt("correlation_functions/dndr.txt")
    
    W = 0.125
    r_min = np.min(p_r_data[0, 0])
    r_max = np.max(p_r_data[-1, 0])

    # integrals = np.ndarray((len(s_bins) - 1, len(mu_bins) - 1))
    integrals = np.ndarray(len(s_bins) - 1)
    
    
    for i in trange(len(s_bins) - 1):
        integrals[i] = nquad(integrand, ((-1, 1), (s_bins[i], s_bins[i+1]), (r_min, r_max)), opts={"epsabs": 1e-3, "epsrel": 1e-3})[0]
        # for j in range(len(mu_bins) - 1):
            # mu_limits = lambda s, r_1: (mu_lim(r_1, s, mu_bins[j]), mu_lim(r_1, s, mu_bins[j+1]))

            # integrals[i, j] = nquad(integrand, ((-1, 1), (s_bins[i], s_bins[i+1]), (r_min, r_max)), opts={"epsabs": 1e-3, "epsrel": 1e-3})[0]
            # integrals[i, j] = tplquad(integrand, r_min, r_max, s_min, s_max, mu_min, mu_max, epsabs=1e-5, epsrel=1e-5)[0] / (2*W**2)
    integrals *= 1/(4*W**2)
    np.savetxt("correlation_functions/RR_anal_test.txt")


def radial_distribution(data_catalogue):
    start_time = datetime.now()
    
    print("Calculating radial distribution...")
    with h5py.File(f"catalogues/{data_catalogue}", "r") as catalogue:
        for z_bin in catalogue:
            dist = np.array(catalogue[z_bin]["Pos"])[:,2]/h
            print(f"Galaxies in bin {z_bin}: {dist.shape[0]}")

            p, bins = np.histogram(dist, bins="fd", density=True)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            fig.suptitle("Radial Distribution Function")

            x = np.linspace(np.min(bins), np.max(bins), 1000)
            ax.plot(x, [linear_interp(x_i, bins[:-1].ctypes.data, p.ctypes.data, p.shape[0]) for x_i in x])
            
            ax.scatter(bins[:-1], p)
            ax.set_xlabel("$r$")
            ax.set_ylabel("$p(r)$")

            fig.savefig("correlation_functions/test_radial_distribution.png")

            np.savetxt("correlation_functions/dndr.txt", np.transpose([bins[:-1], p]))
    print(f"Done calculating radial distribution, time elapsed: {datetime.now() - start_time}")  


def angular_correlation_function(data_catalogue, save_name=None):
    start_time = datetime.now()

    print("Calculating angular correlation function...")
    filename = f"correlation_functions/w_theta_{data_catalogue.split('.')[0]}.hdf5"
    try:
        raise FileNotFoundError
        # with h5py.File(filename, "r") as data_pairs:
            # for z_bin in data_pairs:
                # z_bins.append(z_bin)
                # DD_pairs.append(np.array(data_pairs[z_bin]["npairs"]))
    except FileNotFoundError:
        with h5py.File(f"catalogues/{data_catalogue}", "r") as catalogue:
            for z_bin in catalogue:
                # save = data_pairs.create_group(z_bin)
                pos = np.array(catalogue[z_bin]["Pos"])[:,0:2]
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")

                nside = 2**10  # map resolution = 12 * nside**2

                theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
                mask = ((theta < 0.5*np.pi) & (phi < 0.5*np.pi)).astype("float")

                map = np.zeros(hp.nside2npix(nside))
                for galaxy in pos:
                    pixel = hp.ang2pix(nside, galaxy[0], galaxy[1], lonlat=True)
                    map[pixel] += 1

                hp.write_map("correlation_functions/map.fits", map, overwrite=True, partial=True)
                hp.write_map("correlation_functions/mask.fits", mask, overwrite=True, partial=True)

                fig = plt.figure(figsize=(10, 10))
                hp.orthview(map, rot=(45, 35, 0), half_sky=True, hold=True)
                fig.savefig("correlation_functions/test_pixellated_map.png")

                ispice("correlation_functions/map.fits", "NO", corfile="correlation_functions/corr_D.txt", fits_out="NO", binpath=f"{'/'.join(sys.executable.split('/')[:-1])}/spice")
                ispice("correlation_functions/map.fits", "NO", mapfile2="correlation_functions/mask.fits", corfile="correlation_functions/corr_DR.txt", fits_out="NO", binpath=f"{'/'.join(sys.executable.split('/')[:-1])}/spice")
                ispice("correlation_functions/mask.fits", "NO", corfile="correlation_functions/corr_R.txt", fits_out="NO", binpath=f"{'/'.join(sys.executable.split('/')[:-1])}/spice")

                # averages
                phi = np.linspace(0, 2*np.pi, 1000)
                theta = np.linspace(0, np.pi, 1000)

                inners = np.ndarray(1000)
                for i in range(1000):
                    inners[i] = np.trapz(map[hp.ang2pix(nside, theta, phi[i])]*np.sin(theta), theta)
                    # inners[i] = np.trapz(mask[hp.ang2pix(nside, theta, phi[i])]*mask[hp.ang2pix(nside, theta, phi[i])]*np.sin(theta), theta)
                W = 1/(4*np.pi)*np.trapz(inners, phi)
                print(W)

                spice_R = np.loadtxt("correlation_functions/corr_R.txt")
                spice_D = np.loadtxt("correlation_functions/corr_D.txt")
                spice_DR = np.loadtxt("correlation_functions/corr_DR.txt")
                theta = np.array(spice_R[:,0])
                corr_R = np.array(spice_R[:,2])
                corr_D = np.array(spice_D[:,2])
                corr_DR = np.array(spice_DR[:,2])

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                fig.suptitle("Angular Correlation Function")


                ax.plot(theta * 180/np.pi, corr_D)
                ax.plot(theta * 180/np.pi, corr_R, linestyle="dashed")
                ax.plot(theta * 180/np.pi, corr_DR, linestyle="dotted")
                ax.set_xlabel("$\\theta$")
                ax.set_ylabel("$\\omega(\\theta)$")
                
                ax.set_xlim(0, 180)
                ax.set_ylim(0, 0.2)
                
                fig.savefig("correlation_functions/test_corrfunc_spice.png")
                
    print(f"Bruh, elapsed time: {datetime.now() - start_time}")
    



def correlation_function(data_catalogue, random_catalogue, s_bins, rsd=True, save_name=None, test=False):
    start_time = datetime.now()

    nmu_bins = 100
    filename = f"correlation_functions/corrfunc_{data_file}"
    with h5py.File(filename, "a") as save_file:
        if f"{catalogue}/xi_0" in save_file:
            xi = np.array(save_file[catalogue]["xi_0"])
            sigma = np.array(save_file[catalogue]["sigma"])
            median_z = save_file[catalogue].attrs["median_z_cos"]
            print(f"Correlation function loaded, elapsed time: {datetime.now() - start_time}")
        else:
            with h5py.File(f"catalogues/{data_file}", "r") as sample:
                d_pos = np.array(sample[catalogue]["Pos"])
                d_ra = d_pos[:,0]
                d_dec = d_pos[:,1]
                d_dist = d_pos[:,2]
                r_pos = np.array(sample[f"{catalogue}/random"]["Pos"])
                r_ra = r_pos[:,0]
                r_dec = r_pos[:,1]
                r_dist = r_pos[:,2]
                median_z = sample[catalogue].attrs["median_z_cos"]

    print("Counting data pairs...")
    filename = f"correlation_functions/DD_pair_counts_{data_catalogue.split('.')[0]}.hdf5"
    z_bins = []
    DD_pairs = []
    try:
        with h5py.File(filename, "r") as data_pairs:
            for z_bin in data_pairs:
                z_bins.append(z_bin)
                DD_pairs.append(np.array(data_pairs[z_bin]["npairs"]))
    except FileNotFoundError:
        with h5py.File(filename, "w") as data_pairs, h5py.File(f"catalogues/{data_catalogue}", "r") as catalogue:
            for z_bin in catalogue:
                save = data_pairs.create_group(z_bin)
                pos = np.array(catalogue[z_bin]["Pos"])
                dist = np.array(catalogue[z_bin]["ObsDist"])/h if rsd == True else pos[:,2]/h
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")
                pairs = (1/(pos.shape[0]**2)) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=pos[:,0], DEC1=pos[:,1], CZ1=dist, is_comoving_dist=True)["npairs"]
                save.create_dataset("npairs", data=pairs)
                DD_pairs.append(pairs)
    print(f"Data pairs counted, elapsed time: {datetime.now() - start_time}")
    
    print("Counting random pairs...")
    filename = f"correlation_functions/RR_pair_counts_{random_catalogue.split('.')[0]}.hdf5"
    RR_pairs = []
    try:
        with h5py.File(filename, "r") as random_pairs:
            for z_bin in random_pairs:
                RR_pairs.append(np.array(random_pairs[z_bin]["npairs"]))
    except FileNotFoundError:
        with h5py.File(filename, "w") as random_pairs, h5py.File(f"catalogues/{random_catalogue}", "r") as catalogue:
            for z_bin in catalogue:
                save = random_pairs.create_group(z_bin)
                pos = np.array(catalogue[z_bin]["Pos"])
                dist = np.array(catalogue[z_bin]["ObsDist"])/h if rsd == True else pos[:,2]/h
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")
                pairs = (1/(pos.shape[0]**2)) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=pos[:,0], DEC1=pos[:,1], CZ1=dist, is_comoving_dist=True)["npairs"]
                save.create_dataset("npairs", data=pairs)
                RR_pairs.append(pairs)
    print(f"Random pairs counted, elapsed time: {datetime.now() - start_time}")
    
    print("Counting data-random pairs...")
    filename = f"correlation_functions/DR_pair_counts_{data_catalogue.split('.')[0]}_{random_catalogue.split('.')[0]}.hdf5"
    DR_pairs = []
    try:
        with h5py.File(filename, "r") as data_random_pairs:
            mu = data_random_pairs.attrs["mu"]
            for z_bin in data_random_pairs:
                DR_pairs.append(np.array(data_random_pairs[z_bin]["npairs"]))

    except FileNotFoundError:
        with h5py.File(filename, "w") as data_random_pairs, h5py.File(f"catalogues/{data_catalogue}", "r") as data, h5py.File(f"catalogues/{random_catalogue}", "r") as random:
            for z_bin in data:
                save = data_random_pairs.create_group(z_bin)
                d_pos = np.array(data[z_bin]["Pos"])
                d_dist = np.array(data[z_bin]["ObsDist"])/h if rsd == True else d_pos[:,2]/h
                r_pos = np.array(random[z_bin]["Pos"])
                r_dist = np.array(random[z_bin]["ObsDist"])/h if rsd == True else r_pos[:,2]/h
                print(f"Galaxies in bin {z_bin}: Data={d_pos.shape[0]}, Random={r_pos.shape[0]}")
                count = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_pos[:,0], DEC1=d_pos[:,1], CZ1=d_dist, RA2=r_pos[:,0], DEC2=r_pos[:,1], CZ2=r_dist, is_comoving_dist=True)
                mu = count["mumax"][0:nmu_bins]
                pairs = (1/(d_pos.shape[0] * r_pos.shape[0])) * count["npairs"]
                dataset = save.create_dataset("npairs", data=pairs)
                dataset.attrs["ndata"] = d_pos.shape[0]
                dataset.attrs["nrandom"] = r_pos.shape[0]
                data_random_pairs.attrs["mu"] = mu
                DR_pairs.append(pairs)
    print(f"Data-random pairs counted, elapsed time: {datetime.now() - start_time}")


    if test:
        RR_counts = np.loadtxt(open("correlation_functions/RR_mid_suave_1e-5.txt").readlines()[:10000])
        DR_counts = np.loadtxt(open("correlation_functions/RR_mid_suave_1e-5.txt").readlines()[:10000])

        start_time = datetime.now()
        # print("Calculating RR")
        # RR_int = RR_integral(s_bins)[:,0]
        # print(f"Finished, elapsed time: {datetime.now() - start_time}")

        # print(s_bins)
        # print(RR_counts[:,1])
        
        RR = RR_counts[:,5] * 2
        DD = DD_pairs[0]
        DR = DR_counts[:,5] * 2
        # xi_mu = (DD - 2*DR + RR) / RR
        xi_mu = DD/RR - 1

        xi = np.ndarray(len(s_bins) - 1)
        RR_1d = np.ndarray(len(s_bins) - 1)
        RR_1d_2 = np.ndarray(len(s_bins) - 1)
        RR_1d_3 = np.ndarray(len(s_bins) - 1)
        for j in range(len(s_bins) - 1):
            k = j*nmu_bins
            xi[j] = np.trapz(xi_mu[k:k+nmu_bins], mu)
            RR_1d[j] = np.trapz(DD[k:k+nmu_bins], mu)
            RR_1d_2[j] = np.trapz(RR[k:k+nmu_bins], mu)
            RR_1d_3[j] = np.trapz(RR_pairs[0][k:k+nmu_bins], mu)

        # return RR_1d, RR_1d_2, RR_1d_3


    else:
        # calculate correlation function
        print("Calculating correlation function...")
        corrfunc = []
        with h5py.File(f"catalogues/{data_catalogue}", "r") as data, h5py.File(f"catalogues/{random_catalogue}", "r") as random:
                for i, z_bin in enumerate(data):
                    corrfunc.append((DD_pairs[i] - 2*DR_pairs[i] + RR_pairs[i]) / RR_pairs[i])
        print(f"Correction function calculated, elapsed time: {datetime.now() - start_time}")

        xi = []
        for i in range(len(z_bins)):
            xi_multipoles = np.ndarray((5, len(s_bins) - 1))
            for j in range(len(s_bins) - 1):
                k = nmu_bins*j
                xi_multipoles[0, j] = np.trapz(corrfunc[i][k:k+nmu_bins], mu)  # monopole
                xi_multipoles[1, j] = 3 * np.trapz(corrfunc[i][k:k+nmu_bins] * mu, mu)  # dipole
                xi_multipoles[2, j] = 5 * np.trapz(corrfunc[i][k:k+nmu_bins] * 0.5 * (3*mu**2 - 1), mu)  # quadrupole
                xi_multipoles[3, j] = 7 * np.trapz(corrfunc[i][k:k+nmu_bins] * 0.5 * (5*mu**3 - 3*mu), mu)  # octupole
                xi_multipoles[4, j] = 9 * np.trapz(corrfunc[i][k:k+nmu_bins] * 0.125 * (35*mu**4 - 30*mu**2 + 3), mu)  # hexadecapole
            xi.append(xi_multipoles)

    if save_name is not None:
        with h5py.File(f"correlation_functions/{save_name}", "w") as save_file:
            save_file.create_dataset("xi", data=xi)
    return xi, z_bins


if __name__ == "__main__":
    s = np.linspace(0.1, 200, 101)

    radial_distribution("data_bose_r_19_A.hdf5")

    # xi_A, z_bins = correlation_function("random_bose_r_19_A_small.hdf5", "random_bose_r_19_A.hdf5", s, rsd=False)
    # # xi_B, z_bins = correlation_function("data_bose_r_19_B.hdf5", "random_bose_r_19_B.hdf5", s, rsd=False)
    # # xi = [(xi_A_bin + xi_B_bin) / 2 for xi_A_bin, xi_B_bin in zip(xi_A, xi_B)]

    # xi_test, z_bins = correlation_function("random_bose_r_19_A_small.hdf5", "random_bose_r_19_A.hdf5", s, rsd=False, test=True)
    # # RR, RR_2, RR_3 = correlation_function("random_bose_r_19_A_small.hdf5", "random_bose_r_19_A.hdf5", s, rsd=False, test=True)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # fig.suptitle("Correlation Function in Real Space")

    # ax.plot(s[:-1], s[:-1]**2 * xi_A[0][0])
    # ax.plot(s[:-1], s[:-1]**2 * xi_test)
    # # ax.plot(s[:-1], RR)
    # # ax.plot(s[:-1], RR_2/RR_3, linestyle="dashed")
    # # ax.plot(s[:-1], RR_3, linestyle="dotted")
    # # ax.set_xlabel("$r$ [cMpc]")
    # # ax.set_ylabel("$r^2\\xi(r)$ [(cMpc/h)$^2$]")
    
    # ax.set_xlim(0, 200)
    # # ax.set_ylim(-20, 100)
    # # ax.set_ylim(0, 0.02)

    # fig.savefig("correlation_functions/test_corrfunc.png")

    # angular_correlation_function("data_bose_r_19_A.hdf5")


# %%
