import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# from scipy import interpolate, integrate
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks

# cosmological parameters
h = 0.6774


def correlation_function(data_file, catalogue, s_bins):
    start_time = datetime.now()

    nmu_bins = 100
    filename = f"correlation_functions/corrfunc_{data_file}"
    with h5py.File(filename, "a") as save_file:
        if f"{catalogue}/xi_0" in save_file:
            xi = np.array(save_file[catalogue]["xi_0"])
            sigma = np.array(save_file[catalogue]["sigma"])
            print(f"Correlation function loaded, elapsed time: {datetime.now() - start_time}")
        else:
            with h5py.File(f"catalogues/{data_file}", "r") as sample:
                d_pos = np.array(sample[catalogue]["Pos"])
                d_ra = d_pos[:,0]
                d_dec = d_pos[:,1]
                d_dist = d_pos[:,2] / h
                z_cos = np.array(sample[catalogue]["CosZ"])
                r_pos = np.array(sample[f"{catalogue}/random"]["Pos"])
                r_ra = r_pos[:,0]
                r_dec = r_pos[:,1]
                r_dist = r_pos[:,2] / h

            # count data pairs
            print(f"Counting data pairs, sample size: {d_pos.shape[0]}...")
            if f"{catalogue}/DD" in save_file:    
                DD_pairs = np.array(save_file[catalogue]["DD"])
                N_D = save_file[catalogue].attrs["N_D"]
            else:
                save = save_file.create_group(catalogue)
                if d_pos.shape[0] < 2:
                    DD_pairs = np.zeros((len(s_bins) - 1) * nmu_bins)
                    N_D = 0
                else:
                    DD_pairs = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_ra, DEC1=d_dec, CZ1=d_dist, is_comoving_dist=True)["npairs"]
                    N_D = d_pos.shape[0]
                save.create_dataset("DD", data=DD_pairs)
                save.attrs["N_D"] = N_D
            print(f"Data pairs counted, elapsed time: {datetime.now() - start_time}")
    
            # count random pairs
            print(f"Counting random pairs, sample size: {r_pos.shape[0]}...")
            if f"{catalogue}/RR" in save_file:
                RR_pairs = np.array(save_file[catalogue]["RR"])
                N_R = save_file[catalogue].attrs["N_R"]
            else:
                if r_pos.shape[0] < 2:
                    RR_pairs = np.zeros((len(s_bins) - 1) * nmu_bins)
                    N_R = 0
                else:
                    RR_pairs = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=r_ra, DEC1=r_dec, CZ1=r_dist, is_comoving_dist=True)["npairs"]
                    N_R = r_pos.shape[0]
                save_file[catalogue].create_dataset("RR", data=RR_pairs)
                save_file[catalogue].attrs["N_R"] = r_pos.shape[0]
            print(f"Random pairs counted, elapsed time: {datetime.now() - start_time}")

            # count data-random pairs
            print("Counting data-random pairs...")
            if f"{catalogue}/DR" in save_file:
                DR_pairs = np.array(save_file[catalogue]["DR"])
                mu = save_file[catalogue].attrs["mu"]
                median_z = save_file[catalogue].attrs["median_z"]
            else:
                if d_pos.shape[0] < 2:
                    DR_pairs = np.zeros((len(s_bins) - 1) * nmu_bins)
                    mu = np.linspace(0, 1, nmu_bins)
                else:
                    DR_pairs_struct = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_ra, DEC1=d_dec, CZ1=d_dist, RA2=r_ra, DEC2=r_dec, CZ2=r_dist, is_comoving_dist=True)
                    DR_pairs = DR_pairs_struct["npairs"]
                    mu = DR_pairs_struct["mumax"][0:nmu_bins]
                median_z = np.median(z_cos)
                save_file[catalogue].create_dataset("DR", data=DR_pairs)
                save_file[catalogue].attrs["mu"] = mu
                save_file[catalogue].attrs["median_z"] = median_z
            print(f"Data-random pairs counted, elapsed time: {datetime.now() - start_time}")

            # calculate correlation function
            print("Calculating correlation function...")
            DD = DD_pairs / (N_D*(N_D - 1))
            RR = RR_pairs / (N_R*(N_R - 1))
            DR = DR_pairs / (N_D * N_R)
            corrfunc = (DD - 2*DR + RR) / RR
            xi = np.ndarray(len(s_bins) - 1)
            sigma = np.ndarray(len(s_bins) - 1)
            for i in range(len(s_bins) - 1):
                j = nmu_bins * i
                xi[i] = np.trapz(corrfunc[j:j+nmu_bins], mu)  # monopole
                # n = N_D * 6/(np.pi * (s_bins[i+1]**3 - s_bins[i]**3))
                # xi_interp = interpolate.CubicSpline(np.concatenate(([0], s_bins[:i+1])), np.concatenate(([0], xi[:i+1])))
                # J_3 = integrate.quad(lambda r: xi_interp(r) * r**2, 0, s_bins[i])[0]
                # sigma[i] = (1 + xi[i]) * (1 + 4*np.pi*n*J_3) / np.sqrt(np.trapz(DD_pairs[j:j+nmu_bins], mu))
                sigma[i] = np.sqrt((1 + xi[i]) / np.trapz(DD_pairs[j:j+nmu_bins], mu))
            save_file[catalogue].create_dataset("s", data=s_bins[:-1])
            save_file[catalogue].create_dataset("xi_0", data=xi)
            save_file[catalogue].create_dataset("sigma", data=sigma)
            print(f"Correction function calculated, elapsed time: {datetime.now() - start_time}")

    return xi, sigma


if __name__ == "__main__":
    s = np.linspace(0.1, 100, 101)

    xi_A, sigma_A = correlation_function("magnitude_limited_A.hdf5", "0.0<z<0.2", s)
    xi_B, sigma_B = correlation_function("magnitude_limited_B.hdf5", "0.0<z<0.2", s)
    xi = np.mean([xi_A, xi_B], axis=0)
    sigma = np.mean([sigma_A, sigma_B], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Correlation Function in Real Space")

    ax.plot(s[:-1], s[:-1]**2 * xi)
    ax.set_xlabel("$r$ [cMpc]")
    ax.set_ylabel("$r^2\\xi(r)$ [cMpc$^2$]")
    
    fig.savefig("correlation_functions/corrfunc_magnitude_limited.png")
