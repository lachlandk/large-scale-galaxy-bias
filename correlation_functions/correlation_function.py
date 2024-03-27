import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks

# cosmological parameters
h = 0.6774


def correlation_function(data_file, catalogue, s_bins):
    start_time = datetime.now()

    # count pairs
    # calculate_flags = {"DD": False, "RR": False, "DR": False}
    # filename = f"correlation_functions/corrfunc_{data_catalogue.split('.')[0]}_{random_catalogue.split('.')[0]}_nmu_bins={nmu_bins}.hdf5"
    # try:
    #     with h5py.File(filename, "r") as pairs:
    #         for dataset in ("DD", "RR", "DR"):
    #             try:
    #                 for z_bin in pairs:
    #                     z_bins.append(z_bin)
    #                     DD_pairs.append(np.array(pairs[z_bin][dataset]))
    #             except KeyError:
    #                 calculate_flags[dataset] = True
    # except FileNotFoundError:
    #     with h5py.File(filename, "w") as pairs:
    #         print("Counting data pairs...")

    nmu_bins = 100

    with h5py.File(f"catalogues/{data_file}", "r") as sample:
        d_pos = np.array(sample[catalogue]["Pos"])
        d_ra = d_pos[:,0]
        d_dec = d_pos[:,1]
        d_dist = d_pos[:,2] / h
        r_pos = np.array(sample[f"{catalogue}/random"]["Pos"])
        r_ra = r_pos[:,0]
        r_dec = r_pos[:,1]
        r_dist = r_pos[:,2] / h

        print(f"Counting data pairs, sample size: {d_pos.shape[0]}...")
        filename = f"correlation_functions/DD_pair_counts_{data_file}"
        try:
            with h5py.File(filename, "r") as pairs:
                DD_pairs = np.array(pairs[catalogue]["DD"])
        except (FileNotFoundError, KeyError):
            with h5py.File(filename, "a") as pairs:
                save = pairs.create_group(catalogue)
                if d_pos.shape[0] < 2:
                    DD_pairs = np.zeros((len(s_bins) - 1) * nmu_bins)
                else:
                    DD_pairs = 1/(d_pos.shape[0]**2) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_ra, DEC1=d_dec, CZ1=d_dist, is_comoving_dist=True)["npairs"]
                save.create_dataset("DD", data=DD_pairs)
        print(f"Data pairs counted, elapsed time: {datetime.now() - start_time}")

        print(f"Counting random pairs, sample size: {r_pos.shape[0]}...")
        filename = f"correlation_functions/RR_pair_counts_{data_file}"
        try:
            with h5py.File(filename, "r") as pairs:
                RR_pairs = np.array(pairs[catalogue]["RR"])
        except (FileNotFoundError, KeyError):
            with h5py.File(filename, "a") as pairs:
                save = pairs.create_group(catalogue)
                if r_pos.shape[0] < 2:
                    RR_pairs = np.zeros((len(s_bins) - 1) * nmu_bins)
                else:
                    RR_pairs = 1/(r_pos.shape[0]**2) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=r_ra, DEC1=r_dec, CZ1=r_dist, is_comoving_dist=True)["npairs"]
                save.create_dataset("RR", data=RR_pairs)
        print(f"Random pairs counted, elapsed time: {datetime.now() - start_time}")

        print("Counting data-random pairs...")
        filename = f"correlation_functions/DR_pair_counts_{data_file}"
        try:
            with h5py.File(filename, "r") as pairs:
                mu = pairs[catalogue].attrs["mu"]
                DR_pairs = np.array(pairs[catalogue]["DR"])
        except (FileNotFoundError, KeyError):
            with h5py.File(filename, "a") as pairs:
                save = pairs.create_group(catalogue)
                if d_pos.shape[0] < 2:
                    DR_pairs = np.zeros((len(s_bins) - 1) * nmu_bins)
                    mu = np.linspace(0, 1, nmu_bins)
                else:
                    count = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=16, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_ra, DEC1=d_dec, CZ1=d_dist, RA2=r_ra, DEC2=r_dec, CZ2=r_dist, is_comoving_dist=True)
                    mu = count["mumax"][0:nmu_bins]
                    DR_pairs = (1/(d_pos.shape[0] * r_pos.shape[0])) * count["npairs"]
                save.create_dataset("DR", data=DR_pairs)
                save.attrs["ndata"] = d_pos.shape[0]
                save.attrs["nrandom"] = r_pos.shape[0]
                save.attrs["mu"] = mu
        print(f"Data-random pairs counted, elapsed time: {datetime.now() - start_time}")

    # calculate correlation function
    print("Calculating correlation function...")
    corrfunc = (DD_pairs - 2*DR_pairs + RR_pairs) / RR_pairs
    xi = np.ndarray((5, len(s_bins) - 1))
    for i in range(len(s_bins) - 1):
        j = nmu_bins*i
        xi[0, i] = np.trapz(corrfunc[j:j+nmu_bins], mu)  # monopole
        xi[1, i] = 3 * np.trapz(corrfunc[j:j+nmu_bins] * mu, mu)  # dipole
        xi[2, i] = 5 * np.trapz(corrfunc[j:j+nmu_bins] * 0.5 * (3*mu**2 - 1), mu)  # quadrupole
        xi[3, i] = 7 * np.trapz(corrfunc[j:j+nmu_bins] * 0.5 * (5*mu**3 - 3*mu), mu)  # octupole
        xi[4, i] = 9 * np.trapz(corrfunc[j:j+nmu_bins] * 0.125 * (35*mu**4 - 30*mu**2 + 3), mu)  # hexadecapole
    print(f"Correction function calculated, elapsed time: {datetime.now() - start_time}")

    return xi


if __name__ == "__main__":
    s = np.linspace(0.1, 200, 101)

    xi_A = correlation_function("magnitude_limited_A.hdf5", "0.0<z<0.2", s)
    xi_B = correlation_function("magnitude_limited_B.hdf5", "0.0<z<0.2", s)
    xi = [(xi_A_bin + xi_B_bin) / 2 for xi_A_bin, xi_B_bin in zip(xi_A, xi_B)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Correlation Function in Real Space")

    ax.plot(s[:-1], s[:-1]**2 * xi[0])
    ax.set_xlabel("$r$ [cMpc]")
    ax.set_ylabel("$r^2\\xi(r)$ [cMpc$^2$]")
    
    fig.savefig("correlation_functions/corrfunc_magnitude_limited.png")
