import h5py
import numpy as np
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks


def correlation_function(data_catalogue, random_catalogue, s_bins, mu_max, nmu_bins, rsd=True, save_name=None):
    start_time = datetime.now()

    # count pairs
    print("Counting data pairs...")
    filename = f"correlation_functions/DD_pair_counts_{data_catalogue.split('.')[0]}_nmu_bins={nmu_bins}.hdf5"
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
                dist = np.array(catalogue[z_bin]["ObsDist"]) if rsd == True else pos[:,2] 
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")
                pairs = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=10, mu_max=mu_max, nmu_bins=nmu_bins, binfile=s_bins, RA1=pos[:,0], DEC1=pos[:,1], CZ1=dist, is_comoving_dist=True)
                save.create_dataset("npairs", data=pairs["npairs"])
                DD_pairs.append(pairs["npairs"])
    print(f"Data pairs counted, elapsed time: {datetime.now() - start_time}")
    
    print("Counting random pairs...")
    filename = f"correlation_functions/RR_pair_counts_{random_catalogue.split('.')[0]}_nmu_bins={nmu_bins}.hdf5"
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
                dist = np.array(catalogue[z_bin]["ObsDist"]) if rsd == True else pos[:,2] 
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")
                pairs = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=10, mu_max=mu_max, nmu_bins=nmu_bins, binfile=s_bins, RA1=pos[:,0], DEC1=pos[:,1], CZ1=dist, is_comoving_dist=True)
                save.create_dataset("npairs", data=pairs["npairs"])
                RR_pairs.append(pairs["npairs"])
    print(f"Random pairs counted, elapsed time: {datetime.now() - start_time}")
    
    print("Counting data-random pairs...")
    filename = f"correlation_functions/DR_pair_counts_{data_catalogue.split('.')[0]}_{random_catalogue.split('.')[0]}_nmu_bins={nmu_bins}.hdf5"
    DR_pairs = []
    try:
        with h5py.File(filename, "r") as data_random_pairs:
            for z_bin in data_random_pairs:
                DR_pairs.append(np.array(data_random_pairs[z_bin]["npairs"]))
    except FileNotFoundError:
        with h5py.File(filename, "w") as data_random_pairs, h5py.File(f"catalogues/{data_catalogue}", "r") as data, h5py.File(f"catalogues/{random_catalogue}", "r") as random:
            for z_bin in data:
                save = data_random_pairs.create_group(z_bin)
                d_pos = np.array(data[z_bin]["Pos"])
                d_dist = np.array(data[z_bin]["ObsDist"]) if rsd == True else d_pos[:,2] 
                r_pos = np.array(random[z_bin]["Pos"])
                r_dist = np.array(random[z_bin]["ObsDist"]) if rsd == True else r_pos[:,2] 
                print(f"Galaxies in bin {z_bin}: Data={d_pos.shape[0]}, Random={r_pos.shape[0]}")
                pairs = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=10, mu_max=mu_max, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_pos[:,0], DEC1=d_pos[:,1], CZ1=d_dist, RA2=r_pos[:,0], DEC2=r_pos[:,1], CZ2=r_dist, is_comoving_dist=True)
                dataset = save.create_dataset("npairs", data=pairs["npairs"])
                dataset.attrs["ndata"] = d_pos.shape[0]
                dataset.attrs["nrandom"] = r_pos.shape[0]
                DR_pairs.append(pairs["npairs"])
    print(f"Data-random pairs counted, elapsed time: {datetime.now() - start_time}")

    # calculate correlation function
    print("Calculating correlation function...")
    corrfunc = []
    with h5py.File(f"catalogues/{data_catalogue}", "r") as data, h5py.File(f"catalogues/{random_catalogue}", "r") as random:
            for i, z_bin in enumerate(data):
                data_length = data[z_bin]["Pos"].shape[0]
                random_length = random[z_bin]["Pos"].shape[0]
                corrfunc.append(convert_3d_counts_to_cf(data_length, data_length, random_length, random_length, DD_pairs[i], DR_pairs[i], DR_pairs[i], RR_pairs[i]))
    print(f"Correction function calculated, elapsed time: {datetime.now() - start_time}")

    # figure out if xi is a function of one or two variables
    if nmu_bins == 1:
        if save_name is not None:
            with h5py.File(f"correlation_functions/{save_name}", "w") as save_file:
                save_file.create_dataset("xi", data=corrfunc)
        return corrfunc, z_bins
    # else:
    #     # xi is a two variable function, calculate moments
    #     print(f"Calculating moments...")
    #     mu = DD_pairs["mumax"][:nmu_bins]
    #     corrfunc_length_1D = corrfunc.shape[0] // nmu_bins
    #     xi_0 = np.ndarray(corrfunc_length_1D)
    #     xi_1 = np.ndarray(corrfunc_length_1D)
    #     xi_2 = np.ndarray(corrfunc_length_1D)
    #     xi_3 = np.ndarray(corrfunc_length_1D)
    #     xi_4 = np.ndarray(corrfunc_length_1D)

    #     for i in trange(s_bins.shape[0] - 1):
    #         j = nmu_bins*i
    #         xi_0[i] = np.trapz(corrfunc[j:j+nmu_bins], mu)
    #         xi_1[i] = 3 * np.trapz(corrfunc[j:j+nmu_bins] * mu, mu)
    #         xi_2[i] = 5 * np.trapz(corrfunc[j:j+nmu_bins] * 0.5 * (3*mu**2 - 1), mu)
    #         xi_3[i] = 7 * np.trapz(corrfunc[j:j+nmu_bins] * 0.5 * (5*mu**3 - 3*mu), mu)
    #         xi_4[i] = 9 * np.trapz(corrfunc[j:j+nmu_bins] * 0.125 * (35*mu**4 - 30*mu**2 + 3), mu)
    #     print(f"Moments calculated, elapsed time: {datetime.now() - start_time}")

    #     if save_name is not None:
    #         with h5py.File(f"correlation_functions/{save_name}", "w") as save_file:
    #             save_file.create_dataset("xi_0", data=xi_0)
    #             save_file.create_dataset("xi_1", data=xi_1)
    #             save_file.create_dataset("xi_2", data=xi_2)
    #             save_file.create_dataset("xi_3", data=xi_3)
    #             save_file.create_dataset("xi_4", data=xi_4)

    #     return (xi_0, xi_1, xi_2, xi_3, xi_4)


if __name__ == "__main__":
    s = np.linspace(0.1, 200, 101)

    xi, z_bins = correlation_function("data_catalgoue.hdf5", "random_catalogue.hdf5", s, 1, 1, rsd=False)
    # (xi_0, xi_1, xi_2, xi_3, xi_4) = correlation_function("data_catalogue.hdf5", "random_catalogue.hdf5", s, 1, 100)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

    for i in range(5):
        ax1.plot(s[:-1], s[:-1]**2 * xi[i], label=z_bins[i])
    ax1.set_xlabel("s [cMpc/h]")
    ax1.set_ylabel("$s^2\\xi [(cMpc/h)^2]$")
    ax1.set_ylim(0, 100)
    ax1.legend()

    fig.suptitle("Correlation Function for r<19.5")

    # ax2.plot(s[:-1], xi_0, label="Monopole")
    # ax2.plot(s[:-1], xi_1, label="Dipole")
    # ax2.plot(s[:-1], xi_2, label="Quadrupole")
    # ax2.plot(s[:-1], xi_3, label="Octupole")
    # ax2.plot(s[:-1], xi_4, label="Hexadecapole")
    # ax2.set_xlabel("s [cMpc/h]")
    # ax2.set_ylabel("Multipole moments of $\\xi$")
    # ax2.set_yscale("log")
    # ax2.legend()

    plt.savefig("correlation_functions/corrfunc.png")
