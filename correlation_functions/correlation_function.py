import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks


def correlation_function(data_catalogue, random_catalogue, s_bins, rsd=True, save_name=None):
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
                dist = np.array(catalogue[z_bin]["ObsDist"]) if rsd == True else pos[:,2] 
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")
                pairs = (1/(pos.shape[0]**2)) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=10, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=pos[:,0], DEC1=pos[:,1], CZ1=dist, is_comoving_dist=True)["npairs"]
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
                dist = np.array(catalogue[z_bin]["ObsDist"]) if rsd == True else pos[:,2] 
                print(f"Galaxies in bin {z_bin}: {pos.shape[0]}")
                pairs = (1/(pos.shape[0]**2)) * DDsmu_mocks(autocorr=True, cosmology=2, nthreads=10, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=pos[:,0], DEC1=pos[:,1], CZ1=dist, is_comoving_dist=True)["npairs"]
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
                d_dist = np.array(data[z_bin]["ObsDist"]) if rsd == True else d_pos[:,2] 
                r_pos = np.array(random[z_bin]["Pos"])
                r_dist = np.array(random[z_bin]["ObsDist"]) if rsd == True else r_pos[:,2] 
                print(f"Galaxies in bin {z_bin}: Data={d_pos.shape[0]}, Random={r_pos.shape[0]}")
                count = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=10, mu_max=1, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_pos[:,0], DEC1=d_pos[:,1], CZ1=d_dist, RA2=r_pos[:,0], DEC2=r_pos[:,1], CZ2=r_dist, is_comoving_dist=True)
                mu = count["mumax"][0:nmu_bins]
                pairs = (1/(d_pos.shape[0] * r_pos.shape[0])) * count["npairs"]
                dataset = save.create_dataset("npairs", data=pairs)
                dataset.attrs["ndata"] = d_pos.shape[0]
                dataset.attrs["nrandom"] = r_pos.shape[0]
                data_random_pairs.attrs["mu"] = mu
                DR_pairs.append(pairs)
    print(f"Data-random pairs counted, elapsed time: {datetime.now() - start_time}")

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
    h = 0.6774
    s = np.linspace(0.1, 200, 101)

    xi_A, z_bins = correlation_function("data_catalogue_A.hdf5", "random_catalogue_A.hdf5", s, rsd=False)
    xi_B, z_bins = correlation_function("data_catalogue_B.hdf5", "random_catalogue_B.hdf5", s, rsd=False)
    xi = [(xi_A_bin + xi_B_bin) / 2 for xi_A_bin, xi_B_bin in zip(xi_A, xi_B)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Correlation Function in Real Space")

    ax.plot(s[:-1]/h, s[:-1]**2 * xi[0]/(h**2))
    ax.set_xlabel("$r$ [cMpc/h]")
    ax.set_ylabel("$r^2\\xi(r)$ [(cMpc/h)$^2$]")
    
    fig.savefig("correlation_functions/corrfunc.png")
