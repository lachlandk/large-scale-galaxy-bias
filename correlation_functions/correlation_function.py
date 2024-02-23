import h5py
import numpy as np
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks


def correlation_function(data_catalogue, random_catalogue, s_bins, mu_max, nmu_bins, save_name=None):
    with h5py.File(f"catalogues/{data_catalogue}", "r") as data, h5py.File(f"catalogues/{random_catalogue}", "r") as random:
        d_pos = np.array(data["Pos"])
        r_pos = np.array(random["Pos"])

    start_time = datetime.now()

    # count pairs
    print("Counting data pairs...")
    DD_pairs = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=10, mu_max=mu_max, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_pos[:,0], DEC1=d_pos[:,1], CZ1=d_pos[:,2], is_comoving_dist=True)
    print(f"Data pairs counted, elapsed time: {datetime.now() - start_time}")
    print("Counting random pairs...")
    RR_pairs = DDsmu_mocks(autocorr=True, cosmology=2, nthreads=10, mu_max=mu_max, nmu_bins=nmu_bins, binfile=s_bins, RA1=r_pos[:,0], DEC1=r_pos[:,1], CZ1=r_pos[:,2], is_comoving_dist=True)
    print(f"Random pairs counted, elapsed time: {datetime.now() - start_time}")
    print("Counting data-random pairs...")
    DR_pairs = DDsmu_mocks(autocorr=False, cosmology=2, nthreads=10, mu_max=mu_max, nmu_bins=nmu_bins, binfile=s_bins, RA1=d_pos[:,0], DEC1=d_pos[:,1], CZ1=d_pos[:,2], RA2=r_pos[:,0], DEC2=r_pos[:,1], CZ2=r_pos[:,2], is_comoving_dist=True)
    print(f"Data-random pairs counted, elapsed time: {datetime.now() - start_time}")

    # calculate correlation function
    data_length = d_pos.shape[0]
    random_length = r_pos.shape[0]
    print("Calculating correlation function...")
    corrfunc = convert_3d_counts_to_cf(data_length, data_length, random_length, random_length, DD_pairs, DR_pairs, DR_pairs, RR_pairs)
    print(f"Correction function calculated, elapsed time: {datetime.now() - start_time}")

    # figure out if xi is a function of one or two variables
    if nmu_bins == 1:
        if save_name is not None:
            with h5py.File(f"correlation_functions/{save_name}", "w") as save_file:
                save_file.create_dataset("xi", data=corrfunc)
        return corrfunc
    else:
        # xi is a two variable function, calculate moments
        print(f"Calculating moments...")
        mu = DD_pairs["mumax"][:nmu_bins]
        corrfunc_length_1D = corrfunc.shape[0] // nmu_bins
        xi_0 = np.ndarray(corrfunc_length_1D)
        xi_1 = np.ndarray(corrfunc_length_1D)
        xi_2 = np.ndarray(corrfunc_length_1D)
        xi_3 = np.ndarray(corrfunc_length_1D)
        xi_4 = np.ndarray(corrfunc_length_1D)

        for i in trange(s_bins.shape[0] - 1):
            j = nmu_bins*i
            xi_0[i] = np.trapz(corrfunc[j:j+nmu_bins], mu)
            xi_1[i] = 3 * np.trapz(corrfunc[j:j+nmu_bins] * mu, mu)
            xi_2[i] = 5 * np.trapz(corrfunc[j:j+nmu_bins] * 0.5 * (3*mu**2 - 1), mu)
            xi_3[i] = 7 * np.trapz(corrfunc[j:j+nmu_bins] * 0.5 * (5*mu**3 - 3*mu), mu)
            xi_4[i] = 9 * np.trapz(corrfunc[j:j+nmu_bins] * 0.125 * (35*mu**4 - 30*mu**2 + 3), mu)
        print(f"Moments calculated, elapsed time: {datetime.now() - start_time}")

        if save_name is not None:
            with h5py.File(f"correlation_functions/{save_name}", "w") as save_file:
                save_file.create_dataset("xi_0", data=xi_0)
                save_file.create_dataset("xi_1", data=xi_1)
                save_file.create_dataset("xi_2", data=xi_2)
                save_file.create_dataset("xi_3", data=xi_3)
                save_file.create_dataset("xi_4", data=xi_4)

        return (xi_0, xi_1, xi_2, xi_3, xi_4)


if __name__ == "__main__":
    s = np.geomspace(0.1, 300, 201)

    xi = correlation_function("data_catalogue.hdf5", "random_catalogue.hdf5", s, 1, 1)
    (xi_0, xi_1, xi_2, xi_3, xi_4) = correlation_function("data_catalogue.hdf5", "random_catalogue.hdf5", s, 1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.plot(s[:-1], xi)
    ax1.set_xlabel("s [cMpc/h]")
    ax1.set_ylabel("$\\xi$")
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    ax2.plot(s[:-1], xi_0, label="Monopole")
    ax2.plot(s[:-1], xi_1, label="Dipole")
    ax2.plot(s[:-1], xi_2, label="Quadrupole")
    ax2.plot(s[:-1], xi_3, label="Octupole")
    ax2.plot(s[:-1], xi_4, label="Hexadecapole")
    ax2.set_xlabel("s [cMpc/h]")
    ax2.set_ylabel("Multipole moments of $\\xi$")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend()

    plt.savefig("correlation_functions/correlation_function.png")
