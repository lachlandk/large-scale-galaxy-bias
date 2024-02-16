import h5py
import numpy as np
import matplotlib.pyplot as plt
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks

bins = np.linspace(0.1, 300, 101)

with h5py.File("catalogue.hdf5", "r") as catalogue:
    data_pos = np.array(catalogue["Pos"])
 
with h5py.File("random_catalogue.hdf5", "r") as random:
    random_pos = np.array(random["Pos"])

DD_pairs = DDsmu_mocks(autocorr=True, cosmology=1, nthreads=10, mu_max=1, nmu_bins=10, binfile=bins, RA1=data_pos[:,0], DEC1=data_pos[:,1], CZ1=data_pos[:,2], is_comoving_dist=True)
RR_pairs = DDsmu_mocks(autocorr=True, cosmology=1, nthreads=10, mu_max=1, nmu_bins=10, binfile=bins, RA1=random_pos[:,0], DEC1=random_pos[:,1], CZ1=random_pos[:,2], is_comoving_dist=True)
DR_pairs = DDsmu_mocks(autocorr=False, cosmology=1, nthreads=10, mu_max=1, nmu_bins=10, binfile=bins, RA1=data_pos[:,0], DEC1=data_pos[:,1], CZ1=data_pos[:,2], RA2=random_pos[:,0], DEC2=random_pos[:,1], CZ2=random_pos[:,2], is_comoving_dist=True)

data_length = data_pos.shape[0]
random_length = random_pos.shape[0]
corrfunc = convert_3d_counts_to_cf(data_length, data_length, random_length, random_length, DD_pairs, DR_pairs, DR_pairs, RR_pairs)

# moments
mu = np.linspace(0.1, 1, 10)

xi_0 = np.ndarray(100)
xi_1 = np.ndarray(100)
xi_2 = np.ndarray(100)
xi_3 = np.ndarray(100)
xi_4 = np.ndarray(100)

for i in range(100):
    j = 10*i
    xi_0[i] = np.trapz(corrfunc[j:j+10], mu)
    xi_1[i] = 3 * np.trapz(corrfunc[j:j+10] * mu, mu)
    xi_2[i] = 5 * np.trapz(corrfunc[j:j+10] * 0.5 * (3*mu**2 - 1), mu)
    xi_3[i] = 7 * np.trapz(corrfunc[j:j+10] * 0.5 * (5*mu**3 - 3*mu), mu)
    xi_4[i] = 9 * np.trapz(corrfunc[j:j+10] * 0.125 * (35*mu**4 - 30*mu**2 + 3), mu)

fig, ax = plt.subplots(1, 1)

ax.plot(bins[:-1], bins[:-1]**2 * xi_0, label="monopole")
ax.plot(bins[:-1], bins[:-1]**2 * xi_1, label="dipole")
ax.plot(bins[:-1], bins[:-1]**2 * xi_2, label="quadrupole")
ax.plot(bins[:-1], bins[:-1]**2 * xi_3, label="octupole")
ax.plot(bins[:-1], bins[:-1]**2 * xi_4, label="hexadecapole")

ax.legend()

# ax.plot(bins[:-1], (bins[:-1])**2 * corrfunc)

plt.savefig("corrfunc.png")
