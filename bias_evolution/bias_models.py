import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# cosmological parameters
Omega_m_0 = 0.3089
Omega_lambda_0 = 0.6911

# cosmological functions
# ------------------------------------------------
# Hubble parameter squared
def H2(z):
    return Omega_m_0*(1+z)**3 + Omega_lambda_0


# density parameters
def Omega_m(z):
    return Omega_m_0 * (1+z)**3 / H2(z)


def Omega_lambda(z):
    return Omega_lambda_0 / H2(z)


# linear growth factor
def D(z):
    return 2.5*Omega_m(z)/((1+z)*(Omega_m(z)**(4/7) - Omega_lambda(z) + (1 + Omega_m(z)/2)*(1 + Omega_lambda(z)/70)))


# bias evolution
# ------------------------------------------------
def b_1_conserved_tracers(z, b_1_i):
    return (b_1_i - 1) * D(np.max(z))/D(z) + 1


# sources and sinks of galaxies as a function of time
def A(z, z_star, sigma_0):
    return np.exp(-np.log((z_star+1)/(z+1))**2/(2*sigma_0**2))


def j(z, alpha_1, alpha_2):
    return alpha_1*(1+z)**3 + alpha_2*(1+z)**6


# comoving galaxy number density as a function of time
def n_g(z, n_g_ref, z_ref, z_star, sigma_0, alpha_1, alpha_2):
    integrals = np.ndarray(z.shape[0])
    for i in range(z.shape[0]):
        integrals[i] = integrate.quad(lambda z_: A(z_, z_star, sigma_0)*j(z_, alpha_1, alpha_2)/(1+z_), z[i], z_ref)[0]
    return n_g_ref + integrals


def b_1_non_conserved_tracers(z, b_1_ref, n_g_ref, z_ref, z_star, sigma_0, alpha_1, alpha_2):
    # instantaneous formation bias
    def b_1_star(z):
        return 1 + alpha_2 / (alpha_1/(1+z)**3 + alpha_2)


    integrals = np.ndarray(z.shape[0])
    number_density = n_g(z, n_g_ref, z_ref, z_star, sigma_0, alpha_1, alpha_2)
    for i in range(z.shape[0]):
        integrals[i] = integrate.quad(lambda z_: A(z_, z_star, sigma_0)*j(z_, alpha_1, alpha_2)*(b_1_star(z_) - 1)*D(z_)/(1+z_), z[i], z_ref)[0]
    return 1 + (b_1_ref - 1)*number_density[np.argmax(z)]*D(np.max(z))/(number_density*D(z)) + 1/(number_density*D(z)) * integrals


if __name__ == "__main__":
    import h5py

    with h5py.File("bias_evolution/number_density_evolution.hdf5", "r") as number_density_save:
        z_star, sigma_0, alpha_1, alpha_2 = np.array(number_density_save["const_stellar_mass/11.5<m<inf/params"])

    with h5py.File("number_density/number_density_const_stellar_mass.hdf5", "r") as number_density_save:
        z = np.array(number_density_save["11.5<m<inf/z"])
        n_g_i = number_density_save["11.5<m<inf/n_g"][-1]
        n_g_f = number_density_save["11.5<m<inf/n_g"][0]

    with h5py.File("bias_evolution/bias_measurements.hdf5", "r") as bias_save:
        b_1_i = bias_save["const_stellar_mass/11.5<m<inf/b_1"][-1]
        b_1_f = bias_save["const_stellar_mass/11.5<m<inf/b_1"][0]

    z = np.linspace(1, 0, 100)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    
    ax.plot(z, b_1_conserved_tracers(z, 2), label="Conserved Tracers")
    ax.plot(z, b_1_non_conserved_tracers(z, 2, 1, 1, 0.3, 0.2, 4, 1), label="Non-conserved Tracers")

    # ax.plot(z, n_g(z, n_g_i, np.max(z), z_star, sigma_0, alpha_1, alpha_2), label="Past-anchored")
    # ax.plot(z, n_g(z, n_g_f, np.min(z), z_star, sigma_0, alpha_1, alpha_2), label="Present-anchored", linestyle="dashed")

    # ax.plot(z, b_1_non_conserved_tracers(z, b_1_i, n_g_i, np.max(z), z_star, sigma_0, alpha_1, alpha_2), label="Past-anchored")
    # ax.plot(z, b_1_non_conserved_tracers(z, b_1_f, n_g_f, np.min(z), z_star, sigma_0, alpha_1, alpha_2), label="Present-anchored", linestyle="dashed")

    ax.invert_xaxis()

    ax.set_ylabel("Linear Bias $b_1$")
    ax.set_xlabel("Redshift $z$")
    ax.legend()

    fig.savefig("bias_evolution/bias_models.pdf")