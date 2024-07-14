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


def logistic(z, z_0, k):
    return 1/(1 + np.exp(-k*(z_0 - z)))


# sources and sinks of galaxies as a function of time
def A(z, z_0, k):
    return -k*logistic(z, z_0, k)*(1 - logistic(z, z_0, k))


def j(z, alpha_1, alpha_2):
    return alpha_1*(1+z)**3 + alpha_2*(1+z)**6


# comoving galaxy number density as a function of time
def n_g(z, n_g_i, z_0, k, alpha_1, alpha_2):
    integrals = np.ndarray(z.shape[0])
    for i in range(z.shape[0]):
        integrals[i] = integrate.quad(lambda z_: A(z_, z_0, k)*j(z_, alpha_1, alpha_2)/(1+z_), z[i], np.max(z))[0]
    return n_g_i + integrals


def b_1_non_conserved_tracers(z, n_g_i, z_0, k, alpha_1, alpha_2, b_1_i):
    # instantaneous formation bias
    def b_1_star(z):
        return 1 + alpha_2 / (alpha_1/(1+z)**3 + alpha_2)


    integrals = np.ndarray(z.shape[0])
    number_density = n_g(z, n_g_i, z_0, k, alpha_1, alpha_2)
    for i in range(z.shape[0]):
        integrals[i] = integrate.quad(lambda z_: A(z_, z_0, k)*j(z_, alpha_1, alpha_2)*(b_1_star(z_) - 1)*D(z_)/(1+z_), z[i], np.max(z))[0]
    return 1 + (b_1_i - 1)*number_density[np.argmax(z)]*D(np.max(z))/(number_density*D(z)) + 1/(number_density*D(z)) * integrals


if __name__ == "__main__":
    z = np.linspace(1, 0, 100)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    
    ax.plot(z, b_1_conserved_tracers(z, 2), label="Conserved Tracers")
    ax.plot(z, b_1_non_conserved_tracers(z, 1, 0.3, 0.2, 4, 1, 2), label="Non-conserved Tracers")
    ax.invert_xaxis()

    ax.set_ylabel("Linear Bias $b_1$")
    ax.set_xlabel("Redshift $z$")
    ax.legend()

    fig.savefig("bias_evolution/bias_models.pdf")