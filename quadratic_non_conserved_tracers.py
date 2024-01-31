# %%
import numpy as np
import matplotlib.pyplot as plt

# scale factor
scale_factor = np.linspace(0.1, 1, 100)
# density parameters today
Omega_m_0 = 0.25
Omega_lambda_0 = 0.75

# bias evolution
# --------------------------------------------------
# Hubble parameter squared
def H2(a):
    return Omega_m_0/a**3 + Omega_lambda_0

# density parameters
def Omega_m(a):
    return Omega_m_0 / (a**3 * H2(a)) 

def Omega_lambda(a):
    return Omega_lambda_0 / H2(a)

# linear growth factor
def D(a):
    return 2.5*a*Omega_m(a)/(Omega_m(a)**(4/7) - Omega_lambda(a) + (1 + Omega_m(a)/2)*(1 + Omega_lambda(a)/70))

# sources and sinks of galaxies as a function of time
def A(a):
    return np.exp(-(np.log(a)-np.log(a_0))**2/(2*sigma_0**2))

def j(a):
    return alpha_1/a**3 + alpha_2/a**6

# instantaneous formation bias
def b_1_star(a):
    return 1 + alpha_2 / (alpha_1 * a**3 + alpha_2)

def b_2_star(a):
    return 2*alpha_2/(alpha_1 * a**3 + alpha_2)

# comoving galaxy number density as a function of time
def n_g(a):
    integrals = np.ndarray(100)
    for i in range(a.size):
        sf = np.linspace(0.01, a[i], 100)
        integrals[i] = np.trapz(A(sf)*j(sf)/sf, sf)
    return integrals

# linear density bias as a function of time
def b_1(a):
    integrals = np.ndarray(100)
    for i in range(a.size):
        sf = np.linspace(0.01, a[i], 100)
        integrals[i] = np.trapz(A(sf)*j(sf)/sf * (b_1_star(sf) - 1)*D(sf), sf)
    return 1 + 1/(n_g(a)*D(a)) * integrals

# (normalised) multipole moments of second order bias as a function of time
def chi_2(a):
    integrals = np.ndarray(100)
    for i in range(a.size):
        sf = np.linspace(0.01, a[i], 100)
        integrals[i] = np.trapz(A(sf)*j(sf)/sf * D(sf)*(D(a[i]) - D(sf))*(b_1_star(sf) - 1), sf)
    return -1/(n_g(a)*D(a)**2) * integrals

def chi_0(a):
    integrals = np.ndarray(100)
    for i in range(a.size):
        sf = np.linspace(0.01, a[i], 100)
        integrals[i] = np.trapz(A(sf)*j(sf)/sf * 0.5*b_2_star(sf)*D(sf)**2, sf)
    return 21/(17*D(a)**2) * (-chi_2(a) + 1/n_g(a) * integrals)

# plot of quadratic bias evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
axes = axes.flatten()

# Parameters
# alpha_1, alpha_2
alpha = [[4, 1], [1, 1], [1, 4]]
styles = ["solid", "dashed", "dotted"]
# sigma_0, a_0
sfr = [[0.2, 0.3], [0.2, 0.5], [0.2, 0.7]]
colours = ["blue", "red", "green"]

for i in range(3):
    alpha_1 = alpha[i][0]
    alpha_2 = alpha[i][1]

    axes[0].plot(scale_factor, b_1_star(scale_factor), c="black", linestyle=styles[i])
    axes[1].plot(scale_factor, b_2_star(scale_factor), c="black", linestyle=styles[i])

    for k in range(3):
        sigma_0 = sfr[k][0]
        a_0 = sfr[k][1]

        number_density = n_g(scale_factor)
        axes[2].plot(scale_factor, number_density/number_density[-1], c=colours[k], linestyle=styles[i])
        axes[3].plot(scale_factor, b_1(scale_factor), c=colours[k], linestyle=styles[i])
        axes[4].plot(scale_factor, chi_2(scale_factor), c=colours[k], linestyle=styles[i])
        axes[5].plot(scale_factor, chi_0(scale_factor), c=colours[k], linestyle=styles[i])

fig.suptitle("Evolution of Linear and Quadratic Bias with Non-Conserved Tracers")
axes[0].set_ylabel("$b_1^*$")
axes[0].set_xlabel("Scale Factor")
axes[0].set_xlim(0.1, 1)
axes[0].set_ylim(1.1, 2)
axes[1].set_ylabel("$b_2^*$")
axes[1].set_xlabel("Scale Factor")
axes[1].set_xlim(0.1, 1)
axes[1].set_ylim(0.4, 2)
axes[2].set_ylabel("$\\bar{n}_g^{(c)}/\\bar{n}_g^{(c)}(1)$")
axes[2].set_xlabel("Scale Factor")
axes[2].set_xlim(0.1, 1)
axes[2].set_ylim(0, 1)
axes[3].set_ylabel("$b_1$")
axes[3].set_xlabel("Scale Factor")
axes[3].set_xlim(0.1, 1)
axes[3].set_ylim(1.3, 2)
axes[4].set_ylabel("$\\chi_2^{(2)}$")
axes[4].set_xlabel("Scale Factor")
axes[4].set_xlim(0.1, 1)
axes[4].set_ylim(-0.25, 0)
axes[5].set_ylabel("$\\chi_0^{(2)}$")
axes[5].set_xlabel("Scale Factor")
axes[5].set_xlim(0.1, 1)
axes[5].set_ylim(0, 1.2)

# %%
