# %%
import numpy as np
import matplotlib.pyplot as plt

# scale factor
scale_factor = np.linspace(0.1, 1, 100)

# bias evolution
# --------------------------------------------------
# linear growth factor
def D(a):
    return a

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
    return np.array([np.trapz(A(a[:p])*j(a[:p])/a[:p], a[:p]) for p in range(a.size)])

# linear density bias as a function of time
def b_1(a):
    return 1 + 1/(n_g(a)*D(a)) * np.array([np.trapz(A(a[:p])*j(a[:p])/a[:p] * (b_1_star(a[:p]) - 1)*D(a[:p]), a[:p]) for p in range(a.size)])

# (normalised) multipole moments of second order bias as a function of time
def chi_2(a):
    return -1/(n_g(a)*D(a)**2) * np.array([np.trapz(A(a[:p])*j(a[:p])/a[:p] * D(a[:p])*(D(a[p]) - D(a[:p]))*(b_1_star(a[:p]) - 1), a[:p]) for p in range(a.size)])

def chi_0(a):
    return 21/(17*D(a)**2) * (-chi_2(a) + 1/n_g(a)*np.array([np.trapz(A(a[:p])*j(a[:p])/a[:p] * D(a[:p])**2 * 0.5*b_2_star(a[:p]), a[:p]) for p in range(a.size)]))

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
