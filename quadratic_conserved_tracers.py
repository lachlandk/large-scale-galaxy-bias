# %%
import numpy as np
import matplotlib.pyplot as plt

# EdS universe, D=a
a = np.linspace(0.2, 1, 100)

# bias evolution
# --------------------------------------------------
# linear density bias as a function of time
def b_1(y):
    return 1 + (b_1_i - 1)*np.exp(-y) + 2*(b_v_i - 1)*np.exp(-y)*(1 - np.exp(-y/2))

# linear velocity bias as a function of time
def b_v(y):
    return 1 + (b_v_i - 1)*np.exp(-3*y/2)

# coefficients
def epsilon_delta(y):
    return (b_1(y) - 1)*np.exp(y)*(np.exp(y) - 1)

def epsilon_v(y):
    return (b_v(y) - 1)*np.exp(y)*(np.exp(y/2) - 1)

# (normalised) multipole moments of second order bias as a function of time
def chi_0(y):
    return (b_2_i/2 + 4/21*epsilon_delta(y) + 2/21*epsilon_v(y)*(3 + 14*np.exp(y/2) - 14*epsilon_v(y) + 21*epsilon_delta(y)/(np.exp(y) - 1)))/(17/21*np.exp(2*y))

def chi_1(y):
    return epsilon_v(y)*(-1 + 2*np.exp(y/2) + epsilon_delta(y)/(np.exp(y) - 1))/(1/2*np.exp(2*y))

def chi_2(y):
    return (-epsilon_delta(y) + epsilon_v(y)*(-12 + 14*np.exp(y/2) + 7*epsilon_v(y)))/(np.exp(2*y))


# plot of quadratic bias evolution
fig, ax = plt.subplots(1, 1)

# Initial conditions
initial_bias = [[2, 0.5, 1.1], [2, 0.5, 1], [2, 0.5, 0.9]]
styles = ["solid", "dashed", "dotted"]

for i in range(3):
    b_1_i = initial_bias[i][0]
    b_2_i = initial_bias[i][1]
    b_v_i = initial_bias[i][2]
    ax.plot(a, chi_0(np.log(a/a[0])), label=f"monopole, $b_v^*={b_v_i}$", c="blue", linestyle=styles[i])
    ax.plot(a, chi_1(np.log(a/a[0])), label=f"dipole, $b_v^*={b_v_i}$", c="red", linestyle=styles[i])
    ax.plot(a, chi_2(np.log(a/a[0])), label=f"quadrupole, $b_v^*={b_v_i}$", c="green", linestyle=styles[i])

ax.set_title("Evolution of Quadratic Bias with Conserved Tracers")
ax.set_ylabel("Normalised Multipoles")
ax.set_xlabel("Scale Factor")
ax.legend(loc="upper right")

# %%
