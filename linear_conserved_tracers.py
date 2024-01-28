# %%
import numpy as np
import matplotlib.pyplot as plt

# EdS universe, D=a
a = np.linspace(0.2, 1, 100)

# linear density bias as a function of time
def b_1(y):
    return 1 + (b_1_i - 1)*np.exp(-y) + 2*(b_v_i - 1)*np.exp(-y)*(1 - np.exp(-y/2))

# linear velocity bias as a function of time
def b_v(y):
    return 1 + (b_v_i - 1)*np.exp(-3*y/2)

# plot of linear bias evolution
fig, ax = plt.subplots(1, 1)

# Initial conditions
initial_bias = [[2, 1.1], [2, 0.9], [2, 1]]
colours = ["tab:blue", "tab:orange", "tab:green"]

for i in range(3):
    b_1_i = initial_bias[i][0]
    b_v_i = initial_bias[i][1]
    ax.plot(a, b_1(np.log(a/a[0])), label=f"$b_1$, $b_v^*={b_v_i}$", c=colours[i])
    ax.plot(a, b_v(np.log(a/a[0])), label=f"$b_v$, $b_v^*={b_v_i}$", c=colours[i], linestyle="--")

ax.set_title("Evolution of Linear Bias with Conserved Tracers")
ax.set_ylabel("Linear Bias")
ax.set_xlabel("Scale Factor")
ax.legend(loc="upper right")

# %%
