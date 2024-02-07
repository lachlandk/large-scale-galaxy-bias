# %%
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

planck = cosmology.setCosmology("planck18")
# EdS with baryon density from planck18
EdS = cosmology.setCosmology("EdS", Ob0=0.0490)

k = 10**np.linspace(-5, 3, 200)
R = 10**np.linspace(-3, 2.69, 200)
# redshifts
epochs = [0, 0.1, 0.5, 1, 2, 5, 10]

# plot of linear matter power spectrum and correlation function
fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
axes = axes.flatten()

for z in epochs:
    axes[0].plot(k, planck.matterPowerSpectrum(k, z=z), label=f"z={z}")
    axes[1].plot(R, np.abs(planck.correlationFunction(R, z=z)), label=f"z={z}")
    axes[2].plot(k, EdS.matterPowerSpectrum(k, z=z), label=f"z={z}")
    axes[3].plot(R, np.abs(EdS.correlationFunction(R, z=z)), label=f"z={z}")

axes[0].set_title("Linear Matter Power Spectrum (Planck18)")
axes[0].set_ylabel("$P(k)$")
axes[0].set_xlabel("k (h/Mpc)")
axes[0].set_yscale("log")
axes[0].set_xscale("log")
axes[0].legend(loc="lower left")
axes[1].set_title("Matter Correlation Function (Planck18)")
axes[1].set_ylabel("$|\\xi(R)|$")
axes[1].set_xlabel("R (Mpc/h)")
axes[1].set_yscale("log")
axes[1].set_xscale("log")
axes[1].legend(loc="lower left")
axes[2].set_title("Linear Matter Power Spectrum (EdS with $\\Omega_{B,0}$ from Planck18)")
axes[2].set_ylabel("$P(k)$")
axes[2].set_xlabel("k (h/Mpc)")
axes[2].set_yscale("log")
axes[2].set_xscale("log")
axes[2].legend(loc="lower left")
axes[3].set_title("Matter Correlation Function (EdS with $\\Omega_{B,0}$ from Planck18)")
axes[3].set_ylabel("$|\\xi(R)|$")
axes[3].set_xlabel("R (Mpc/h)")
axes[3].set_yscale("log")
axes[3].set_xscale("log")
axes[3].legend(loc="lower left")

# %%

