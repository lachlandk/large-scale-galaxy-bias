import numpy as np
from scipy import interpolate, optimize, integrate


# cosmological parameters
c = 3e5  # km/s
Omega_m_0 = 0.3089
Omega_Lambda_0 = 0.6911
H_0 = 67.74  # km/s/Mpc
h = H_0/100

# cosmological functions
# create interpolating function for comoving distance
num_interp_points = 100
interp_points_r = np.ndarray(num_interp_points)
interp_points_z = np.linspace(0, 1.5, num_interp_points)
for i in range(num_interp_points):
    integration_range = np.linspace(0, interp_points_z[i], 100)
    interp_points_r[i] = np.trapz(c / (H_0*np.sqrt(Omega_m_0 * (1 + integration_range)**3 + Omega_Lambda_0)), integration_range)  # [cMpc]
comoving_distance_interp = interpolate.CubicSpline(interp_points_z, interp_points_r, extrapolate=False)  # [cMpc]


def comoving_distance(z):
    if isinstance(z, np.ndarray):
        return comoving_distance_interp(z.clip(0, None))
    else:
        return comoving_distance_interp(z) if z > 0 else 0
    

def z_at_comoving_distance(r):
    if isinstance(r, np.ndarray):
        z = np.ndarray(r.shape[0])
        for i in range(r.shape[0]):
            z[i] = optimize.root_scalar(lambda z: r[i] - comoving_distance(z), bracket=[0, 1.5]).root
    else:
        z = optimize.root_scalar(lambda z: r - comoving_distance(z), bracket=[0, 1.5]).root
    return z


# redshift with redshift space distortions
def observed_redshift(cosmo_z, v_r):
    return (1 + cosmo_z)*(1 + v_r/c) - 1  # correct to linear order


# redshift without redshift space distortions
def cosmological_redshift(obs_z, v_r):
    return (1 + obs_z)/(1 + v_r/c) - 1  # correct to linear order


# 4pi * fraction of the sky included in the input region
def sky_fraction(ra_min, ra_max, dec_min, dec_max):
    return integrate.dblquad(lambda _, theta: np.cos(theta), ra_min*np.pi/180, ra_max*np.pi/180, dec_min*np.pi/180, dec_max*np.pi/180)[0]


# volume of part of a spherical shell between some radii and angles
def volume(r_min, r_max, ra_min, ra_max, dec_min, dec_max):
    # volume = 4/3pi * fraction of sky included * difference in radii cubed
    return sky_fraction(ra_min, ra_max, dec_min, dec_max)*(r_max**3 - r_min**3)/3
