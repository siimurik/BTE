#******************************************************************
# This code is released under the GNU General Public License (GPL).
#
# Siim Erik Pugal, 2023-2024
#******************************************************************
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

def spherical_harmonics(l, m):
    # Calculates and plots real spherical harmonic functions

    num = 500
    theta = np.linspace(0, np.pi, num)
    phi = np.linspace(0, 2 * np.pi, num-1)
    phi = np.append(phi, 2 * np.pi)  # Ensure phi spans the full range

    x = np.zeros((num, num))
    y = np.zeros((num, num))
    z = np.zeros((num, num))

    for i in range(num):
        for j in range(num):
            Y_lm = sph_harm(m, l, phi[j], theta[i]).real

            x[i, j] = np.abs(Y_lm) * np.sin(theta[i]) * np.cos(phi[j])
            y[i, j] = np.abs(Y_lm) * np.sin(theta[i]) * np.sin(phi[j])
            z[i, j] = np.abs(Y_lm) * np.cos(theta[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k', linewidth=0.1)
    ax.set_box_aspect([np.ptp(coord) for coord in [x, y, z]])
    ax.axis('off')
    plt.show()
    #fig.savefig('Fig.pdf', bbox_inches='tight')

# Example usage
spherical_harmonics(6, -5)
