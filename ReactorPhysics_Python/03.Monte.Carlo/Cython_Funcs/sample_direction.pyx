#******************************************************************
# This code is released under the GNU General Public License (GPL).
#
# Siim Erik Pugal, 2023-2024
#******************************************************************
import numpy as np
cimport numpy as np
cimport cython

def sample_direction():
    cdef double teta = np.pi * np.random.rand()
    cdef double phi = 2.0 * np.pi * np.random.rand()
    cdef double dirX = np.sin(teta) * np.cos(phi)
    cdef double dirY = np.sin(teta) * np.sin(phi)
    return dirX, dirY