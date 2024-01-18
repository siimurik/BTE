#******************************************************************
# This code is released under the GNU General Public License (GPL).
#
# Siim Erik Pugal, 2023-2024
#******************************************************************
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def russian_roulette(np.ndarray[double, ndim=1] weight, np.ndarray[double, ndim=1] weight0):
    cdef int numNeutrons = weight.shape[0]
    cdef double terminateP
    for iNeutron in range(numNeutrons):
        terminateP = 1 - weight[iNeutron] / weight0[iNeutron]
        if terminateP >= np.random.rand():
            weight[iNeutron] = 0  # killed
        elif terminateP > 0:
            weight[iNeutron] = weight0[iNeutron]  # restore the weight
    return weight
