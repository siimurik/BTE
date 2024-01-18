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
def update_indices(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[np.int64_t, ndim=1] iGroup, np.ndarray[double, ndim=1] weight):
    # Get the indices of non-zero weight
    indices = np.nonzero(weight)[0]

    # Perform indexing
    x = x[indices]
    y = y[indices]
    iGroup = iGroup[indices]
    weight = weight[indices]
    
    # Update numNeutrons
    numNeutrons = weight.shape[0]
    
    return x, y, iGroup, weight, numNeutrons
