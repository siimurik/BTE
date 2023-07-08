import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def split_neutrons( np.ndarray[double, ndim=1] weight, 
                    int numNeutrons, 
                    np.ndarray[double, ndim=1] x, 
                    np.ndarray[double, ndim=1] y, 
                    np.ndarray[np.int64_t, ndim=1] iGroup):
    cdef int numNew = 0
    for iNeutron in range(numNeutrons):
        if weight[iNeutron] > 1:
            N = int(weight[iNeutron])
            if weight[iNeutron] - N > np.random.rand():
                N += 1
            weight[iNeutron] = weight[iNeutron] / N
            for iNew in range(N - 1):
                numNew += 1
                x = np.append(x, x[iNeutron])
                y = np.append(y, y[iNeutron])
                weight = np.append(weight, weight[iNeutron])
                iGroup = np.append(iGroup, iGroup[iNeutron])
    numNeutrons += numNew
    return weight, numNeutrons, x, y, iGroup
