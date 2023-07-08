import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_keff_cycle(int iCycle, int numCycles_inactive, int numCycles_active, np.ndarray[double, ndim=1] weight,
                         np.ndarray[double, ndim=1] weight0, int numNeutrons, np.ndarray[double, ndim=1] keff_active_cycle,
                         np.ndarray[double, ndim=1] keff_expected, np.ndarray[double, ndim=1] sigma_keff):
    cdef int iActive
    cdef double keff_cycle

    # Calculate k-eff in a cycle
    keff_cycle = np.sum(weight) / np.sum(weight0)

    iActive = iCycle - numCycles_inactive
    if iActive <= 0:
        msg = 'Inactive cycle = {:3d}/{:3d}; k-eff cycle = {:.5f}; numNeutrons = {:3d}'.format(
            iCycle, numCycles_inactive, keff_cycle, numNeutrons)
        print(msg)
    else:
        # Update k-effective of the cycle
        keff_active_cycle[iActive-1] = keff_cycle

        # Update k-effective of the problem
        keff_expected[iActive-1] = np.mean(keff_active_cycle[:iActive])

        # Calculate standard deviation of k-effective
        sigma_keff[iActive-1] = np.sqrt(
            np.sum((keff_active_cycle[:iActive] - keff_expected[iActive-1]) ** 2) / max(iActive - 1, 1) / iActive)

        msg = 'Active cycle = {:3d}/{:3d}; k-eff cycle = {:.5f}; numNeutrons = {:3d}; k-eff expected = {:.5f}; sigma = {:.5f}'.format(
            iCycle - numCycles_inactive, numCycles_active, keff_cycle, numNeutrons, keff_expected[iActive-1], sigma_keff[iActive-1])
        print(msg)
