import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def perform_collision(bint virtualCollision, bint absorbed, 
                        np.ndarray[double, ndim=2] SigS, double SigA, double SigP, 
                        np.ndarray[double, ndim=1] SigTmax, 
                        np.ndarray[np.int64_t, ndim=1] iGroup, 
                        int iNeutron, 
                        np.ndarray[double, ndim=1] detectS, 
                        np.ndarray[double, ndim=1] weight, 
                        np.ndarray[double, ndim=1] fuel_chi):
    cdef double SigS_sum = np.sum(SigS)
    # ... total
    cdef double SigT = SigA + SigS_sum
    # ... virtual
    cdef double SigV = SigTmax[iGroup[iNeutron]] - SigT

    # Sample the type of the collision: virtual (do nothing) or real
    if SigV / SigTmax[iGroup[iNeutron]] >= np.random.rand():  # virtual collision

        virtualCollision = True

    else:  # real collision

        virtualCollision = False

    # Sample type of the collision: scattering or absorption
    if SigS_sum / SigT >= np.random.rand():  # isotropic scattering

        # Score scatterings with account for weight divided by the
        # total scattering cross section
        detectS[iGroup[iNeutron]] += weight[iNeutron] / SigS_sum

        # Sample the energy group of the secondary neutron
        iGroup[iNeutron] = np.argmax(np.cumsum(SigS) / SigS_sum >= np.random.rand())

    else:  # absorption

        absorbed = True

        # Neutron is converted to the new fission neutron with
        # the weight increased by eta
        weight[iNeutron] *= SigP / SigA

        # Sample the energy group for the new-born neutron
        iGroup[iNeutron] = np.argmax(np.cumsum(fuel_chi) >= np.random.rand())

    return absorbed, virtualCollision, iGroup, weight, detectS