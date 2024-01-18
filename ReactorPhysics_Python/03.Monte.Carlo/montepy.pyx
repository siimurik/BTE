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
cpdef sample_direction():
    cdef double teta = np.pi * np.random.rand()
    cdef double phi = 2.0 * np.pi * np.random.rand()
    cdef double dirX = np.sin(teta) * np.cos(phi)
    cdef double dirY = np.sin(teta) * np.sin(phi)
    return dirX, dirY

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef move_neutron( np.ndarray[double, ndim=1] x, 
                    np.ndarray[double, ndim=1] y, 
                    int iNeutron, double pitch, 
                    double freePath, 
                    double dirX, double dirY):
                    
    x[iNeutron] += freePath * dirX
    y[iNeutron] += freePath * dirY

    # If outside the cell, find the corresponding point inside the cell
    while x[iNeutron] < 0:
        x[iNeutron] += pitch
    while y[iNeutron] < 0:
        y[iNeutron] += pitch
    while x[iNeutron] > pitch:
        x[iNeutron] -= pitch
    while y[iNeutron] > pitch:
        y[iNeutron] -= pitch

    return x, y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_cross_sections( double fuelLeft, double fuelRight,
                                double coolLeft, double coolRight,
                                np.ndarray[double, ndim=1] x, int iNeutron,
                                np.ndarray[np.int64_t, ndim=1] iGroup,
                                dict fuel, dict cool, dict clad):
    cdef double SigA, SigP
    cdef np.ndarray[double, ndim=2] SigS

    if fuelLeft < x[iNeutron] < fuelRight:  # INPUT
        SigA = fuel['SigF'][iGroup[iNeutron]] + fuel['SigC'][iGroup[iNeutron]] + fuel['SigL'][iGroup[iNeutron]]
        SigS = fuel["sigS_G"]["sparse_SigS[0]"][iGroup[iNeutron], :].reshape(-1, 1)
        SigP = fuel['SigP'][0, iGroup[iNeutron]]
    elif x[iNeutron] < coolLeft or x[iNeutron] > coolRight:  # INPUT
        SigA = cool['SigC'][iGroup[iNeutron]] + cool['SigL'][iGroup[iNeutron]]
        SigS = cool["sigS_G"]["sparse_SigS[0]"][iGroup[iNeutron], :].reshape(-1, 1)
        SigP = 0
    else:
        SigA = clad['SigC'][iGroup[iNeutron]] + clad['SigL'][iGroup[iNeutron]]
        SigS = clad["sigS_G"]["sparse_SigS[0]"][iGroup[iNeutron], :].reshape(-1, 1)
        SigP = 0

    return SigA, SigS, SigP

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef perform_collision(bint virtualCollision, bint absorbed, 
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef russian_roulette(np.ndarray[double, ndim=1] weight, np.ndarray[double, ndim=1] weight0):
    cdef int numNeutrons = weight.shape[0]
    cdef double terminateP
    for iNeutron in range(numNeutrons):
        terminateP = 1 - weight[iNeutron] / weight0[iNeutron]
        if terminateP >= np.random.rand():
            weight[iNeutron] = 0  # killed
        elif terminateP > 0:
            weight[iNeutron] = weight0[iNeutron]  # restore the weight
    return weight

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef split_neutrons( np.ndarray[double, ndim=1] weight, 
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_indices( np.ndarray[double, ndim=1] x, 
                    np.ndarray[double, ndim=1] y, 
                    np.ndarray[np.int64_t, ndim=1] iGroup, 
                    np.ndarray[double, ndim=1] weight):
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_keff_cycle(int iCycle, int numCycles_inactive, int numCycles_active, np.ndarray[double, ndim=1] weight,
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
            np.sum((keff_active_cycle[:iActive] - keff_expected[iActive-1])** 2) / max(iActive - 1, 1) / iActive)

        msg = 'Active cycle = {:3d}/{:3d}; k-eff cycle = {:.5f}; numNeutrons = {:3d}; k-eff expected = {:.5f}; sigma = {:.5f}'.format(
            iCycle - numCycles_inactive, numCycles_active, keff_cycle, numNeutrons, keff_expected[iActive-1], sigma_keff[iActive-1])
        print(msg)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef main_power_iteration_loop(  int numNeutrons_born, int numCycles_inactive, 
                                int numCycles_active, int numNeutrons, 
                                double pitch, double fuelLeft, double fuelRight,
                                double coolLeft, double coolRight,
                                dict fuel, dict cool, dict clad,
                                np.ndarray[double, ndim=1] weight,
                                np.ndarray[double, ndim=1] SigTmax,
                                np.ndarray[double, ndim=1] detectS,
                                np.ndarray[double, ndim=1] x, 
                                np.ndarray[double, ndim=1] y,
                                np.ndarray[np.int64_t, ndim=1] iGroup,
                                np.ndarray[double, ndim=1] keff_expected,
                                np.ndarray[double, ndim=1] sigma_keff,
                                np.ndarray[double, ndim=1] keff_active_cycle,
                                bint virtualCollision
                                 ):
    
    cdef bint absorbed
    cdef double freePath
    #cdef double dirX, dirY
    cdef double SigA, SigP
    cdef np.ndarray[double, ndim=2] SigS
    cdef np.ndarray[double, ndim=1] weight0

    # Main (power) iteration loop
    for iCycle in range(1, numCycles_inactive + numCycles_active + 1):

        # Normalize the weights of the neutrons to make the total weight equal to
        # numNeutrons_born (equivalent to division by keff_cycle)
        weight = (weight / np.sum(weight, axis=0, keepdims=True)) * numNeutrons_born
        weight0 = weight.copy()
        #----------------------------------------------------------------------
        # Loop over neutrons
        for iNeutron in range(numNeutrons):

            absorbed = False

            #------------------------------------------------------------------
            # Neutron random walk cycle: from emission to absorption

            while not absorbed:

                # Sample free path length according to the Woodcock method
                freePath = -np.log(np.random.rand()) / SigTmax[iGroup[iNeutron]]
                #print(freePath)

                if not virtualCollision:
                    # Sample the direction of neutron flight assuming both
                    # fission and scattering are isotropic in the lab (a strong
                    # assumption!)
                    dirX, dirY = sample_direction()
                    
                # Fly
                x, y = move_neutron(x, y, iNeutron, pitch, freePath, dirX, dirY)

                # Find the total and scattering cross sections                    
                SigA, SigS, SigP = calculate_cross_sections(fuelLeft, fuelRight, coolLeft, coolRight, x, iNeutron, iGroup, fuel, cool, clad)
                
                # Sample the type of the collision: virtual (do nothing) or real
                absorbed, virtualCollision, iGroup, weight, detectS = perform_collision(virtualCollision, absorbed, SigS, SigA, SigP, 
                                                                                    SigTmax, iGroup, iNeutron, detectS, weight, fuel["chi"])
            # End of neutron random walk cycle: from emission to absorption
        # End of loop over neutrons
        #-------------------------------------------------------------------------------------------
        # Russian roulette
        weight = russian_roulette(weight, weight0)

        #-------------------------------------------------------------------------------------------
        # Clean up absorbed or killed neutrons    
        # Convert x and y to NumPy arrays
        #x = np.array(x)
        #y = np.array(y)
        x, y, iGroup, weight, numNeutrons = update_indices(x, y, iGroup, weight)

        #-------------------------------------------------------------------------------------------
        # Split too "heavy" neutrons
        weight, numNeutrons, x, y, iGroup = split_neutrons(weight, numNeutrons, x, y, iGroup)

        #-------------------------------------------------------------------------------------------
        # k-eff in a cycle equals the total weight of the new generation over
        # the total weight of the old generation (the old generation weight =
        # numNeutronsBorn)
        calculate_keff_cycle(iCycle, numCycles_inactive, numCycles_active, weight, weight0, numNeutrons, keff_active_cycle, keff_expected, sigma_keff)

        # End of main (power) iteration