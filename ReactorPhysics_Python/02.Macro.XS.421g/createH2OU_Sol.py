import re
import h5py
import numpy as np
import scipy as sp
from numba import jit
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam

#=============================================================================================================
"""
=================================================================
Documentation of get_subg_data_as_array() function... sort of
------------------------------------------------------------------
This function aims to combine multiple subgroups of a given 
group into a single NumPy 2D array. It is specifically designed 
to work with subgroups that follow a naming convention like 
'sigT(0,:)', 'sigT(1,:)', ..., 'sigT(9,:)'.

Parameters:
- group: h5py.Group
The group object representing the parent group containing the 
subgroups to be combined.

Returns:
- combined_data: numpy.ndarray
A NumPy 2D array containing the combined data from all the 
subgroups.

Please Note:
The function assumes that the subgroups follow a specific naming 
convention where the subgroups are named 'sigT(0,:)', 'sigT(1,:)',
..., 'sigT(9,:)'.
The combined_data array returned by the function is structured as 
np.array([[data]]), which is useful for further operations like 
np.concatenate().
===================================================================
"""
def get_subg_data_as_array(group):
    subgroups_data = []

    def get_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            subgroup_data = np.array(obj)
            subgroups_data.append(subgroup_data)

    group.visititems(get_data)
    return np.array(subgroups_data) # The output is np.array([[data]]) 
                                    # That is useful for np.concatenate() 
#=============================================================================================================
"""
The 'sigmaZeros()' function calculates the background cross sections, 
sigma-zeros, based on given input parameters. 

**Inputs:**
- 'sigTtab': A cell array containing matrices of total microscopic 
    cross sections for each isotope. Each matrix has dimensions 
    (nsigz x ng), where nsigz is the number of base points 
    sigma-zeros and ng is the number of energy groups.
- 'sig0tab': A cell array containing vectors of base points of 
    tabulated sigma-zeros for each energy group.
- 'aDen': A vector of atomic densities of isotopes.
- 'SigEscape': The escape cross section (1/cm) for simple convex objects.

**Outputs:**
- 'sig0': A 2D matrix of sigma-zeros with dimensions (nIso x ng).
- 'sigT': A 2D matrix of total macroscopic cross sections corrected 
    with account for sigma-zero, with dimensions (nIso x ng).

The function uses the input parameters to calculate the sigma-zeros and 
the corrected total macroscopic cross sections.
------------------------------------------------------------------------
More on sigma0:
In the context of radiation transport and cross-section data, "sigma-zero" 
refers to the base point or reference value of the cross section at a 
specific energy group. It represents the background or base cross section 
for a given interaction process between particles and materials.

The sigma-zero value is used as a reference point for interpolating or 
extrapolating the cross section values at different energies or sigma-zero 
points. By knowing the sigma-zero value and the variation of the cross 
section with respect to sigma-zero, one can calculate the cross section at 
any desired energy or sigma-zero value.

In practical terms, sigma-zero allows for the normalization and adjustment 
of cross-section data to a common reference point. It helps characterize 
the behavior of cross sections as a function of energy and provides a 
consistent framework for comparing and analyzing different materials 
and isotopes.

By accounting for sigma-zero, one can accurately determine the total 
macroscopic cross sections, which play a crucial role in calculations 
related to radiation shielding, dose calculations, reactor design, and 
other applications in nuclear physics and radiation science.
"""
def sigmaZeros(sigTtab, sig0tab, aDen, SigEscape):
    # Number of energy groups
    ng = 421

    # Define number of isotopes in the mixture
    nIso = len(aDen)

    # Define the size of sigT and a temporary value 
    # named sigT_tmp for interpolation values
    sigT = np.zeros((nIso, ng))
    sigT_tmp = 0

    # first guess for sigma-zeros is 1e10 (infinite dilution)
    sig0 = np.ones((nIso, ng)) * 1e10

    # Loop over energy group
    for ig in range(ng):
        # Error to control sigma-zero iterations
        err = 1e10
        nIter = 0

        # sigma-sero iterations until the error is below selected tolerance (1e-6)
        while err > 1e-6:
            # Loop over isotopes
            for iIso in range(nIso):
                # Find cross section for the current sigma-zero by interpolating
                # in the table
                if np.count_nonzero(sig0tab[iIso]) == 1:
                    sigT[iIso, ig] = sigTtab[iIso][0, ig]
                else:
                    log10sig0 = np.minimum(10, np.maximum(0, np.log10(sig0[iIso, ig])))
                    sigT_tmp = sp.interpolate.interp1d( np.log10(sig0tab[iIso][np.nonzero(sig0tab[iIso])]), 
                                                    sigTtab[iIso][:, ig][np.nonzero(sigTtab[iIso][:, ig])], 
                                                    kind='linear')(log10sig0)
                    sigT[iIso, ig] = sigT_tmp
                #sigT = sigT.item() 
                #sigTtab[iIso][np.isnan(sigTtab[iIso])] = 0  # not sure if 100% necessary, but a good mental check

            err = 0
            # Loop over isotopes
            for iIso in range(nIso):
                # Find the total macroscopic cross section for the mixture of
                # the background isotopes
                summation = 0
                # Loop over background isotopes
                for jIso in range(nIso):
                    if jIso != iIso:
                        summation += sigT[jIso, ig] * aDen[jIso]

                tmp = (SigEscape + summation) / aDen[iIso]
                err += (1 - tmp / sig0[iIso, ig])**2
                sig0[iIso, ig] = tmp

            err = np.sqrt(err)
            nIter += 1
            if nIter > 100:
                print('Error: too many sigma-zero iterations.')
                return

    return sig0

#=============================================================================================================
"""
==========================================================
prepareInto3D() Function Documentation
----------------------------------------------------------
This function is designed to prepare and reshape three 
input arrays, 'sigA_H01', 'sigA_O16', and 'sigA_U235', 
into a 3D NumPy array. The function performs the following 
steps to achieve the desired transformation:

**Inputs:**
    - sigA_H01: numpy.ndarray
        The input array representing sigA data for H01.
    - sigA_O16: numpy.ndarray
        The input array representing sigA data for O16.
    - sigA_U235: numpy.ndarray
        The input array representing sigA data for U235.

**Output:**
    - result3D: numpy.ndarray
        A 3D NumPy array with the transformed data.

Description:
The prepareInto3D() function concatenates the three input arrays, 
'sigA_H01', 'sigA_O16', and 'sigA_U235', along the 0th axis, 
resulting in a 2D array named 'data2D'. It then reshapes the 
'data2D' array into a 3D array, 'result3D', where each cell in 
the 0th axis corresponds to one of the input arrays.

The resulting 'result3D' array has the shape 
(num_cells, max(num_rows_per_cell), num_cols), where:
- num_cells: The number of input arrays (in this case, 3).
- max(num_rows_per_cell): The maximum number of rows among the input arrays.
- num_cols: The number of columns in the 'data2D' array.

The function creates an empty 3D array, 'result3D', with the 
desired shape and then fills it with the data from the 'data2D' 
array. The data is copied into the 'result3D' array, cell by 
cell, while maintaining the original shape of each input array.
==========================================================
"""
def prepareInto3D(sigA_H01, sigA_O16, sigA_U235):
    data2D = np.concatenate([sigA_H01, sigA_O16, sigA_U235], axis=0)
    
    # Reshape the sigTtab from 2D into a 3D array
    num_cells = 3
    num_rows_per_cell = [sigA_H01.shape[0], sigA_O16.shape[0], sigA_U235.shape[0]]  # Number of rows per cell; 1, 6, 10
    num_cols = data2D.shape[1]  # Number of columns

    # Create an empty 3D array with the desired shape
    result3D = np.zeros((num_cells, max(num_rows_per_cell), num_cols))

    # Fill the 3D array with data from the 2D matrix
    row_start = 0
    for cell in range(num_cells):
        num_rows = num_rows_per_cell[cell]
        result3D[cell, :num_rows, :] = data2D[row_start:row_start+num_rows, :]
        row_start += num_rows

    return result3D
#=============================================================================================================
"""
=======================================================================================
Documentation for "interpolate_data()"
---------------------------------------------------------------------------------------
The 'kind='linear'' argument in 'interp1d' specifies the type of interpolation to be 
performed. In this case, it indicates that linear interpolation should be used. 
Linear interpolation calculates the values between two known data points as a 
straight line. 

Regarding "fill_value='extrapolate'", it is used to enable extrapolation of values 
outside the range of the given data. By default, 'interp1d' raises an error if you 
try to interpolate/extrapolate outside the range of the input data. Setting 
'fill_value='extrapolate'' allows the function to extrapolate values beyond the 
given data range.
=======================================================================================
"""
def interpolate_data(x, y, xi):
    interp_func = sp.interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
    return interp_func(xi)

#=============================================================================================================
"""
======================================================================================
Function Documentation interpSigS() 
--------------------------------------------------------------------------------------
 The interpSigS function performs interpolation to calculate the scattering matrix 
 sigS based on provided input parameters.
 
 **Inputs**:
 - 'jLgn': An integer representing the index of the energy group.
 - 'element': A string specifying the element.
 - 'Sig0': A numpy array representing the sigma-zero values for target points.

 **Outputs:**
 - 'sigS': A numpy array representing the resulting scattering matrix.
======================================================================================
 The 'interpSigS()' function calculates and returns the scattering matrix 'sigS'. 
 This matrix is obtained through interpolation between scattering matrices ('s_SigS') 
 that correspond to specific sigma-zero base points. The interpolation is performed 
 for a set of target points represented by the vector 'Sig0', which has a length of 
 'ng' (the number of energy groups).
 
 In other words, the function takes as input the base scattering matrices for different 
 sigma-zero values. These matrices capture the scattering behavior of a material under 
 different conditions. The function then uses these base matrices to estimate the 
 scattering behavior at target points specified by 'Sig0'.
 
 By performing interpolation, the function infers the scattering matrix values at the 
 target points based on the known scattering matrices for the sigma-zero base points. 
 The resulting 'sigS' matrix provides an approximation of the scattering behavior at 
 the target points, enabling further analysis or calculations involving the material's 
 scattering properties.
======================================================================================
"""
def interpSigS(jLgn, element, Sig0):
    # Number of energy groups
    ng = 421
    elementDict = {
    'H01':  'H_001',
    'O16':  'O_016',
    'U235': 'U_235',
    }
    # Path to microscopic cross section data:
    micro_XS_path = '../01.Micro_Python'
    # Open the HDF5 file based on the element
    filename = f"micro_{elementDict[element]}__294K.h5"
    # *May be a stupid way to do it, but seems more intuitive for me:
    # Essentailly, only the file for the appropriate element needs to 
    # be opened in order to read the necessary data. I remember 
    # elements more easily if they are written as 'H01' rather than
    # 'H_001', but the last method is how the files are named.
    with h5py.File(micro_XS_path + '/' + filename, 'r') as f:
        s_sig0 = np.array(f.get('sig0_G').get('sig0'))
        findSigS = list(f.get('sigS_G').items())
        # ..., ('sigS(2,5)', <HDF5 dataset "sigS(2,5)": shape (421, 421), type "<f8">)]
        string = findSigS[-1][0]  # 'sigS(2,5)'

        # Extract numbers using regular expression pattern
        pattern = r"sigS\((\d+),(\d+)\)"
        match = re.search(pattern, string)
        # In order to get the 'x' and 'y' dimensions, we check the 'string' for a 
        # certain pattern with 'pattern' to extract the numbers for the max dim size.

        if match:
            x_4D = int(match.group(1)) + 1
            y_4D = int(match.group(2)) + 1
        else:
            print("No match found.")

        # Create the empty 3D numpy array
        s_sigS = np.zeros((x_4D, y_4D, ng, ng))

        # Access the data from the subgroups and store it in the 3D array
        for i in range(x_4D):
            for j in range(y_4D):
                dataset_name = f'sigS({i},{j})'
                s_sigS[i, j] = np.array(f.get('sigS_G').get(dataset_name))
                
        # Number of sigma-zeros
        nSig0 = len(s_sig0)

        if nSig0 == 1:
            sigS = s_sigS[jLgn][0]
        else:
            tmp1 = np.zeros((nSig0, sp.sparse.find(s_sigS[jLgn][0])[2].shape[0]))
            for iSig0 in range(nSig0):
                ifrom, ito, tmp1[iSig0, :] = sp.sparse.find(s_sigS[jLgn][iSig0])

            # Number of non-zeros in a scattering matrix
            nNonZeros = tmp1.shape[1]
            tmp2 = np.zeros(nNonZeros)
            for i in range(nNonZeros):
                log10sig0 = min(10, max(0, np.log10(Sig0[ifrom[i]])))
                # Usually I would use the SciPy interp1d() here, but it is PAINFULLY slow
                # for this code. However if ndim == 1 then I can use the NumPy alternative 
                # which can be 5 to 6 times faster than the SciPy function.
                tmp2[i] = np.interp(np.log10(log10sig0), np.log10(s_sig0), tmp1[:, i])

            sigS = sp.sparse.coo_matrix((tmp2, (ifrom, ito)), shape=(ng, ng)).toarray()

    return sigS

#=============================================================================================================
"""
==========================================================
 writeMacroXS() Function Documentation
----------------------------------------------------------
 This function writes all group macroscopic cross sections
 from a HDF5.h5 structure 's_filename' to a HDF5 file with
 the name stored in matName.

**Inputs:**
    - s_filename: HDF5 file
    - matName: string

**Output:**
    - "matName.h5": HDF5 file
==========================================================
"""
def writeMacroXS(s_filename, matName):
    print(f'Write macroscopic cross sections to the file: {matName}.h5')
    
    # Open the input HDF5 file
    with h5py.File(s_filename, 'r') as s_file:
        # Create the output HDF5 file
        with h5py.File(matName + '.h5', 'w') as f:
            # Read and write the header as attributes of the root group
            header_keys = [key for key in s_file.attrs.keys() if key.startswith('header')]
            # header_keys = ['header0', 'header1', 'header2', 'header3', 'header4', 'header5', 
            # 'header6', 'header7', 'header8', 'header9']
            for i, key in enumerate(header_keys):
                f.attrs[key] = s_file.attrs[key]

            # Read and write other parameters as datasets
            f.create_dataset('aw', data=s_file['aw'][()])
            f.create_dataset('den', data=s_file['den'][()])
            f.create_dataset('temp', data=s_file['temp'][()])
            f.create_dataset('ng', data=s_file['ng'][()])
            f.create_dataset('eg', data=s_file['eg'][()])
            f.create_dataset('isoName', data=s_file['isoName'][()])
            f.create_dataset('numDen', data=s_file['numDen'][()])

            if 'sig0' in s_file:
                f.create_dataset('sig0', data=s_file['sig0'][()])

            f.create_dataset('SigC', data=s_file['SigC'][()])
            f.create_dataset('SigL', data=s_file['SigL'][()])

            s_SigS = np.zeros((3, 421, 421))
            for i in range(3):
                s_SigS[i] = s_file[f'SigS{i}'][()]

            SigS_G = f.create_group("sigS_G")
            ng = s_file['ng'][()]
            #print(s_SigS.shape)
            for j in range(s_SigS.shape[0]):
            #    for j in range(s_SigS.shape[0]):
                Sig = np.zeros(sp.sparse.find(s_SigS[j])[2].shape[0])
                ito, ifrom, Sig = sp.sparse.find(s_SigS[j])
                sigS_sparse = sp.sparse.coo_matrix((Sig, (ifrom, ito)), shape=(ng, ng))
                sigS_new = sigS_sparse.toarray()
                SigS_G.create_dataset(f"Sig({j})", data=Sig)
                SigS_G.create_dataset(f"sparse_SigS({j})", data=sigS_new)
            SigS_G.create_dataset("ifrom", data=ifrom)
            SigS_G.create_dataset("ito", data=ito)

            # (n,2n) matrix for 1 Legendre component
            s_Sig2 = s_file['Sig2'][()]
            Sig2_G = f.create_group("sig2_G")
            Sig = np.zeros(sp.sparse.find(s_Sig2)[2].shape[0])
            ito, ifrom, Sig = sp.sparse.find(s_Sig2)
            sigS_sparse = sp.sparse.coo_matrix((Sig, (ifrom, ito)), shape=(ng, ng))
            sigS_new = sigS_sparse.toarray()
            Sig2_G.create_dataset("Sig", data=Sig)
            Sig2_G.create_dataset("sparse_Sig2", data=sigS_new)
            Sig2_G.create_dataset("ifrom", data=ifrom)
            Sig2_G.create_dataset("ito", data=ito)
    
            if 'SigP' in s_file:
                f.create_dataset('fissile', data=1)
                f.create_dataset('SigF', data=s_file['SigF'][()])
                f.create_dataset('SigP', data=s_file['SigP'][()])
                f.create_dataset('chi', data=s_file['chi'][()])
            else:
                f.create_dataset('fissile', data=0)
                f.create_dataset('SigF', data=[0] * s_file['ng'][()])
                f.create_dataset('SigP', data=[0] * s_file['ng'][()])
                f.create_dataset('chi', data=[0] * s_file['ng'][()])

            f.create_dataset('SigT', data=s_file['SigT'][()])

    print('Done.')

#=============================================================================================================
"""
===========================================================================
 Documentation for the main() section of the code:
---------------------------------------------------------------------------
 The function reads the MICROscopic group cross sections in Python
 format and calculates from them the MACROscopic cross sections for water
 solution of uranium-235 which is a homogeneous aqueous reactor.
===========================================================================
"""
def main():
    # number of energy groups
    H2OU_ng = 421

    # Path to microscopic cross section data:
    micro_XS_path = '../01.Micro_Python'

    # Call the functions for H2O and B isotopes and store the data in the
    # structures. As an example it is done below for temperature of 294K,
    # pressure of 7 MPa and boron concentration of 760 ppm.
    # Change when other parameters needed.
    # Open the HDF5 files
    hdf5_H01 = h5py.File(micro_XS_path + '/micro_H_001__294K.h5', 'r')
    print(f"File 'micro_H_001__294K.h5' has been read in.")

    hdf5_O16 = h5py.File(micro_XS_path + '/micro_O_016__294K.h5', 'r')
    print(f"File 'micro_O_016__294K.h5' has been read in.")

    hdf5_U235 = h5py.File(micro_XS_path + '/micro_U_235__294K.h5', 'r')
    print(f"File 'micro_U_235__294K.h5' has been read in.")

    H2OU_temp = 294  # K
    H2OU_p = 0.1  # MPa
    H2OU_Uconc = 1000e-6  # 1e-6 = 1 ppm
    H2OU_eg = np.array(hdf5_H01.get('en_G').get('eg'))   # From the group 'en_G' get subgroup containing the data named 'eg'

    # Get the atomic weight from the metadata
    H01_aw  = hdf5_H01.attrs.get('atomic_weight_amu')
    O16_aw  = hdf5_O16.attrs.get('atomic_weight_amu')
    U235_aw = hdf5_U235.attrs.get('atomic_weight_amu')
    #print('\natomic_weight of H01: ', H01_aw,'\ndata type:', type(H01_aw))

    # Mass of one "average" H2OU molecule in atomic unit mass [a.u.m.]:
    H2OU_aw = 2 * H01_aw + O16_aw + H2OU_Uconc * U235_aw

    # The function returns water density at specified pressure (MPa) and
    # temperature (C):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
    density = steamTable.rho_pt(H2OU_p/10, H2OU_temp-273)

    # The water density:
    H2OU_den = density*1e-3  # [g/cm3]
    rho = H2OU_den*1.0e-24  # [g/(barn*cm)]
    rho = rho / 1.660538e-24  # [(a.u.m.)/(barn*cm)]
    rho = rho / H2OU_aw  # [number of H2O molecules/(barn*cm)]

    # The names of fissionable isotopes and oxygen
    H2OU_isoName = ['H01', 'O16', 'U235']

    # The number densities of isotopes:
    H2OU_numDen = np.array([2*rho, rho, rho*H2OU_Uconc])                     

    # Get total micrscopic cross-section data 
    sigT_H01  = get_subg_data_as_array(hdf5_H01.get('sigT_G'))
    sigT_O16  = get_subg_data_as_array(hdf5_O16.get('sigT_G'))
    sigT_U235 = get_subg_data_as_array(hdf5_U235.get('sigT_G'))

    # Prepare for sigma-zero iterations:
    sigTtab = prepareInto3D(sigT_H01, sigT_O16, sigT_U235)

    # Explanation by Rodrigo on "sigma zero" or sig0:
    # “Sigma zero” is what is called the “background cross section”.
    # Nuclei are subject to something called “self-shielding”, which means that the 
    # more of a particular nuclei you have, the smaller it’s effective cross-section gets. 
    # This is because the neutron cannot exercise perfect degree of freedom during collision. 
    # However, determining self-shielding by the quantity of a particular nuclei is difficult, 
    # instead what you can do is look at the whole mix and see how much of that nuclei you have 
    # in the mix. So it’s the complete opposite idea, in one you care about how many of a 
    # particular nuclei you have, in the other you care about everything that is not that 
    # particular nuclei.
    # This idea is the background cross-section. It’s a way to account for how the environment 
    # affects the effective cross-section of a nuclei.
    sig0_H01  = np.array(hdf5_H01.get( 'sig0_G').get('sig0'))
    sig0_O16  = np.array(hdf5_O16.get( 'sig0_G').get('sig0'))
    sig0_U235 = np.array(hdf5_U235.get('sig0_G').get('sig0'))
    sig0tab = np.concatenate([sig0_H01, sig0_O16, sig0_U235], axis=0)

    # Determine the length of each sig0_* variable
    sig0_sizes = [len(sig0_H01), len(sig0_O16), len(sig0_U235)]
    isotope = ["H01", "O16", "U235"]

    # Create a 2D array where the number of columns is determined by the 
    # length of the sig0_* variables
    sig0tab = np.zeros((3, max(sig0_sizes)))    # (3,10)
    col_start = 0
    for i, size in enumerate(sig0_sizes):
        sig0tab[i, col_start:col_start+size] = eval(f'sig0_{isotope[i]}')

    # The number densities of isotopes, but in a new variable
    aDen = H2OU_numDen

    # SigEscape -- escape cross section, for simple convex objects (such as
    # plates, spheres, or cylinders) is given by S/(4V), where V and S are the
    # volume and surface area of the object, respectively
    SigEscape = 0

    print('Sigma-zero iterations. ')
    H2OU_sig0 = sigmaZeros(sigTtab, sig0tab, aDen, SigEscape)
    print('Done.')

    print("Interpolation of microscopic cross sections for the found sigma-zeros.")
    # sigC: This represents the capture cross section. It indicates the probability 
    # of a nucleus capturing a neutron without causing fission or scattering.
    sigC_H01 = get_subg_data_as_array(hdf5_H01.get(  'sigC_G'))
    sigC_O16 = get_subg_data_as_array(hdf5_O16.get(  'sigC_G'))
    sigC_U235 = get_subg_data_as_array(hdf5_U235.get('sigC_G'))

    # sigL: This represents the non-fissionable (non-leakage) absorption cross section. 
    # It quantifies the probability of a neutron being absorbed by the nucleus without 
    # causing fission or escaping from the system.
    sigL_H01 =  np.array(hdf5_H01.get('sigL_G').get('sigL'))
    sigL_O16 =  np.array(hdf5_O16.get('sigL_G').get('sigL'))
    sigL_U235 = np.array(hdf5_U235.get('sigL_G').get('sigL'))

    # sigF: This represents the fission cross section. It characterizes the probability 
    # of a nucleus undergoing fission when it absorbs a neutron.
    sigF_H01 = get_subg_data_as_array(hdf5_H01.get(  'sigF_G'))
    sigF_O16 = get_subg_data_as_array(hdf5_O16.get(  'sigF_G'))
    sigF_U235 = get_subg_data_as_array(hdf5_U235.get('sigF_G'))

    sigCtab = prepareInto3D(sigC_H01, sigC_O16, sigC_U235)
    sigLtab = prepareInto3D(sigL_H01, sigL_O16, sigL_U235)
    sigFtab = prepareInto3D(sigF_H01[0], sigF_O16[0], sigF_U235) # I don't know why, I don't know how but sigF_H01 and sigF_O16 are 3D, 
                                                                # but should be 2D...
                                                                # Nevertheless I only need to use their first cell to get the 2D matrix

    sigC = np.zeros((3, H2OU_ng))
    sigF = np.zeros((3, H2OU_ng))
    sigL = np.zeros((3, H2OU_ng))

    for ig in range(H2OU_ng):
        # Loop over isotopes
        for iIso in range(3):
            # Find cross sections for the sigma-zero
            if np.count_nonzero(sig0tab[iIso]) == 1:
                sigC[iIso, ig] = sigCtab[iIso][0, ig]
                sigL[iIso, ig] = sigLtab[iIso][0, ig]
                sigF[iIso, ig] = sigFtab[iIso][0, ig]
            else:
                log10sig0 = min(10, max(0, np.log10(H2OU_sig0[iIso, ig])))  #sig0tab[1][np.nonzero(sig0tab[1])]
                arrayLength = len(sig0tab[iIso][np.nonzero(sig0tab[iIso])])
                #sigC[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][np.nonzero(sig0tab[iIso])]), sigCtab[iIso][:, ig][np.nonzero(sigCtab[iIso][:, ig])], log10sig0)
                sigC[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][:arrayLength]), sigCtab[iIso][:arrayLength, ig], log10sig0)
                sigL[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][:arrayLength]), sigLtab[iIso][:arrayLength, ig], log10sig0) # May need fixing but work at the moment
                sigF[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][:arrayLength]), sigFtab[iIso][:arrayLength, ig], log10sig0) # May need fixing but work at the moment
                # The lines also work, but give warnings bc log10(0.0) was encountered.
                #sigC[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][:]), sigCtab[iIso][:, ig], log10sig0)
                #sigL[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][:]), sigLtab[iIso][:, ig], log10sig0) # May need fixing but work at the moment
                #sigF[iIso, ig] = interpolate_data(np.log10(sig0tab[iIso][:]), sigFtab[iIso][:, ig], log10sig0) # May need fixing but work at the moment
                
                # **Some helpful information for the lines after the 'else' statement:**
                #-----
                # You might be wondering why [:arrayLength] is used instead of [:]. This is to solve two separate problems
                # with one stone. 
                # 
                # The first issue becomes apparent when all elements in a row from sig0tab are used. The
                # 2D matrix sig0tab contains nonzero values in 3 separate rows with 3 separate dimensions: [1, 6, 10].
                # All the empty slots that don't contain any useful values are filled with zeros like so:
                #  [[1.e+10, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00],
                #   [1.e+10, 1.e+04, 1.e+03, 1.e+02, 1.e+01, 1.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00],
                #   [1.e+10, 1.e+04, 3.e+03, 1.e+03, 3.e+02, 1.e+02, 6.e+01, 3.e+01, 1.e+01, 1.e+00]]
                # As it is apparent from the code snippet above that we need to find a log10() of these nonzero values.
                # Obviously we can't take a log10() of the value 0.0, but Python usually handles this with a simple
                # warning. If we want to get rid of the warning in a clever way, we only need to count how many elements
                # in a single array are not equal to zero and then count up only to that value and no more, hence we use
                # ...][:arrayLength]. 
                # 
                # The second problem becomes apparent now, because sig*tab[] can contain zero elements
                # for some rows and not for others. Meaning we also need to define the correct length of values for 
                # how many elements we take from sig*tab. Since the inputs x and y in the SciPy interp1d() need to 
                # have the same length, it means we only need to take into account only as many elements that are in x, 
                # hence we also use ...][:arrayLength, ig] for sig*tab. 

    # Preallocate the array with zeros
    sigS = np.zeros((3, 3, 421, 421))
    for j in range(3):
        sigS[j, 0, :, :] = interpSigS(j, 'H01',  H2OU_sig0[0, :])
        sigS[j, 1, :, :] = interpSigS(j, 'O16',  H2OU_sig0[1, :])
        sigS[j, 2, :, :] = interpSigS(j, 'U235', H2OU_sig0[2, :])

    #print(sigS.shape)
    print('Done.')

    # Macroscopic cross section [1/cm] is microscopic cross section for the 
    # molecule [barn] times the number density [number of molecules/(barn*cm)]
    H2OU_SigC = np.transpose(sigC) @ aDen
    H2OU_SigL = np.transpose(sigL) @ aDen
    H2OU_SigF = np.transpose(sigF) @ aDen
    U235_nubar = get_subg_data_as_array((hdf5_U235.get('nubar_G')))
    H2OU_SigP = U235_nubar * sigF[2, :] * aDen[2] 

    H2OU_SigS = np.zeros((3, 421, 421))
    for j in range(3):
        H2OU_SigS[j] = sigS[j, 0]*aDen[0] + sigS[j, 1]*aDen[1] + sigS[j, 2]*aDen[2]

    H01_sig2  = np.array(hdf5_H01.get('sig2_G').get('sig2'))
    O16_sig2  = np.array(hdf5_O16.get('sig2_G').get('sig2'))
    U235_sig2 = np.array(hdf5_U235.get('sig2_G').get('sig2'))

    H2OU_Sig2 = H01_sig2 * aDen[0] + O16_sig2 * aDen[1] + U235_sig2 * aDen[2]
    H2OU_SigT = H2OU_SigC + H2OU_SigL + H2OU_SigF + np.sum(H2OU_SigS[0], axis=1) + np.sum(H2OU_Sig2, axis=1)

    # Fission spectrum
    U235_chi = get_subg_data_as_array((hdf5_U235.get('chi_G')))
    H2OU_chi = U235_chi

    # Change the units of number density from 1/(barn*cm) to 1/cm2
    H2OU_numDen = H2OU_numDen*1e24

    # Make a file name which includes the isotope name and the temperature
    if H2OU_temp < 1000:
        matName = f"macro421_H2OU__{round(H2OU_temp)}K"  # name of the file with a temperature index
    else:
        matName = f"macro421_H2OU_{round(H2OU_temp)}K"  # name of the file with a temperature index

    # Create the HDF5 file
    with h5py.File("H2OU.h5", 'w') as hdf:
        # Make a header for the file to be created with important parameters
        header = [
            '---------------------------------------------------------',
            'Python-based Open-source Reactor Physics Education System',
            '---------------------------------------------------------',
            'Author: Siim Erik Pugal',
            '',
            'Macroscopic cross sections for water solution of uranium-235',
            f'Water temperature:   {H2OU_temp:.1f} K',
            f'Water pressure:      {H2OU_p:.1f} MPa',
            f'Water density:       {H2OU_den:.5f} g/cm3',
            f'U-235 concentration:  {H2OU_Uconc*1e6:.1f} ppm'
        ]

        # Write the header as attributes of the root group
        for i, line in enumerate(header):
            hdf.attrs[f'header{i}'] = line

        # Write the macroscopic cross sections as datasets
        hdf.create_dataset('ng', data=H2OU_ng)
        hdf.create_dataset('temp', data=H2OU_temp)
        hdf.create_dataset('p', data=H2OU_p)
        hdf.create_dataset('Uconc', data=H2OU_Uconc)
        hdf.create_dataset('eg', data=H2OU_eg)
        hdf.create_dataset('aw', data=H2OU_aw)
        hdf.create_dataset('den', data=H2OU_den)
        hdf.create_dataset('isoName', data=H2OU_isoName)
        hdf.create_dataset('numDen', data=H2OU_numDen)
        hdf.create_dataset('sig0', data=H2OU_sig0)
        hdf.create_dataset('SigC', data=H2OU_SigC)
        hdf.create_dataset('SigL', data=H2OU_SigL)
        hdf.create_dataset('SigF', data=H2OU_SigF)
        hdf.create_dataset('SigP', data=H2OU_SigP)
        hdf.create_dataset('SigS0', data=np.transpose(H2OU_SigS[0]))
        hdf.create_dataset('SigS1', data=np.transpose(H2OU_SigS[1]))
        hdf.create_dataset('SigS2', data=np.transpose(H2OU_SigS[2]))
        hdf.create_dataset('Sig2', data=np.transpose(H2OU_Sig2))
        hdf.create_dataset('SigT', data=H2OU_SigT)
        hdf.create_dataset('chi', data=H2OU_chi)

    writeMacroXS('H2OU.h5', matName)

    # Close the HDF5 files
    hdf5_H01.close()
    hdf5_O16.close()
    hdf5_U235.close()

if __name__ == '__main__':
    main()