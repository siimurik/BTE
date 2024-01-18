#******************************************************************
# This code is released under the GNU General Public License (GPL).
#
# Siim Erik Pugal, 2023-2024
#******************************************************************
import h5py
import time as t
import numpy as np
import matplotlib.pyplot as plt
import montepy as mp    # A custom module written with Cython

def hdf2dict(file_name):
    """
    ===========================================================================
    hdf2dict() function documentation
    ---------------------------------------------------------------------------
    Essentially a modified version of the 'read_matpro()' function that 
    converts all the data inside a HDF5 file from a HDF5 dict into a Python
    dict.
    ---------------------------------------------------------------------------
    Parameters:
            file_name (str): The name or path of the HDF5 file to be processed.

    Returns:
        data (dict): A nested dictionary containing the datasets from the HDF5 
                    file. The keys of the top-level dictionary correspond to 
                    dataset names, and the values can be either nested 
                    dictionaries (for struct datasets) or numpy arrays 
                    (for regular datasets).

    Example:
        data = hdf2dict("data.h5")
    ---------------------------------------------------------------------------
        Notes:
            - This function takes an HDF5 file name or path as input.
            - It reads datasets from the file and organizes them into a nested 
                dictionary structure.
            - Each dataset is represented by a key-value pair in the dictionary.
            - If a dataset is a struct (group in HDF5), it is further nested 
                within the dictionary.
            - Regular datasets are stored as numpy arrays.
    ===========================================================================
    """
    # Define the dictionary to store the datasets
    data = {}
    #file_name = 'macro421_UO2_03__900K.h5'
    with h5py.File("..//02.Macro.XS.421g/" + file_name, "r") as file:
        # Iterate over the dataset names in the group
        for dataset_name in file.keys():
            # Read the dataset
            dataset = file[dataset_name]
            # Check if the dataset is a struct
            if isinstance(dataset, h5py.Group):
                # Create a dictionary to store the struct fields
                struct_data = {}
                # Iterate over the fields in the struct
                for field_name in dataset.keys():
                    # Read the field dataset
                    field_dataset = np.array(dataset[field_name])
                    # Store the field dataset in the struct dictionary
                    struct_data[field_name] = field_dataset
                # Store the struct data in the main data dictionary
                data[dataset_name] = struct_data
            else:
                # Read the dataset as a regular array
                dataset_array = np.array(dataset)
                # Store the dataset array in the dictionary
                data[dataset_name] = dataset_array

    return data

def main():
    """
    ===========================================================================
    Documentation for the main() section of the code:
    ---------------------------------------------------------------------------
    Author: Siim Erik Pugal, 2023

    The function calculates the neutron transport in a 2D (x,y) unit cell
    similar to the unit cell of the pressurized water reactor using the Monte
    Carlo method. 
    ---------------------------------------------------------------------------
    This version uses the Cython boosted custom Python module named 'montepy',
    which is supposed to speed up the main for-loop resposible for the Monte
    Carlo method by translating time-consuming functions into optimized C/C++ 
    code and compiling them as Python extension modules.
    ---------------------------------------------------------------------------
    # Without Optimization
        $ real	4m23.505s
        $ user	4m22.852s
        $ sys	0m1.205s

    # After Cython Optimization
        $ real	3m28.593s
        $ user	3m28.118s
        $ sys   0m1.240s
    ===========================================================================
    """
    # Start the timer
    start_time = t.time()

    #--------------------------------------------------------------------------
    # Number of source neutrons
    numNeutrons_born = 100          # INPUT

    # Number of inactive source cycles to skip before starting k-eff accumulation
    numCycles_inactive = 100        # INPUT

    # Number of active source cycles for k-eff accumulation
    numCycles_active = 2000         # INPUT

    # Size of the square unit cell
    pitch = 3.6  # cm               # INPUT

    # Define fuel and coolant regions
    fuelLeft, fuelRight = 0.9, 2.7  # INPUT
    coolLeft, coolRight = 0.7, 2.9  # INPUT

    #--------------------------------------------------------------------------
    # Path to macroscopic cross section data:
    # (Assuming the corresponding data files are available and accessible)
    #macro_xs_path = '..//02.Macro.XS.421g'

    # Fill the structures fuel, clad, and cool with the cross-section data
    fuel = hdf2dict('macro421_UO2_03__900K.h5')  # INPUT
    print(f"File 'macro421_UO2_03__900K.h5' has been read in.")
    clad = hdf2dict('macro421_Zry__600K.h5')     # INPUT
    print(f"File 'macro421_Zry__600K.h5' has been read in.")
    cool  = hdf2dict('macro421_H2OB__600K.h5')   # INPUT
    print(f"File 'macro421_H2OB__600K.h5' has been read in.")

    # Define the majorant: the maximum total cross-section vector
    SigTmax = np.max(np.vstack((fuel["SigT"], clad["SigT"], cool["SigT"])), axis=0)

    # Number of energy groups
    ng = fuel['ng']

    #--------------------------------------------------------------------------
    # Detectors
    detectS = np.zeros(ng)

    #--------------------------------------------------------------------------
    # Four main vectors describing the neutrons in a batch
    x = np.zeros(numNeutrons_born * 2)
    y = np.zeros(numNeutrons_born * 2)
    weight = np.ones(numNeutrons_born * 2)
    iGroup = np.ones(numNeutrons_born * 2, dtype=np.int64)

    #--------------------------------------------------------------------------
    # Neutrons are assumed born randomly distributed in the cell with weight 1
    # with sampled fission energy spectrum
    numNeutrons = numNeutrons_born
    for iNeutron in range(numNeutrons):
        x[iNeutron] = np.random.rand() * pitch
        y[iNeutron] = np.random.rand() * pitch
        weight[iNeutron] = 1
        # Sample the neutron energy group
        iGroup[iNeutron] = np.argmax(np.cumsum(fuel['chi']) >= np.random.rand()) - 1
        #print(f"iNeutron: {iNeutron}, iGroup[iNeutron]: {iGroup[iNeutron]}")

    #--------------------------------------------------------------------------
    # Prepare vectors for keff and standard deviation of keff
    keff_expected = np.ones(numCycles_active)
    sigma_keff = np.zeros(numCycles_active)
    keff_active_cycle = np.ones(numCycles_active)
    virtualCollision = False

    #--------------------------------------------------------------------------
    # Main (power) iteration loop
    mp.main_power_iteration_loop(   numNeutrons_born, numCycles_inactive, 
                                    numCycles_active, numNeutrons, 
                                    pitch,  fuelLeft,  fuelRight,
                                    coolLeft,  coolRight,
                                    fuel, cool, clad,
                                    weight,
                                    SigTmax,
                                    detectS,
                                    x, y,
                                    iGroup,
                                    keff_expected,
                                    sigma_keff,
                                    keff_active_cycle,
                                    virtualCollision )

    # Calculate the elapsed time
    elapsed_time = t.time() - start_time

    # Create a new HDF5 file
    with h5py.File('resultsPWR.h5', 'w') as hdf:
        # Make a header for the file to be created with important parameters
        header = [
                '---------------------------------------------------------',
                'Python-based Open-source Reactor Physics Education System',
                '---------------------------------------------------------',
                '',
                'function s = resultsPWR',
                '% Results for 2D neutron transport calculation in the PWR-like unit cell using method of Monte Carlo',
                f' Number of source neutrons per k-eff cycle is {numNeutrons_born}',
                f' Number of inactive source cycles to skip before starting k-eff accumulation {numCycles_inactive}',
                f' Number of active source cycles for k-eff accumulation {numCycles_active}'
            ]

        # Write the header as attributes of the root group
        for i, line in enumerate(header):
            hdf.attrs[f'header{i}'] = line

        time = hdf.create_group("time")
        time.create_dataset("elapsedTime_(s)", data = elapsed_time)
        time.create_dataset("elapsedTime_(min)", data = elapsed_time/60)
        time.create_dataset("elapsedTime_(hrs)", data = elapsed_time/3600)

        hdf.create_dataset("keff_expected", data = keff_expected[-1])
        hdf.create_dataset("sigma", data = sigma_keff[-1])
        hdf.create_dataset("keffHistory", data = keff_expected)
        hdf.create_dataset("keffError", data = sigma_keff)

        hdf.create_dataset("eg", data = (fuel["eg"][0:ng] + fuel["eg"][1:ng+1]) / 2)

        # Calculate du and flux_du
        du = np.log(fuel["eg"][1:ng+1] / fuel["eg"][:-1])
        flux_du = detectS / du
        hdf.create_dataset("flux", data = flux_du)
        

    # Plot the k-effective
    plt.figure()
    plt.plot(keff_expected, '-r', label='k_eff')
    plt.plot(keff_expected + sigma_keff, '--b', label='k_eff +/- sigma')
    plt.plot(keff_expected - sigma_keff, '--b')
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('k-effective')
    plt.legend()
    plt.tight_layout()
    plt.savefig('MC_01_keff.pdf')

    # Plot the spectrum
    plt.figure()
    plt.semilogx((fuel["eg"][0:ng] + fuel["eg"][1:ng+1]) / 2, flux_du)
    plt.grid(True)
    plt.xlabel('Energy, eV')
    plt.ylabel('Neutron flux per unit lethargy, a.u.')
    plt.tight_layout()
    plt.savefig('MC_02_flux_lethargy.pdf')

    # End of function


if __name__ == '__main__':
    main()