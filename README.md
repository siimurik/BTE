# BTE
Files for the master thesis on the topic named:
### Development of a simplified Monte Carlo Neutron Transport routine in Python

# Current progress (MATLAB -> Python):
* File **convertGSXtoCSV.m** has successfully been converted into a Python file named **convertGSX2CSV.py**.
* File **convertCSVtoM.m** depends on three main functions: **extractNwords()**, **extract_mf3()** and **extract_mf6()**. These functions are located in a Jupyter Notebook named **testPython.ipynb**. A MATLAB file named **testMATLAB.m** has also been added as a testing ground for secluded functions. Current attempts to run examples of **extract_mf6()** in MATLAB have been unsuccesful. Out of those three functions two are working at the moment:
  - **extractNwords()**;  [Converted successfully]
  - **extract_mf3()**;    [Converted successfully]
  - **extract_mf6()**;    [Work in progress]
