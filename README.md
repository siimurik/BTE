# BTE
Files for the master thesis on the topic named:
### Development of a simplified Monte Carlo Neutron Transport routine in Python

# Current progress (MATLAB -> Python):
* File **convertGSXtoCSV.m** has successfully been converted into a Python file named **convertGSX2CSV.py**.
* File **convertCSVtoM.m** depends on three main functions: **extractNwords()**, **extract_mf3()** and **extract_mf6()**. These functions are located in a Jupyter Notebook named **testPython.ipynb** in the ***ReactorPhysics_Python*** folder. A MATLAB file named **testMATLAB.m** has also been added as a testing ground for secluded functions. Final code is written in the file **convertCSV2H5.py**. Progress at the moment:
  - **extractNwords()**;  [Converted successfully]
  - **extract_mf3()**;    [Converted successfully]
  - **extract_mf6()**;    [Converted successfully] 
  - **convertCSVtoM.m**:  [Converted successfully]
  * Final file: **convertCSV2H5.py**
---
* Starting work on **createH2OU.m**. Created a test version "_Test.m". Also started to structurize the folders similarly. New testing ground is in folder 02.Macro.XS.421g in file **testMacro.ipynb**. **createH2OU.m** needs 3 functions to work plus a speacial library called XSteam. Current progress:
  - XSteam - A similar library made for Python named [pyXSteam](https://github.com/drunsinn/pyXSteam).
  - sigmaZeros();   [Converted successfully]
  - interpSigS();   [Converted successfully]
  - writeMacroXS(); [Converted successfully]
* **createH2OU.m** has been succesfully converted into Python and now bears the name **createH2OU_Sol.py**. File can be found in the folder 02.Macro.XS.421g.
---
* Next up: starting work on **createPWR_like_mix.m**. Two new functions that need to be rewritten from 00.Lib
  - input_and_initialize_PWR_like()
  - matpro()
  Data that these files create are stored in the same folder, functions are part of the larger code named **createPWR_like_mix.py**
* **createPWR_like_mix.m** finished.
* Significantly optimized the **convertCSV2H5.py** code. Improved version is named **boostedCSV2H5.py** and uses the [Numba](https://numba.readthedocs.io/en/stable/) just-in-time compiler. 
---
Task 1. Download nuclear data from IAEA site.

Folder: 01.Micro.XS.421g

Download the GENDF files for the required isotopes from the open-access 
[IAEA website](https://www-nds.iaea.org/ads/adsgendf.html)

The isotopes used for the PWR-like unit cell calculations:

* B_010.GXS
* B_011.GXS
* H_001.GXS
* O_016.GXS
* U_235.GXS
* U_238.GXS
* ZR090.GXS
* ZR091.GXS
* ZR092.GXS
* ZR094.GXS
* ZR096.GXS

---
