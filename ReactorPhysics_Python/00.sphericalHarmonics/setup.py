#******************************************************************
# This code is released under the GNU General Public License (GPL).
#******************************************************************
# Run with:
#   $ python3 setup.py build_ext --inplace


from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("lebedev.pyx"),
    include_dirs=[np.get_include()]
)
