# Run with:
#   $ python3 setup.py build_ext --inplace


from setuptools import setup
from Cython.Build import cythonize
import numpy as np

#setup(
#    ext_modules=cythonize("montepy.pyx"),
#    include_dirs=[np.get_include()]
#)

setup(
    ext_modules=cythonize("montepy.pyx", 
                        compiler_directives={'language_level': "3"}),   #   -O3
                        include_dirs=[np.get_include()] # Allows the use of NumPy library
)