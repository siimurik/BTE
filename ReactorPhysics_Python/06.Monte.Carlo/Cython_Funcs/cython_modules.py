from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

cython_modules = [
    Extension("sample_direction", sources=["sample_direction.pyx"]),
    Extension("move_neutron", sources=["move_neutron.pyx"]),
    Extension("russian_roulette", sources=["russian_roulette.pyx"]),
    Extension("split_neutrons", sources=["split_neutrons.pyx"]),
    Extension("calculate_keff_cycle", sources=["calculate_keff_cycle.pyx"]),
    Extension("update_indices", sources=["update_indices.pyx"]),
    Extension("perform_collision", sources=["perform_collision.pyx"]),


]

setup(ext_modules=cythonize(cython_modules), include_dirs=[np.get_include()])
