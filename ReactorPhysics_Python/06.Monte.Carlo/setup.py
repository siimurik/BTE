"""
 Run this code with the command:
    > python3 setup.py build_ext --inplace
"""
from distutils.core import setup
from Cython.Build import cythonize

# List of Cython source files or module names
#cython_modules = ['sample_direction.pyx', 'move_neutron.pyx', 'russian_roulette.pyx', 'split_neutrons.pyx', 'calculate_keff_cycle.pyx', 'update_indices.pyx', 'perform_collision.pyx']

cython_modules = ['montepy.pyx']

# Configure the setup options
setup_options = {
    'ext_modules': cythonize(cython_modules)
}

# Run the setup
setup(**setup_options)
