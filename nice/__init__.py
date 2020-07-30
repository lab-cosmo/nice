'''from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [  
    Extension("*", ["*.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...]),
]
setup(
    name="nice",
    ext_modules=cythonize(extensions),
)'''


'''import cython
import pyximport; pyximport.install()'''
from nice import transformers
from nice import ClebschGordan
from nice import contracted_pca
from nice import parallelized
from nice import test_utilities


from nice import unrolling_individual_pca
from nice import naive
from nice import nice_utilities
from nice import parallelized
from nice import radial_basis
from nice import rascal_coefficients
from nice import spherical_coefficients
from nice import test_utilities
from nice import thresholding
from nice import unrolling_pca
from nice import test_parallel
from nice import packing