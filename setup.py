from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension("nice.*", ["nice/*.pyx"],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'])
]
setup(
    name='nice',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
