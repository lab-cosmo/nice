from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

'''import os
extensions = []
files = os.listdir('nice/')
for file in files:
    if (file.endswith(".pyx")):
        name = file[:-4]
        extensions.append(Extension(name, ['nice/' + file], extra_compile_args=['-O3']))'''
            
extensions = [Extension("nice.*",
                  ["nice/*.pyx"],
                  extra_compile_args=['-O3', '-fopenmp'],
                  extra_link_args = ['-fopenmp'])]
setup(
    name='nice',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    zip_safe=False,
)