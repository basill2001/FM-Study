from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(name = 'basic_cython',
    cmdclass = {"build_ext":build_ext},
    ext_modules = cythonize("basic_cython.pyx"))