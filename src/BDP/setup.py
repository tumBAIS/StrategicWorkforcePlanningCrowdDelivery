"""
File: psi_calculation.py
Author: Julius Luy, Gerhard Hiermann
Date: November 10th 2023
Description: Technical interface to cython.
"""
# run with parameters: build_ext --inplace
import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='psi_calculation_cy',
    ext_modules=cythonize("psi_calculation_cy.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
