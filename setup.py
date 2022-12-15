"@author = atran"

from distutils.core import setup, Extension
import os

from numpy.distutils.misc_util import *

numpyincl = get_numpy_include_dirs()

cvbldamodule = Extension("cvbLDA",
                    sources = ["cvbLDA.c"],
                    include_dirs = [os.getcwd()] + numpyincl,
                    library_dirs = [],
                    libraries = [],
                    extra_compile_args = ['-O3','-Wall'],
                    extra_link_args = [])

setup(name = 'cvbLDA',
      description = 'Collapsed Variational Bayesian inference for LDA',
      ext_modules = [cvbldamodule],
      py_modules = [])
