from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name='cmr2 test app',

    # insert name of file to compile
    ext_modules = cythonize("CMR2_pack_cyth.pyx"),
    include_dirs=[numpy.get_include()]

)
