from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name='cmr2 test app',

    # insert name of file to compile
    ext_modules = cythonize("CMR2_pack_cyth_source_3.pyx"),
    include_dirs=[numpy.get_include()]

)
