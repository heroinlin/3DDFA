'''
python setup.py build_ext -i
to compile
'''

# setup.py
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
name='mesh_core_cython',
cmdclass={'build_ext': build_ext},
ext_modules=[Extension("mesh_core_cython",
sources=["mesh_core_cython.pyx", "mesh_core.cpp"],
language='c++',
extra_compile_args=['-stdlib=libc++', '-mmacosx-version-min=10.9','-std=c++11', '-D_hypot=hypot'],
extra_link_args=['-stdlib=libc++', '-mmacosx-version-min=10.9','-std=c++11', '-D_hypot=hypot'],
include_dirs=[numpy.get_include()])],
)
