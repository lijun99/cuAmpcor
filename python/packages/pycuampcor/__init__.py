"""
pycuampcor: amplitude cross-correlation (ampcor) with CPU and CUDA backends

PyCPUAmpcor : the CPU (OpenMP) implementation, always available
PyCuAmpcor  : the CUDA implementation, available if built with CUDA support

Parameters are set as properties, followed by setupParams() and setConstantGrossOffset()
or setVaryingGrossOffset(), or given to the constructor as keyword arguments, e.g.,
PyCPUAmpcor(windowSizeHeight=64, ..., grossOffsetDown=0, grossOffsetAcross=0),
which sets the object up to run.

Images are raw binary files, or 2D datasets in HDF5 files named as HDF5:<file>:<dataset>
if built with HDF5 support (has_hdf5). The reader of each image is chosen by its name
(referenceImageReader/secondaryImageReader = "auto", default) or set to "raw" or "hdf5".
"""

from ._version import __version__
from ._cpu import PyCPUAmpcor, has_hdf5

try:
    from ._cuda import PyCuAmpcor
    has_cuda = True
except ImportError:
    has_cuda = False
