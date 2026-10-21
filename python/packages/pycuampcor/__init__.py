"""
pycuampcor: amplitude cross-correlation (ampcor) with CPU and CUDA backends

PyCPUAmpcor : the CPU (OpenMP) implementation, always available
PyCuAmpcor  : the CUDA implementation, available if built with CUDA support

Parameters are set as properties, followed by setupParams() and setConstantGrossOffset()
or setVaryingGrossOffset(), or given to the constructor as keyword arguments, e.g.,
PyCPUAmpcor(windowSizeHeight=64, ..., grossOffsetDown=0, grossOffsetAcross=0),
which sets the object up to run.

Options (e.g., workflow, derampMethod) are integer enums of pycuampcor.options
(e.g., Workflow), set by the member, the integer value or the name (e.g., "one_pass").

Images are raw binary files, or 2D datasets in HDF5 files named as HDF5:<file>:<dataset>
if built with HDF5 support (has_hdf5). The reader of each image is chosen by its name
(referenceImageReader/secondaryImageReader = "auto", default) or set to "raw" or "hdf5".

The messages written (progress, warnings) are chosen with set_log_level, e.g.,
set_log_level("warning") for the warnings only.
"""

from ._version import __version__
from .options import (Algorithm, DataType, DerampAxis, DerampMethod, LogLevel,
                      OversamplingMethod, Workflow)
from . import _cpu
from ._cpu import PyCPUAmpcor, has_hdf5

try:
    from . import _cuda
    from ._cuda import PyCuAmpcor
    has_cuda = True
except ImportError:
    has_cuda = False


def set_log_level(level):
    """
    Set the level of the messages written by pycuampcor (both backends)

    level: LogLevel, its integer value or its name: 'quiet', 'warning' (to stderr),
        'info' (and progress, to stdout; default) or 'debug'
    """
    level = int(LogLevel.from_value(level))
    _cpu._set_log_level(level)
    if has_cuda:
        _cuda._set_log_level(level)


def get_log_level():
    """The level of the messages written by pycuampcor (LogLevel)"""
    return LogLevel(_cpu._get_log_level())
