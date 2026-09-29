"""
pycuampcor: amplitude cross-correlation (ampcor) with CPU and CUDA backends

PyCPUAmpcor : the CPU (OpenMP) implementation, always available
PyCuAmpcor  : the CUDA implementation, available if built with CUDA support
"""

try:
    from ._cuda import PyCuAmpcor
    has_cuda = True
except ImportError:
    has_cuda = False
