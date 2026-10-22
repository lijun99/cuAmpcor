"""
Check that the type stubs of the extension modules are up to date (see make_stubs.py): those of the
source tree, or those installed with the package where the source tree is not available (e.g., in the
tests of a conda package)
"""

import os

import pytest

import pycuampcor
from make_stubs import STUB_DIR, module_stub


# the stubs of the source tree, or of the installed package
STUBS = STUB_DIR if os.path.isdir(STUB_DIR) else os.path.dirname(pycuampcor.__file__)


def read_stub(name):
    """The content of the stub file name"""
    with open(os.path.join(STUBS, name)) as f:
        return f.read()


def test_cpu_stub():
    """The stub of the CPU module is up to date"""
    assert read_stub("_cpu.pyi") == module_stub(pycuampcor.PyCPUAmpcor), \
        "the stubs are outdated: run python tests/python/make_stubs.py"


@pytest.mark.skipif(not pycuampcor.has_cuda, reason="built without CUDA")
def test_cuda_stub():
    """The stub of the CUDA module is up to date"""
    assert read_stub("_cuda.pyi") == module_stub(pycuampcor.PyCuAmpcor), \
        "the stubs are outdated: run python tests/python/make_stubs.py"


def test_typed():
    """The package is marked as typed (py.typed)"""
    assert os.path.exists(os.path.join(STUBS, "py.typed"))
