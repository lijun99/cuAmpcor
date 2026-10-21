"""
Check that the type stubs of the extension modules are up to date (see make_stubs.py)
"""

import os

import pytest

import pycuampcor
from make_stubs import STUB_DIR, module_stub


def read_stub(name):
    with open(os.path.join(STUB_DIR, name)) as f:
        return f.read()


def test_cpu_stub():
    assert read_stub("_cpu.pyi") == module_stub(pycuampcor.PyCPUAmpcor), \
        "the stubs are outdated: run python tests/python/make_stubs.py"


@pytest.mark.skipif(not pycuampcor.has_cuda, reason="built without CUDA")
def test_cuda_stub():
    assert read_stub("_cuda.pyi") == module_stub(pycuampcor.PyCuAmpcor), \
        "the stubs are outdated: run python tests/python/make_stubs.py"


def test_typed():
    assert os.path.exists(os.path.join(STUB_DIR, "py.typed"))
