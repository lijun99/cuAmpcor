#!/usr/bin/env python3
"""
Generate the reference outputs of the regression tests (test_regression.py)

Run this script when a change of the results is intended, and commit the
updated files in tests/data/ampcor/ovs128-rho0.8/golden. The CPU implementation
(deterministic, independent of the number of threads) is used.

usage: python tests/python/make_golden.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_regression import GOLDEN_OUTPUTS, golden_dir, run_dataset  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for workflow in (0, 1):
            for ovs_method in (0, 1):
                out = run_dataset("cpu", os.path.join(tmp, "out"), workflow, ovs_method)
                path = golden_dir(workflow, ovs_method)
                shutil.rmtree(path, ignore_errors=True)
                os.makedirs(path)
                for name in GOLDEN_OUTPUTS:
                    out[name].astype(np.float32).tofile(os.path.join(path, name))
                print("written", path)


if __name__ == "__main__":
    sys.exit(main())
