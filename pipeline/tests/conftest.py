"""Shared test config.

PyTensor wants a C compiler; on boxes without g++ (typical Windows dev setup)
it prints a warning and falls back to the Python backend anyway. Telling it
explicitly there is no compiler keeps test output pristine. Must run before
anything imports pytensor, hence conftest.
"""

import os
import shutil

if shutil.which("g++") is None:
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
