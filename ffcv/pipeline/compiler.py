import pdb
import warnings
from multiprocessing import cpu_count
from os import sched_getaffinity

import torch as ch
from numba import get_num_threads, njit, prange, set_num_threads
from numba import warnings as nwarnings
from numba.core.errors import NumbaPerformanceWarning


class Compiler:

    @classmethod
    def set_enabled(cls, b):
        cls.is_enabled = b

    @classmethod
    def set_num_threads(cls, n):
        if n < 1:
            n = len(sched_getaffinity(0))
        cls.num_threads = n
        set_num_threads(n)
        ch.set_num_threads(n)

    @classmethod
    def compile(cls, code, signature=None):
        parallel = False
        if hasattr(code, "is_parallel"):
            parallel = code.is_parallel and cls.num_threads > 1

        if cls.is_enabled:
            return njit(
                signature,
                fastmath=True,
                nogil=True,
                error_model="numpy",
                parallel=parallel,
            )(code)
        return code

    @classmethod
    def get_iterator(cls):
        if cls.num_threads > 1:
            return prange
        else:
            return range


Compiler.set_enabled(True)
Compiler.set_num_threads(1)
