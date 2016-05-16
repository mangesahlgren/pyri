"""
Module importing murmurhash for online generation of index vectors.
Need to be cythonized, build with:
pyri> python setup.pu build-ext -b /src
"""

from murmurhash.mrmr cimport hash64
from libc.stdint cimport uint64_t

cdef int seed = 0
"""
seed: Salt for the hash function. 
"""

cpdef uint64_t mhash(str word):
    """
    C and Python (Cython) function that computes a 64-bit hash
    from a python string (unicode).
    """
    bytestring = word.encode('UTF-8')
    cdef char* c_string = bytestring
    return(hash64(c_string, len(bytestring), seed))
