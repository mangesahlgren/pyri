from murmurhash.mrmr cimport hash64
from libc.stdint cimport uint64_t

cdef int seed = 0

cpdef uint64_t mhash(str word):
  bytestring = word.encode('UTF-8')
  cdef char* c_string = bytestring
  return(hash64(c_string, len(bytestring), seed))
