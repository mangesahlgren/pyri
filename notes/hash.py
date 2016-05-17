"""
This is a small code diary, beginning with the exploration of python FFI.
"""
from ctypes import *

"""
Messing around with c libraries: cityhash python library does not play well with
python 3. Therefore i tried to install it myself and learn how to call c
libraries from python. 

following instructions from the readme it installed into usr/local/lib.
To use functions from it:
"""

cityhash = cdll('/usr/local/lib/libcityhash.so')

"""
This doesn't work:
cityhash.CityHash128('hej',3)

i used
$> nm -D /usr/local/lib/libcityhash.so
to find the function names. The appropriate one turned out to be:
_Z11CityHash128PKcm
"""

cityhash._Z11CityHash128PKcm('hej',3)

"""
the above returns 127519005, far to small to be correct
apparently, the FFI defaults to returning ints. However CityHash returns
std::pair<uint64,uint64>, small testint shows that this is NOT the same as
two memory consecutive uint64s, need to look into cython. 

Went with murmurhash. The python cityhash library
did not play well with python 3.
After messing around in cython i got it to work, just run 
> python setup.py build-ext -b pyri/
in the pyri directory after initiating the conda environment, and switching to it. 
"""


