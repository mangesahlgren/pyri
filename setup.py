import murmurhash
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            'hasher',
            sources=['src/hasher.pyx'],
            language='c++',
            extra_compile_args=['-O3','-g', '-I'+murmurhash.get_include()]
        )
    ]
)

