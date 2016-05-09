"""
Module for utility functions.
"""

import itertools
from collections import deque
from cityhash import CityHash128
import numpy
import functools

def windows(generator, left_size, right_size):
    """
    Create a generator of type [(left_context, a, right_context)]
    from a generator of type [a], where left_context has size
    left_size and right_context has size right_size.
    The contexts are padded with Nones.
    """
    right_padding = itertools.repeat(None, right_size)
    left_padding = itertools.repeat(None, left_size)

    gen = itertools.chain(generator, right_padding)
    left_context = deque(left_padding, left_size)
    right_context = deque(itertools.islice(gen, 0, right_size), right_size+1)

    for elem in gen:
        right_context.append(elem)
        focus = right_context.popleft()
        yield(left_context, focus, right_context)
        left_context.append(focus)

def split128(num):
    """
    Split a 128 bit word into four 32 bit words. 
    """
    mask = 4294967295
    """
    mask = 11...11 (binary) 
           ^^^^^^^: 32 ones  
    """
    return (num & mask,
            (num >> 32) & mask,
            (num >> 64) & mask,
            (num >> 96) & mask)

def vector_generator(dim, nonzeros, cache_size):
    """
    Initialize a generator of index vectors.
    DONT USE THE NUMPY RANDOM GENERATOR 
    WITHOUT SEEDING AFTER THIS 
    """
    def index_vector(word):
        """
        Get the index vector of a word.
        """
        numpy.random.seed(split128(CityHash128(word)))
        return(numpy.vstack((
            numpy.random.randint(dim-1, size=nonzeros),
            numpy.random.randint(0,high=2,size=nonzeros)*2-1)))
    return(index_vector)

def thread(generator, side_effect):
    """
    Thread some side effect silently through a generator.
    I.e. the generator should not look any different.
    (Unless the side-effect is nasty)
    """
    for elem in generator:
        side_effect(elem)
        yield elem

