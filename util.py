"""
Module for utility functions.
"""

import itertools
from collections import deque 

def windows(generator, left_size, right_size):
    """
    Create a generator of type [(left_context, a, right_context)]
    from a generator of type [a], where left_context has size
    left_size and right_context has size right_size
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

def thread(generator, side_effect):
    """
    Thread some side effect silently through a generator.
    I.e. the generator should not look any different.
    (Unless the side-effect is nasty)
    """
    for elem in generator:
        side_effect(elem)
        yield elem
