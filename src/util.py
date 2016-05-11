"""
Module for utility functions.
"""

import itertools
from collections import deque
import numpy
import functools
import hasher
import icu

def sentencesplitter(locale):
    sentence_iterator = icu.BreakIterator.createSentenceInstance(locale)
    def split(txt):
        sentence_iterator.setText(txt)
        left_end = sentence_iterator.current()
        while sentence_iterator.nextBoundary() != icu.BreakIterator.DONE:
            yield txt[left_end:sentence_iterator.current()]
            left_end = sentence_iterator.current()
    return(split)

def wordsplitter(locale):
    word_iterator = icu.BreakIterator.createWordInstance(locale)
    def split(txt):
        word_iterator.setText(txt)
        left_end = word_iterator.current()
        while word_iterator.nextBoundary() != icu.BreakIterator.DONE:
            if (word_iterator.getRuleStatus() >=100):
                yield txt[left_end:word_iterator.current()]
            left_end = word_iterator.current()
    return(split)

def tokenize(locale):
    sentences = sentence_splitter(locale)
    words = word_splitter(locale)
    def split(txt):
        for sentence in sentences(txt):
            yield words(sentence)

def linegenerator(infile):
    """
    open a file and make a generator that returns the lines of the file.
    """
    with open(infile, 'r') as inp:
      for line in inp: 
        yield(line)

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

def split64(num):
    """
    Split a 64 bit word into two 32 bit words. For seeind the random state.
    """
    mask = 4294967295 # mask = 111..111 = 32 ones
    return (num & mask, (num >> 32) & mask)


def vector_generator(dim, nonzeros, cache_size):
    """
    Initialize a generator of index vectors with an lru cache.
    """
    @functools.lru_cache(cache_size)
    def index_vector(word):
        """
        Get the index vector of a word.
        """
        generator = numpy.random.RandomState(split64(hasher.mhash(word)))
        return(numpy.vstack((
            generator.randint(dim-1, size=nonzeros),
            generator.randint(0, high=2, size=nonzeros)*2-1)))
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

