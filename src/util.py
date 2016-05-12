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

def windows(generator, size):
    """
    Create a generator of type [(left_context, a, right_context)]
    from a generator of type [a], where left_context has size
    left_size and right_context has size right_size.
    """
    g = iter(generator)
    left_context = deque((), size)
    right_context = deque(itertools.islice(g, 0, size), size)
    for elem in g:
        focus = right_context.popleft()
        right_context.append(elem)
        yield(focus, (left_context, right_context))
        left_context.append(focus)
    while right_context:
        focus = right_context.popleft()
        yield(focus, (left_context, right_context))
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
