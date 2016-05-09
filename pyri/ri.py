####################################
# Random Indexing
# re-implementation April 2016
#
# TODO: makeVerySparseIndex
####################################
import math
import numpy as np
import numpy.random as nprnd
from collections import Counter
from scipy import sparse
from time import gmtime,strftime

dimen = 2000
nonzeros = 8
theta = 60
worddict = {}
rivecs = []
distvecs = []

def trainSparseRI(infile,ctxwin):
    print("Started: " + strftime("%H:%M:%S",gmtime()))
    wordtokens = 0
    wordtypes = 0
    with open(infile,"r") as inp:
        for line in inp:
            wrdlst = line.strip().split()
            updatetokens,wordtypes = updateVecs(wrdlst,ctxwin,wordtokens,wordtypes)
            wordtokens += updatetokens
    print("Number of word tokens: " + str(wordtokens))
    print("Number of word types: " + str(wordtypes))
    print("Finished: " + strftime("%H:%M:%S",gmtime()))

def checkReps(wrd,wordtypes):
    global rivecs
    global distvecs
    global worddict
    if wrd in worddict:
        return 0
    else:
        rivecs.append(makeIndex())
        distvecs.append(np.zeros(dimen))
        worddict[wrd] = [wordtypes,0]
        return 1

def updateVecs(wrdlst,win,wordtokens,wordtypes):
    global distvecs
    global worddict
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        wordtypes += checkReps(w,wordtypes)
        wind = worddict[w][0]
        wvec = distvecs[wind]
        wri = rivecs[wind]
        worddict[w][1] += 1
        lind = 1
        while (lind < win+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            wordtypes += checkReps(c,wordtypes)
            cind = worddict[c][0]
            cvec = distvecs[cind]
            np.add.at(wvec,rivecs[cind][:,0]+1,rivecs[cind][:,1]*weightFunc(worddict[c][1],wordtypes,theta))
            np.add.at(cvec,rivecs[wind][:,0]-1,rivecs[wind][:,1]*weightFunc(worddict[w][1],wordtypes,theta))
            lind += 1
        ind += 1
    return localtoken, wordtypes

def makeIndex():
    ret = []
    inds = nprnd.randint(dimen-2, size=nonzeros) # dimen-2 to facilitate directional permutation
    for i in inds:
        sign = nprnd.randint(0,2)*2-1
        ret.append([i+1, sign]) # +1 to facilitate directional permutation
    return np.array(ret)

# online frequency weight defined in:
# Sahlgren et al. (2016) The Gavagai Living Lexicon, LREC
def weightFunc(freq,words,theta):
    return math.exp(-theta*(freq/words))
