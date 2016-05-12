####################################
# Random Indexing
# re-implementation April 2016
####################################
import util
import numpy
import math
from collections import Counter
import itertools
from enum import Enum

def words(line):
    yield from iter(line.strip().split())

class Direction(Enum):
    left  = 0
    right = 1

class WordSpace(object):
    def __init__(self):
        self.total = 0             # Total number of tokens
        self.collocation = {}      # focus o=> (context counter) 
        self.wordcount = Counter() # focus counter

    def addCount(self, focus, context):
        if focus not in self.wordcount:
            self.collocation[focus] = Counter()
        self.collocation[focus].update(context)
        self.wordcount[focus] += 1
        self.total += 1

# online frequency weight defined in:
# Sahlgren et al. (2016) The Gavagai Living Lexicon, LREC
def weightFunc(freq,words,theta):
    return math.exp(-theta*(freq/words))

def dsm(infile, size, ws):
    with open(infile,'r') as handle:
        for line in handle:
            for (focus, (left_context, right_context)) in util.windows(words(line), size):
                context = itertools.chain(
                        ((Direction.left, x) for x in left_context),
                        ((Direction.right, x) for x in right_context))
                ws.addCount(focus, context) 
    return ws

"""
from collections import Counter
import operator
import scipy.spatial as st
import scipy.sparse as sp
from time import gmtime,strftime
np.seterr(divide='ignore', invalid='ignore')

####################################
# Global variables
####################################

dimen = 2000
nonzeros = 8
delta = 60
theta = 0.5
# worddict = {}
# rivecs = []
# distvecs = []

####################################
# Core functions
####################################

def dsm(infile,ctxwin):
    print("Started: " + strftime("%H:%M:%S",gmtime()))
    wordtokens = 0
    wordtypes = 0
    with open(infile,"r") as inp:
        for line in inp:
            wrdlst = line.strip().split()
            updatetokens,wordtypes = update_vecs(wrdlst,ctxwin,wordtokens,wordtypes)
            wordtokens += updatetokens
    print("Number of word tokens: " + str(wordtokens))
    print("Number of word types: " + str(wordtypes))
    print("Finished: " + strftime("%H:%M:%S",gmtime()))

def check_reps(wrd,wordtypes):
    global rivecs
    global distvecs
    global worddict
    if wrd in worddict:
        return 0
    else:
        rivecs.append(make_index())
        distvecs.append(np.zeros(dimen))
        worddict[wrd] = [wordtypes,0]
        return 1

def update_vecs(wrdlst,win,wordtokens,wordtypes):
    global distvecs
    global worddict
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        wordtypes += check_reps(w,wordtypes)
        wind = worddict[w][0]
        wvec = distvecs[wind]
        wri = rivecs[wind]
        worddict[w][1] += 1
        lind = 1
        while (lind < win+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            wordtypes += check_reps(c,wordtypes)
            cind = worddict[c][0]
            cvec = distvecs[cind]
            np.add.at(wvec,rivecs[cind][:,0]+1,rivecs[cind][:,1]*weight_func(worddict[c][1],wordtypes))
            np.add.at(cvec,rivecs[wind][:,0]-1,rivecs[wind][:,1]*weight_func(worddict[w][1],wordtypes))
            lind += 1
        ind += 1
    return localtoken, wordtypes

def update_vecs_ngrams(wrdlst,win,wordtokens,wordtypes):
    global distvecs
    global worddict
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        wordtypes += check_reps(w,wordtypes)
        wind = worddict[w][0]
        wvec = distvecs[wind]
        wri = rivecs[wind]
        worddict[w][1] += 1
        lind = 1
        while (lind < win+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            wordtypes += check_reps(c,wordtypes)
            cind = worddict[c][0]
            cvec = distvecs[cind]
            np.add.at(wvec,rivecs[cind][:,0]+1,rivecs[cind][:,1]*weight_func(worddict[c][1],wordtypes))
            np.add.at(cvec,rivecs[wind][:,0]-1,rivecs[wind][:,1]*weight_func(worddict[w][1],wordtypes))
            lind += 1
        ind += 1
    return localtoken, wordtypes

def make_index():
    ret = []
    inds = nprnd.randint(dimen-2, size=nonzeros) # dimen-2 to facilitate directional permutation
    for i in inds:
        sign = nprnd.randint(0,2)*2-1
        ret.append([i+1, sign]) # +1 to facilitate directional permutation
    return np.array(ret)

# TODO
# def make_very_sparse_index():

####################################
# Vector operations
####################################

# online frequency weight defined in:
# Sahlgren et al. (2016) The Gavagai Living Lexicon, LREC
def weight_func(freq,words):
    global delta
    return math.exp(-delta*(freq/words))

# remove dimensions with high variance
def prune_high_var(nrstd):
    global distvecs
    tmpmat = np.asmatrix(distvecs)
    varvec = np.var(tmpmat,0)
    varmean = np.mean(varvec)
    varstd = np.std(varvec)
    prunevec = np.zeros(len(distvecs[0]))
    ind = 0
    for i in np.nditer(varvec):
        if (i > (varmean + (varstd * nrstd))) or (i < (varmean - (varstd * nrstd))):
            prunevec[ind] = 0.0
        else:
            prunevec[ind] = 1.0
        ind += 1
    cnt = 0
    for d in distvecs:
        distvecs[cnt] = d*prunevec
        cnt += 1

# remove dimensions with low variance
def prune_low_var(nrstd):
    global distvecs
    tmpmat = np.asmatrix(distvecs)
    varvec = np.var(tmpmat,0)
    varmean = np.mean(varvec)
    varstd = np.std(varvec)
    prunevec = np.zeros(len(distvecs[0]))
    ind = 0
    for i in np.nditer(varvec):
        if (i < (varmean + (varstd * nrstd))) and (i > (varmean - (varstd * nrstd))):
            prunevec[ind] = 1.0
        else:
            prunevec[ind] = 0.0
        ind += 1
    cnt = 0
    for d in distvecs:
        distvecs[cnt] = d*prunevec
        cnt += 1

def isvd(upper,lower):
    global distvecs
    tmpmat = np.asmatrix(distvecs)
    u, s, v_t = sp.linalg.svds(tmpmat, k=upper, which='LM')
    dimred = u * s
    return dimred[:,lower:]

####################################
# Similarity
####################################

def sim(word1,word2):
    global worddict
    global distvecs
    return 1-st.distance.cosine(distvecs[worddict[word1][0]],distvecs[worddict[word2][0]])

def nns(word, num):
    global distvecs
    global worddict
    matrix = np.asmatrix(distvecs)
    v = distvecs[worddict[word][0]].reshape(1, -1)
    nns = st.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indices = [i for i in sorted(enumerate(nns), key=lambda x:x[1])]
    cnt = 1
    while cnt <= num:
        ele = [key for (key,value) in worddict.items() if value[0] == indices[cnt][0]]
        print(ele[0] + ' ' + str(1-indices[cnt][1]))
        cnt += 1

####################################
# Evaluation
####################################

def similarity_test(testfile,verb=True):
    global worddict
    global distvecs
    inp = open(testfile,"r")
    gold = {}
    model = {}
    cnt = 1
    for line in inp.readlines():
        word1, word2, sim = line.split()
        gold[cnt] = sim
        if (word1 in worddict) and (word2 in worddict):
            model[cnt] = 1-sp.distance.cosine(distvecs[worddict[word1][0]], distvecs[worddict[word2][0]])
        else:
            model[cnt] = 0
        cnt += 1
    inp.close()
    res = st.spearmanr(list(gold.values()),list(model.values()))
    if verb:
        print("Spearman rank correlation (coefficient, p-value): " + ", ".join([str(x) for x in res]))
    else:
        return res[0]

def vocabulary_test(testfile,verb=True):
    global worddict
    global distvecs
    inp = open(testfile,"r")
    corr = 0
    tot = 0
    unknown_target = []
    unknown_answer = []
    incorrect = []
    for line in inp.readlines():
        linelist = line.split()
        target = linelist[0]
        winner = ["",0]
        tot += 1
        if target in worddict:
            for a in linelist[1:]:
                if a in worddict:
                    sim = 1-sp.distance.cosine(distvecs[worddict[target][0]],distvecs[worddict[a][0]])
                    if sim > winner[1]:
                        winner = [a,sim]
                else:
                    if a is linelist[1]:
                        unknown_answer.append(a)
            if linelist[1] == winner[0]:
                corr += 1
            else:
                incorrect.append(target)
        else:
            unknown_target.append(target)
    inp.close()
    if verb:
        print("TOEFL synonym score: " + str(float(corr)/float(tot)))
        print("Incorrect: " + str(incorrect))
        print("Unknown targets: " + str(unknown_target))
        print("Unknown answers: " + str(unknown_answer))
    else:
        return float(corr)/float(tot)
=======
def weightFunc(freq,words,theta):
    return math.exp(-theta*(freq/words))
"""
