####################################
# Random Indexing
# re-implementation April-May 2016
####################################
import math
import numpy as np
import numpy.random as nprnd
from collections import Counter
import operator
import scipy.spatial as st
import scipy.sparse as sp
import scipy.stats as ss
from time import gmtime,strftime
np.seterr(divide='ignore', invalid='ignore')

####################################
# Global variables
####################################

dimen = 2000
nonzeros = 8
delta = 60
theta = 0.5
vocab = {}
rivecs = []
distvecs = []

####################################
# Getters
####################################

def get_freq(wrd):
    if wrd in vocab:
        return vocab[wrd][1]
    else:
        return False

def get_index(wrd):
    if wrd in vocab:
        return vocab[wrd][0]
    else:
        return False

def get_ri(wrd):
    if wrd in vocab:
        return rivecs[get_index(wrd)]
    else:
        return False

def get_vec(wrd):
    if wrd in vocab:
        return distvecs[get_index(wrd)]
    else:
        return False

####################################
# Core functions
####################################

def dsm(infile,ctxwin,trainfunc='direction',indexfunc='legacy'):
    """
    :param infile: one sentence per line
    :param ctxwin: the extension of the context window to the left and right
    :param trainfunc:
           'window' (bag-of-words) 
           'direction' (directional windows using permutations)
           'ngrams' (using directional windows and incremental ngram learning - currently a bit slow)
    :param indexfunc:
           'legacy' (standard RI)
           'verysparse' (very sparse RI using only two non-zero elements, one random and one controlled)
    """
    print("Started: " + strftime("%H:%M:%S",gmtime()))
    wordtokens = 0
    wordtypes = 0
    ngrams = 0
    with open(infile,"r") as inp:
        for line in inp:
            wrdlst = line.strip().split()
            if trainfunc == 'ngrams':
                updatetokens,wordtypes,ngrams = update_vecs_ngrams(wrdlst,ctxwin,wordtokens,wordtypes,ngrams,indexfunc)
            elif trainfunc == 'window':
                updatetokens,wordtypes = update_vecs(wrdlst,ctxwin,wordtokens,wordtypes,0,indexfunc)
            else:
                updatetokens,wordtypes = update_vecs(wrdlst,ctxwin,wordtokens,wordtypes,1,indexfunc)
            wordtokens += updatetokens
    print("Number of word tokens: " + str(wordtokens))
    print("Number of word types: " + str(wordtypes))
    if trainfunc == 'ngrams':
        print("Number of ngrams: " + str(ngrams))
    print("Finished: " + strftime("%H:%M:%S",gmtime()))

def check_reps(wrd,wordtypes,indexfunc):
    global rivecs
    global distvecs
    global vocab
    if wrd in vocab:
        return 0
    else:
        if indexfunc == 'verysparse':
            rivecs.append(make_very_sparse_index())
        else:
            rivecs.append(make_index())
        distvecs.append(np.zeros(dimen))
        vocab[wrd] = [wordtypes,0]
        return 1

def update_vecs(wrdlst,win,wordtokens,wordtypes,pi,indexfunc):
    global distvecs
    global vocab
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        wordtypes += check_reps(w,wordtypes,indexfunc)
        wind = get_index(w)
        wvec = get_vec(w)
        vocab[w][1] += 1
        lind = 1
        while (lind < win+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            wordtypes += check_reps(c,wordtypes,indexfunc)
            cind = get_index(c)
            cvec = get_vec(c)
### The expensive part is here - we should be able to optimize this (perhaps using Cython?)
            np.add.at(wvec,rivecs[cind][:,0]+pi,rivecs[cind][:,1]*weight_func(get_freq(c),wordtypes))
            np.add.at(cvec,rivecs[wind][:,0]-pi,rivecs[wind][:,1]*weight_func(get_freq(w),wordtypes))
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

verysparsecounter = 0
def make_very_sparse_index():
    global verysparsecounter
    ret = []
    ret.append([verysparsecounter,nprnd.randint(0,2)*2-1])
    ret.append([nprnd.randint(dimen-2, size=1)+1, nprnd.randint(0,2)*2-1]) # dimen-2 and +1 to facilitate directional permutation
    verysparsecounter += 1
    if verysparsecounter == (dimen-2):
        verysparsecounter = 0
    return np.array(ret)

####################################
# Online ngrams
####################################

def update_vecs_ngrams(wrdlst,win,wordtokens,wordtypes,ngrams,indexfunc):
    global distvecs
    global vocab
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    ngramlst = make_ngrams(wrdlst)
    for w in ngramlst:
        localtoken += 1
        wordtypes += check_reps(w,wordtypes,indexfunc)
        wind = get_index(w)
        wvec = get_vec(w)
        vocab[w][1] += 1
        lwin = win
        lind = 1
        while (lind < lwin+1) and ((ind+lind) < stop):
            c = ngramlst[ind+lind]
            bigram = check_ngram(w,c)
            if bigram:
                wordtypes += check_reps(bigram,wordtypes,indexfunc)
                w = bigram
                ngrams += 1
                vocab[w][1] += 1
                wind = get_index(w)
                wvec = get_vec(w)
                lwin = lwin + win
            else:
                wordtypes += check_reps(c,wordtypes,indexfunc)
                cind = get_index(c)
                cvec = get_vec(c)
                np.add.at(wvec,rivecs[cind][:,0]+1,rivecs[cind][:,1]*weight_func(get_freq(c),wordtypes))
                np.add.at(cvec,rivecs[wind][:,0]-1,rivecs[wind][:,1]*weight_func(get_freq(w),wordtypes))
            lind += 1
        ind += 1
    return localtoken, wordtypes, ngrams

def make_ngrams(sentencelist):
    global vocab
    ret = []
    ind = 0
    slen = len(sentencelist)
    for w in sentencelist:
        c = 1
        while c:
            if (ind + c) < slen:
                bigram = w + '_' + sentencelist[c]
                if bigram in vocab:
                    w = bigram
                    c += 1
                else:
                    ind = ind + c
                    ret.append(w)
                    c = False
            else:
                ret.append(w)
                ind = ind + c
                c = False
    return ret

def check_ngram(word1, word2):
    global vocab
    global distvecs
    global rivecs
    vec1 = get_vec(word1)
    ri1 = get_ri(word1)
    if word2 in vocab:
        ri2 = get_ri(word2)
        vec2 = get_vec(word2)
        tmp1 = np.zeros(dimen)
        tmp1 += vec1
        tmp2 = np.zeros(dimen)
        tmp2 += vec2
        np.multiply.at(tmp1,ri2[:,0],ri2[:,1]+1)
        np.multiply.at(tmp2,ri1[:,0],ri1[:,1]+1)
        freqprod = get_freq(word1) * (get_freq(word2) + 1)
        asum = (abs(np.sum(tmp1)) + abs(np.sum(tmp2))) / 2
        acnt = (asum / nonzeros) / freqprod
        if acnt > theta:
            bigram = word1 + '_' + word2
            return bigram
        else:
            return False
    else:
        return False

def get_ngrams():
    global vocab
    ret = []
    for key in vocab:
        if '_' in key:
            ret.append(key)
    return ret

####################################
# Vector operations
####################################

# online frequency weight defined in:
# Sahlgren et al. (2016) The Gavagai Living Lexicon, LREC
def weight_func(freq,words):
    global delta
    return math.exp(-delta*(freq/words))

# remove centroid
def remove_centroid():
    global distvecs
    tmpmat = np.asmatrix(distvecs)
    sumvec = np.sum(tmpmat, axis=0)
    sumnorm = sumvec / np.linalg.norm(sumvec)
    cnt = 0
    for d in distvecs:
        distvecs[cnt] = np.subtract(d,(np.multiply(d,sumnorm) / np.linalg.norm(sumnorm)))
        cnt += 1

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

def tsvd(k):
    global distvecs
    tmpmat = np.asmatrix(distvecs)
    u, s, v_t = sp.linalg.svds(tmpmat, k=k, which='LM')
    dimred = u * s
    return dimred

####################################
# Similarity
####################################

def sim(word1,word2,model=distvecs):
    global vocab
    return 1 - st.distance.cosine(model[get_index(word1)],model[get_index(word2)])

def nns(word,num,model=distvecs):
    global vocab
    if model == distvecs:
        matrix = np.asmatrix(distvecs)
        v = get_vec(word).reshape(1, -1)
    else:
        matrix = model
        v = model[get_index(word)].reshape(1, -1)
    nns = st.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indices = [i for i in sorted(enumerate(nns), key=lambda x:x[1])]
    cnt = 1
    while cnt <= num:
        ele = [key for (key,value) in vocab.items() if value[0] == indices[cnt][0]]
        print(ele[0] + ' ' + str(1 - indices[cnt][1]))
        cnt += 1

####################################
# Evaluation
####################################

def run_test_suite(model=distvecs):
    print("TOEFL - ", end='')
    vocabulary_test("/bigdata/evaluation/synonyms/en/toefl.txt",model=distvecs)
    print("ESL - ", end='')
    vocabulary_test("/bigdata/evaluation/synonyms/en/esl.txt",model=distvecs)
    print("SimLex-999 - ", end='')
    similarity_test("/bigdata/evaluation/similarity/SimLex/simlex.txt",model=distvecs)
    print("MEN - ", end='')
    similarity_test("/bigdata/evaluation/similarity/MEN/men.tsts",model=distvecs)
    print("RW - ", end='')
    similarity_test("/bigdata/evaluation/similarity/rarewords/rarewords.txt",model=distvecs)

def run_test_suite_low(ver,model=distvecs):
    print("TOEFL - ", end='')
    vocabulary_test(str("/home/mange/toefl." + ver + ".low"),model=distvecs)
    print("ESL - ", end='')
    vocabulary_test(str("/home/mange/esl." + ver + ".low"),model=distvecs)
    print("SimLex-999 - ", end='')
    similarity_test(str("/home/mange/simlex." + ver + ".low"),model=distvecs)
    print("MEN - ", end='')
    similarity_test(str("/home/mange/men." + ver + ".low"),model=distvecs)
    print("RW - ", end='')
    similarity_test("/bigdata/evaluation/similarity/rarewords/rarewords.txt",model=distvecs)

def similarity_test(testfile,model=distvecs,verb=True):
    global vocab
    inp = open(testfile,"r")
    gold = {}
    test = {}
    cnt = 1
    for line in inp.readlines():
        word1, word2, sim = line.split()
        gold[cnt] = sim
        if (word1 in vocab) and (word2 in vocab):
            test[cnt] = 1 - st.distance.cosine(model[get_index(word1)], model[get_index(word2)])
        else:
            test[cnt] = 0
        cnt += 1
    inp.close()
    res = ss.spearmanr(list(gold.values()),list(test.values()))
    if verb:
        print("Spearman rank correlation (coefficient, p-value): " + ", ".join([str(x) for x in res]))
    else:
        return res[0]

def vocabulary_test(testfile,model=distvecs,verb=True):
    global vocab
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
        if target in vocab:
            for a in linelist[1:]:
                if a in vocab:
                    sim = 1 - st.distance.cosine(model[get_index(target)],model[get_index(a)])
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
        print("Accuracy: " + str(float(corr)/float(tot)))
        # print("Incorrect: " + str(incorrect))
        # print("Unknown targets: " + str(unknown_target))
        # print("Unknown answers: " + str(unknown_answer))
    else:
        return float(corr)/float(tot)
