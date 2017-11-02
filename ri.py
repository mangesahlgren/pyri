####################################
# Random Indexing
# re-implementation April-May 2016
# cleaning up October 2017
####################################
import math
import numpy as np
import numpy.random as nprnd
from collections import Counter
import operator
import random
import scipy.spatial as st
import scipy.sparse as sp
import scipy.stats as ss
from time import gmtime,strftime

np.seterr(divide='ignore', invalid='ignore')

####################################
# Getters
####################################

def get_freq(wrd, vocab):
    if wrd in vocab:
        return vocab[wrd][1]
    else:
        return False

def get_index(wrd, vocab):
    if wrd in vocab:
        return vocab[wrd][0]
    else:
        return False

def get_ri(wrd, vocab, rivecs):
    if wrd in vocab:
        return rivecs[get_index(wrd, vocab)]
    else:
        return False

def get_vec(wrd, vocab, distvecs):
    if wrd in vocab:
        return distvecs[get_index(wrd, vocab)]
    else:
        return False

####################################
# Core functions
####################################

def dsm(infile, win=2, trainfunc='direction', indexfunc='legacy', dimen=2000, nonzeros=8, delta=60, theta=0.5, use_rivecs=False, use_weights=True):
    """
    Python implementation of Random Indexing

    Input: one sentence per line, preferably tokenized text
    Output: array of distributional vectors, array of random vectors, dictionary of the vocabulary (word -> array_index, frequency)
    
    Parameters are:
    Size of the context window to the left and right (default: 2)
    Function for accumulating distributional vectors. Alternatives are:
    - 'window' (bag-of-words) 
    - 'direction' (directional windows using permutations)
    - 'ngrams' (using directional windows and incremental ngram learning - unpublished and currently a bit slow)
    Function for producing random index vectors. Alternatives are:
    - 'legacy' (standard RI)
    - 'verysparse' (very sparse RI using only two non-zero elements, one random and one controlled)
    Dimensionality of the vectors (default: 2000)
    Non-zero elements in the random index vectors (default: 8)
    Delta constant for the incremental frequency weight (deafult: 60)
    Theta threshold for the online n-gram learning (default: 0.5)
    
    There are two additional flags:
    use_rivecs: use precompiled ri vectors (produced with the function make_ri_vecs())
    use_weights: use incremental frequency weights (default: True)
    """
    print("Started: " + strftime("%H:%M:%S", gmtime()))
    tokens = 0
    types = 0
    ngrams = 0
    vocab = {}
    distvecs = []
    rivecs = []
    rivecs_full = []
    with open(infile, "r") as inp:
        for line in inp:
            wrdlst = line.strip().split()
            if trainfunc == 'ngrams':
                newtokens, types, ngrams, distvecs, rivecs, rivecs_full, vocab = update_vecs_ngrams(wrdlst, win, dimen, nonzeros, delta, theta, tokens, types, ngrams, vocab, rivecs, rivecs_full, distvecs, indexfunc, use_rivecs)
            elif trainfunc == 'window':
                newtokens, types, distvecs, rivecs, vocab = update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, 0, indexfunc, use_rivecs, use_weights)
            else:
                newtokens, types, distvecs, rivecs, vocab = update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, 1, indexfunc, use_rivecs, use_weights)
            tokens += newtokens
    print("Number of word tokens: " + str(tokens))
    print("Number of word types: " + str(types))
    if trainfunc == 'ngrams':
        print("Number of ngrams: " + str(ngrams))
    print("Finished: " + strftime("%H:%M:%S", gmtime()))
    return distvecs, rivecs, vocab

def check_reps(wrd, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab):
    if wrd in vocab:
        return types, rivecs, distvecs, vocab
    else:
        if not use_rivecs:
            if indexfunc == 'verysparse':
                rivecs.append(make_very_sparse_index(dimen))
            else:
                rivecs.append(make_index(dimen, nonzeros))
        distvecs.append(np.zeros(dimen))
        vocab[wrd] = [types, 0]
        return types+1, rivecs, distvecs, vocab

def update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, pi, indexfunc, use_rivecs, use_weights):
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        types, rivecs, distvecs, vocab = check_reps(w, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
        wind = get_index(w, vocab)
        wvec = get_vec(w, vocab, distvecs)
        vocab[w][1] += 1
        lind = 1
        while (lind < win+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            types, rivecs, distvecs, vocab = check_reps(c, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
            cind = get_index(c, vocab)
            cvec = get_vec(c, vocab, distvecs)
            if use_weights:
                np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1]*weight_func(get_freq(c, vocab), types, delta))
                np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1]*weight_func(get_freq(w, vocab), types, delta))
            else:
                np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1])
                np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1])
            lind += 1
        ind += 1
    else:
        return localtoken, types, distvecs, rivecs, vocab

def make_index(dimen, nonzeros):
    ret = []
    inds = nprnd.randint(dimen-2, size=nonzeros) # dimen-2 to facilitate directional permutation
    signs = []
    for i in inds:
        sign = nprnd.randint(0,2)*2-1
        signs.append(sign)
        ret.append([i+1, sign]) # +1 to facilitate directional permutation
    return np.array(ret)

verysparsecounter = 0
def make_very_sparse_index(dimen):
    global verysparsecounter
    ret = []
    ret.append([verysparsecounter,nprnd.randint(0,2)*2-1])
    ret.append([nprnd.randint(dimen-2, size=1)+1, nprnd.randint(0,2)*2-1]) # dimen-2 and +1 to facilitate directional permutation
    verysparsecounter += 1
    if verysparsecounter == (dimen-2):
        verysparsecounter = 0
    return np.array(ret)

def make_ri_vecs(nr, dimen, nonzeros):
    rivecs = []
    cnt = 0
    while cnt < nr:
        rive = make_index(corr, dimen, nonzeros)
        rivecs.append(rive)
        cnt += 1
    return rivecs

####################################
# Online ngrams
####################################

def update_vecs_ngrams(wrdlst, win, dimen, nonzeros, delta, theta, tokens, types, ngrams, vocab, rivecs, rivecs_full, distvecs, indexfunc, use_rivecs):
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    ngramlst = make_skip_ngrams(wrdlst, vocab, win)
    for w in ngramlst:
        localtoken += 1
        types, rivecs, distvecs, vocab, rivecs_full = check_reps_ngrams(w, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
        wind = get_index(w, vocab)
        wvec = get_vec(w, vocab, distvecs)
        vocab[w][1] += 1
        lwin = win
        lind = 1
        while (lind < lwin+1) and ((ind+lind) < stop):
            c = ngramlst[ind+lind]
            bigram = check_ngram(w, c, vocab, rivecs_full, distvecs, theta)
            if bigram:
                types, rivecs, distvecs, vocab, rivecs_full = check_reps_ngrams(bigram, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
                w = bigram
                ngrams += 1
                vocab[w][1] += 1
                wind = get_index(w, vocab)
                wvec = get_vec(w, vocab, distvecs)
                lwin = lwin + win
            else:
                types, rivecs, distvecs, vocab, rivecs_full = check_reps_ngrams(c, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
                cind = get_index(c, vocab)
                cvec = get_vec(c, vocab, distvecs)
                np.add.at(wvec, rivecs[cind][:,0]+1, rivecs[cind][:,1]*weight_func(get_freq(c, vocab), types, delta))
                np.add.at(cvec, rivecs[wind][:,0]-1, rivecs[wind][:,1]*weight_func(get_freq(w, vocab), types, delta))
            lind += 1
        ind += 1
    return localtoken, types, ngrams, distvecs, rivecs, rivecs_full, vocab

def make_skip_ngrams(sentencelist, vocab, win):
    ret = []
    ind = 0
    slen = len(sentencelist)
    for w in sentencelist:
        c = 1
        add = 1
        while c <= win:
            if (ind + c) < slen:
                bigram = w + '_' + sentencelist[c]
                if bigram in vocab:
                    w = bigram
                    add = c
                c += 1
            else:
                ret.append(w)
                ind = ind + c
                c = False
        ind = ind + add
        ret.append(w)
    return ret

def check_ngram(word1, word2, vocab, rivecs_full, distvecs, theta):
    vec1 = get_vec(word1, vocab, distvecs)
    ri1 = np.roll(get_ri(word1, vocab, rivecs_full), +1)
    if word2 in vocab:
        ri2 = np.roll(get_ri(word2, vocab, rivecs_full), -1)
        vec2 = get_vec(word2, vocab, distvecs)
        acnt1 = st.distance.cosine(vec1, ri2)
        acnt2 = st.distance.cosine(vec2, ri1)
        if (1 - acnt1) > theta and (1 - acnt2) > theta:
            bigram = word1 + '_' + word2
            return bigram
        else:
            return False
    else:
        return False

def check_reps_ngrams(wrd, types, indexfunc, dimen, nonzeros, use_rivecs, rivecs, rivecs_full, distvecs, vocab):
    if wrd in vocab:
        return types, rivecs, distvecs, vocab, rivecs_full
    else:
        if not use_rivecs:
            ri_compact, ri_full = make_index(dimen, nonzeros)
            rivecs_full.append(ri_full)
        distvecs.append(np.zeros(dimen))
        vocab[wrd] = [types, 0]
        return types+1, rivecs, distvecs, vocab, rivecs_full

def get_ngrams(vocab):
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
def weight_func(freq, words, delta):
    return math.exp(-delta*(freq/words))

# remove centroid
# Sahlgren et al. (2016) The Gavagai Living Lexicon, LREC
def remove_centroid(model):
    tmpmat = np.asmatrix(model)
    sumvec = np.sum(tmpmat, axis=0)
    sumnorm = sumvec / np.linalg.norm(sumvec)
    cnt = 0
    for d in model:
        model[cnt] = np.subtract(d, (np.multiply(d, sumnorm) / np.linalg.norm(sumnorm)))
        cnt += 1

def svd(model, upperdim=1000):
    tmpmat = np.asmatrix(model)
    u, s, v_t = sp.linalg.svds(tmpmat, k=upperdim, which='LM')
    return u, s

def make_ri_matrix(rivecs, dim):
    mat = np.zeros((len(rivecs), dim))
    ind = 0
    for i in rivecs:
        np.add.at(mat[ind], rivecs[ind][:,0], rivecs[ind][:,1])
        ind += 1
    return mat

####################################
# Similarity
####################################

def sim(word1, word2, model, vocab):
    '''
    Compute the cosine similarity between the distributional vectors of word1 and word2
    using the vectors in model and the dictionary in vocab
    '''
    return 1 - st.distance.cosine(model[vocab[word1][0]], model[vocab[word2][0]])

def synt_sim(word1, word2, rot, synt_matrix, model, vocab):
    '''
    Compute the cosine similarity between the distributional vector of word1 and the random index vectors of word2 
    using rot as the permutation, the random index vectors in synt_matrix, the distributional vectors in model and the dictionary in vocab
    '''
    return 1 - st.distance.cosine(model[vocab[word1][0]], np.roll(synt_matrix[vocab[word2][0]], +rot))

def nns(word, num, model, vocab):
    matrix = np.asmatrix(model)
    v = model[vocab[word][0]].reshape(1, -1)
    nns = st.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indices = [i for i in sorted(enumerate(nns), key=lambda x:x[1])]
    cnt = 1
    while cnt <= num:
        ele = [key for (key,value) in vocab.items() if value[0] == indices[cnt][0]]
        print(ele[0] + ' ' + str(1 - indices[cnt][1]))
        cnt += 1

def nns_return(word, num, model, vocab, sims=True):
    '''
    Return the num nearest neighbors to word in model. 
    vocab holds the vocabulary dictionary.
    '''
    matrix = np.asmatrix(model)
    v = model[vocab[word][0]].reshape(1, -1)
    nns = st.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indices = [i for i in sorted(enumerate(nns), key=lambda x:x[1])]
    cnt = 1
    ret = []
    while cnt <= num:
        ele = [key for (key,value) in vocab.items() if value[0] == indices[cnt][0]]
        if sims:
            ret.append([ele[0], (1 - indices[cnt][1])])
        else:
            ret.append(ele[0])
        cnt += 1
    return ret

def synt_nns(word, num, rot, synt_matrix, model, vocab):
    '''
    Return the num nearest syntagmatic neighbors to word.
    synt_matrix holds the random index vectors, model holds the distributional vectors, and vocab holds the vocabulary dictionary.
    '''
    v = np.roll(model[get_index(word, vocab)].reshape(1, -1),-rot)
    nns = st.distance.cdist(synt_matrix, v, 'cosine').reshape(-1)
    indices = [i for i in sorted(enumerate(nns), key=lambda x:x[1])]
    cnt = 1
    while cnt <= num:
        ele = [key for (key,value) in vocab.items() if value[0] == indices[cnt][0]]
        print(ele[0] + ' ' + str(1 - indices[cnt][1]))
        cnt += 1

####################################
# Evaluation
####################################

def similarity_test(testfile, model, vocab, verb=True):
    inp = open(testfile,"r")
    gold = {}
    test = {}
    cnt = 1
    for line in inp.readlines():
        word1, word2, sim = line.split()
        gold[cnt] = sim
        if (word1 in vocab) and (word2 in vocab):
            test[cnt] = 1 - st.distance.cosine(model[get_index(word1, vocab)], model[get_index(word2, vocab)])
        else:
            test[cnt] = 0
        cnt += 1
    inp.close()
    res = ss.spearmanr(list(gold.values()),list(test.values()))
    if verb:
        print("Spearman rank correlation (coefficient, p-value): " + ", ".join([str(x) for x in res]))
    else:
        return res[0]

def vocabulary_test(testfile, model, vocab, verb=True):
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
                    sim = 1 - st.distance.cosine(model[get_index(target, vocab)], model[get_index(a, vocab)])
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
