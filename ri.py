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
    :param infile: one sentence per line
    :param win: the extension of the context window to the left and right
    :param trainfunc:
           'window' (bag-of-words) 
           'direction' (directional windows using permutations)
           'ngrams' (using directional windows and incremental ngram learning - currently a bit slow)
           'random' (using randomly sized windows bounded by win)
           'corr' (using RI correction)
           'coll' (using collocation dampening)
    :param indexfunc:
           'legacy' (standard RI)
           'verysparse' (very sparse RI using only two non-zero elements, one random and one controlled)
    :param use_rivecs: use precompiled ri vectors
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
                newtokens, types, ngrams, distvecs, rivecs, vocab = update_vecs_ngrams(wrdlst, win, dimen, nonzeros, delta, theta, tokens, types, ngrams, vocab, rivecs, distvecs, indexfunc, use_rivecs, corr=False)
            elif trainfunc == 'window':
                newtokens, types, distvecs, rivecs, vocab = update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, 0, indexfunc, use_rivecs, use_weights, corr=False, coll=False)
            elif trainfunc == 'random':
                newtokens, types, distvecs, rivecs, vocab = update_vecs_random(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, distvecs, 1, indexfunc)
            elif trainfunc == 'corr':
                newtokens, types, distvecs, rivecs, vocab, rivecs_full = update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, 1, indexfunc, use_rivecs, use_weights, corr=True, coll=False)
            elif trainfunc == 'coll':
                newtokens, types, distvecs, rivecs, vocab, rivecs_full = update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, 1, indexfunc, use_rivecs, use_weights, corr=True, coll=True)
            else:
                newtokens, types, distvecs, rivecs, vocab = update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, 1, indexfunc, use_rivecs, use_weights, corr=False, coll=False)
            tokens += newtokens
    print("Number of word tokens: " + str(tokens))
    print("Number of word types: " + str(types))
    if trainfunc == 'ngrams':
        print("Number of ngrams: " + str(ngrams))
    print("Finished: " + strftime("%H:%M:%S", gmtime()))
    return distvecs, rivecs, vocab

def check_reps(wrd, types, indexfunc, dimen, nonzeros, corr, use_rivecs, rivecs, rivecs_full, distvecs, vocab):
    if wrd in vocab:
        if corr:
            return types, rivecs, distvecs, vocab, rivecs_full
        else:
            return types, rivecs, distvecs, vocab, False
    else:
        if not use_rivecs:
            if indexfunc == 'verysparse':
                rivecs.append(make_very_sparse_index(dimen))
            else:
                if corr:
                    ri_compact, ri_full = make_index(corr, dimen, nonzeros)
                    rivecs.append(ri_compact)
                    rivecs_full.append(ri_full)
                else:
                    rivecs.append(make_index(corr, dimen, nonzeros))
        distvecs.append(np.zeros(dimen))
        vocab[wrd] = [types, 0]
        if corr:
            return types+1, rivecs, distvecs, vocab, rivecs_full
        else:
            return types+1, rivecs, distvecs, vocab, False

def update_vecs(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, rivecs_full, distvecs, pi, indexfunc, use_rivecs, use_weights, corr, coll):
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        types, rivecs, distvecs, vocab, rivecs_full = check_reps(w, types, indexfunc, dimen, nonzeros, corr, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
        wind = get_index(w, vocab)
        wvec = get_vec(w, vocab, distvecs)
        vocab[w][1] += 1
        lind = 1
        while (lind < win+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            types, rivecs, distvecs, vocab, rivecs_full = check_reps(c, types, indexfunc, dimen, nonzeros, corr, use_rivecs, rivecs, rivecs_full, distvecs, vocab)
            cind = get_index(c, vocab)
            cvec = get_vec(c, vocab, distvecs)
            if corr and not coll:
                if vocab[c][1] > 1: # stupid check that should not be necessary
                    # only correct ri for when ctx vec is not a stopword
                    if weight_func(get_freq(c, vocab), types) > 0.25:
                        check_ri(rivecs[wind], rivecs_full[wind], cvec, -pi, w)
                    if weight_func(get_freq(w, vocab), types) > 0.25:
                        check_ri(rivecs[cind], rivecs_full[cind], wvec, +pi, c)
                        ### The expensive part is here - we should be able to optimize this (perhaps using Cython?)
            # vec_adder(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1]*weight_func(get_freq(c),types))
            # vec_adder(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1]*weight_func(get_freq(w),types))
            if coll:
                corr1 = st.distance.cosine(wvec, rivecs_full[cind])
                corr2 = st.distance.cosine(cvec, rivecs_full[wind])
                if math.isnan(corr1) or math.isinf(corr1):
                    np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1])
                else:
                    np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1]*corr1)
                if math.isnan(corr2) or math.isinf(corr2):
                    np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1])
                else:
                    np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1]*corr2)
            elif use_weights:
                np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1]*weight_func(get_freq(c, vocab), types, delta))
                np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1]*weight_func(get_freq(w, vocab), types, delta))
            else:
                np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1])
                np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1])
            # wvec += rivecs[cind] ## ADD rotation + weights!!!
            # cvec += rivecs[wind]
            lind += 1
        ind += 1
    if corr or coll:
        return localtoken, types, distvecs, rivecs, vocab, rivecs_full
    else:
        return localtoken, types, distvecs, rivecs, vocab

def update_vecs_random(wrdlst, win, dimen, nonzeros, delta, tokens, types, vocab, rivecs, distvecs, pi, indexfunc):
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    for w in wrdlst:
        localtoken += 1
        types, rivecs, distvecs, vocab = check_reps(w, types, indexfunc, dimen, nonzeros, corr, use_rivecs, rivecs, distvecs, vocab)
        wind = get_index(w, vocab)
        wvec = get_vec(w, vocab, distvecs)
        vocab[w][1] += 1
        lind = 1
        rwin = random.randint(1,win)
        while (lind < rwin+1) and ((ind+lind) < stop):
            c = wrdlst[ind+lind]
            types, rivecs, distvecs, vocab = check_reps(w, types, indexfunc, dimen, nonzeros, corr, use_rivecs, rivecs, distvecs, vocab)
            cind = get_index(c, vocab)
            cvec = get_vec(c, vocab, distvecs)
### The expensive part is here - we should be able to optimize this (perhaps using Cython?)
            np.add.at(wvec, rivecs[cind][:,0]+pi, rivecs[cind][:,1]*weight_func(get_freq(c, vocab), types, delta))
            np.add.at(cvec, rivecs[wind][:,0]-pi, rivecs[wind][:,1]*weight_func(get_freq(w, vocab), types, delta))
            lind += 1
        ind += 1
    return localtoken, types, distvecs, rivecs, vocab

def make_index(corr, dimen, nonzeros):
    ret = []
    inds = nprnd.randint(dimen-2, size=nonzeros) # dimen-2 to facilitate directional permutation
    signs = []
    for i in inds:
        sign = nprnd.randint(0,2)*2-1
        signs.append(sign)
        ret.append([i+1, sign]) # +1 to facilitate directional permutation
    # ret = sp.csr_matrix((signs, (np.zeros(len(signs)), inds+1)), shape=(1, dimen))
    if corr:
        ritmp = np.zeros(dimen)
        np.add.at(ritmp, inds+1, signs)
        return np.array(ret), ritmp
    else:
        return np.array(ret)
    # return ret

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

def make_ri_vecs(nr, dimen, nonzeros, corr, control):
    rivecs = []
    rivecs_full = []
    cnt = 0
    while cnt < nr:
        if corr:
            rive, rive_full = make_index(corr, dimen, nonzeros)
            rivecs_full.append(rive_full)
            rivecs.append(rive)
            cnt += 1
        elif control:
            rive = make_index(corr, dimen, nonzeros)
            ## COMPUTE COS TO ALL EXISTING VECS AND INCLUDE NEW ONE ONLY IF COS = 0
        else:
            rive = make_index(corr, dimen, nonzeros)
            rivecs.append(rive)
            cnt += 1
    return rivecs, rivecs_full

####################################
# Online ngrams
####################################

def update_vecs_ngrams(wrdlst, win, dimen, nonzeros, delta, theta, tokens, types, ngrams, vocab, rivecs, distvecs, indexfunc, use_rivecs, corr):
    localtoken = 0
    ind = 0
    stop = len(wrdlst)
    ngramlst = make_ngrams(wrdlst, vocab)
    for w in ngramlst:
        localtoken += 1
        types, rivecs, distvecs, vocab = check_reps(w, types, indexfunc, dimen, nonzeros, corr, use_rivecs, rivecs, distvecs, vocab)
        wind = get_index(w, vocab)
        wvec = get_vec(w, vocab, distvecs)
        vocab[w][1] += 1
        lwin = win
        lind = 1
        while (lind < lwin+1) and ((ind+lind) < stop):
            c = ngramlst[ind+lind]
            bigram = check_ngram(w, c, vocab, rivecs, distvecs, theta)
            if bigram:
                types, rivecs, distvecs, vocab = check_reps(bigram, types, indexfunc, dimen, nonzeros, use_rivecs, corr, use_rivecs, rivecs, distvecs, vocab)
                w = bigram
                ngrams += 1
                vocab[w][1] += 1
                wind = get_index(w, vocab)
                wvec = get_vec(w, vocab, distvecs)
                lwin = lwin + win
            else:
                types, rivecs, distvecs, vocab = check_reps(c, types, indexfunc, dimen, nonzeros, use_rivecs, corr, use_rivecs, rivecs, distvecs, vocab)
                cind = get_index(c, vocab)
                cvec = get_vec(c, vocab, distvecs)
                np.add.at(wvec, rivecs[cind][:,0]+1, rivecs[cind][:,1]*weight_func(get_freq(c, vocab), types, delta))
                np.add.at(cvec, rivecs[wind][:,0]-1, rivecs[wind][:,1]*weight_func(get_freq(w, vocab), types, delta))
            lind += 1
        ind += 1
    return localtoken, types, ngrams, distvecs, rivecs, vocab

def make_ngrams(sentencelist, vocab):
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

def check_ngram(word1, word2, vocab, rivecs, distvecs, theta):
    vec1 = get_vec(word1, vocab, distvecs)
    ri1 = get_ri(word1, vocab, rivecs)
    if word2 in vocab:
        ri2 = get_ri(word2, vocab, rivecs)
        vec2 = get_vec(word2, vocab, distvecs)
        tmp1 = np.zeros(dimen)
        tmp1 += vec1
        tmp2 = np.zeros(dimen)
        tmp2 += vec2
        np.multiply.at(tmp1, ri2[:,0], ri2[:,1]+1)
        np.multiply.at(tmp2, ri1[:,0], ri1[:,1]+1)
        freqprod = get_freq(word1, vocab) * (get_freq(word2, vocab) + 1)
        asum = (abs(np.sum(tmp1)) + abs(np.sum(tmp2))) / 2
        acnt = (asum / nonzeros) / freqprod
        if acnt > theta:
            bigram = word1 + '_' + word2
            return bigram
        else:
            return False
    else:
        return False

def get_ngrams(vocab):
    ret = []
    for key in vocab:
        if '_' in key:
            ret.append(key)
    return ret

####################################
# RI correction
# as suggested by Fredrik Sandin
####################################

ristds = []
nr_corrs = 0
ri_corrs = {}

def check_ri(ri1, ritmp, vec2, rotation, term, corr_thresh=0.2):
    rolledvec = np.roll(ritmp, rotation) # rotate
    vecsim = 1 - st.distance.cosine(rolledvec, vec2)
    if vecsim > corr_thresh: # expected
        tmpvec = np.zeros(dimen)
        tmpvec += vec2
        prodvec = tmpvec * rolledvec
        prodvec_full = abs(prodvec[np.nonzero(prodvec)])
        if len(prodvec_full) > 0:
            # expected value (~ co-occurrence frequency)
            expected = np.mean(prodvec_full) # using mean
            # expected = np.median(prodvec_full) # using median
            # expected = ss.mode(prodvec_full)[0][0]
            # ristds.append(np.std(prodvec_full))
            prodvecmax = np.max(prodvec)
            maxbytwo = prodvecmax/2 # max divided by two
            prodvecmin = np.min(prodvec)
            mintimestwo = prodvecmin*2 # min times two
            maxes = [i for i, x in enumerate(prodvec) if x == prodvecmax]
            mins = [i for i, x in enumerate(prodvec) if x == prodvecmin]
            if (maxbytwo > expected) and (len(maxes) == 1):
                maxind = np.argmax(prodvec)-rotation # undo rotation
                correct_vecs(maxind, expected, ri1, ritmp, vec2, term, 'max')
            elif (mintimestwo < expected) and (len(mins) == 1):
                minind = np.argmin(prodvec)-rotation # undo rotation
                correct_vecs(minind, expected, ri1, ritmp, vec2, term, 'min')

# thresh = expected above
# def find_other_elements(vec_array, thresh):
#     high_values_indices = vec_array > thresh
#     if not 

def correct_vecs(vec_ind, expected, ri1, ritmp, vec2, term, corr_type):
    global nr_corrs
    global ri_corrs
    maxind_ri = list(ri1[:,0]).index(vec_ind)
    newind = make_new_index(ritmp, vec2, expected)
    if newind is None:
        newind = 1
    # else:
    #     newind -= 1
    sign = nprnd.randint(0,2)*2-1
    ri1[maxind_ri] = np.array([newind, sign])
    ritmp[vec_ind] = 0.0
    ritmp[newind] = sign
    if term in ri_corrs:
        ri_corrs[term] += 1
    else:
        ri_corrs[term] = 1
    nr_corrs += 1
    ## normalize max value of ctx vec, and increase the new index to the expected value
    if corr_type == 'max':
        vec2[vec_ind] -= expected
    else:
        if vec2[vec_ind] > 0:
            vec2[vec_ind] += expected
        else:
            vec2[vec_ind] -= expected
    vec2[newind] += expected

def make_new_index(ri_vec, ctx_vec, expected_val):
    # global ristds
    # mean_std = np.mean(ristds)
    newind = random.randint(1,dimen-2)
    # need to include mean val in vector in the expectation comparison
    if (newind not in ri_vec) and (abs(ctx_vec[newind]) < expected_val * 2): # + mean_std):
        return newind
    else:
        make_new_index(ri_vec, ctx_vec, expected_val)

####################################
# Vector operations
####################################

# online frequency weight defined in:
# Sahlgren et al. (2016) The Gavagai Living Lexicon, LREC
def weight_func(freq, words, delta):
    return math.exp(-delta*(freq/words))

# remove centroid
def remove_centroid(model):
    tmpmat = np.asmatrix(model)
    sumvec = np.sum(tmpmat, axis=0)
    sumnorm = sumvec / np.linalg.norm(sumvec)
    cnt = 0
    for d in model:
        model[cnt] = np.subtract(d, (np.multiply(d, sumnorm) / np.linalg.norm(sumnorm)))
        cnt += 1

# remove dimensions with high variance
def prune_high_var(nrstd, model):
    tmpmat = np.asmatrix(model)
    varvec = np.var(tmpmat, 0)
    varmean = np.mean(varvec)
    varstd = np.std(varvec)
    prunevec = np.zeros(len(model[0]))
    ind = 0
    for i in np.nditer(varvec):
        if (i > (varmean + (varstd * nrstd))) or (i < (varmean - (varstd * nrstd))):
            prunevec[ind] = 0.0
        else:
            prunevec[ind] = 1.0
        ind += 1
    cnt = 0
    for d in model:
        model[cnt] = d*prunevec
        cnt += 1
        
# remove dimensions with low variance
def prune_low_var(nrstd, model):
    tmpmat = np.asmatrix(model)
    varvec = np.var(tmpmat, 0)
    varmean = np.mean(varvec)
    varstd = np.std(varvec)
    prunevec = np.zeros(len(model[0]))
    ind = 0
    for i in np.nditer(varvec):
        if (i < (varmean + (varstd * nrstd))) and (i > (varmean - (varstd * nrstd))):
            prunevec[ind] = 1.0
        else:
            prunevec[ind] = 0.0
        ind += 1
    cnt = 0
    for d in model:
        model[cnt] = d*prunevec
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

def make_ri_weights(rivecs):
    sums = np.sum(make_ri_matrix(rivecs), axis=0)
    max_val = np.max(sums)
    ret = []
    for i in np.abs(sums):
        if i > (max_val / 2):
            ret.append(1.0)
            # ret.append(0.1)
        else:
            va = int(round((i / max_val) * 10)) / 10
            # va = int(round((1 - (i / max_val)) * 10)) / 10
            ret.append(va)
            # ret.append(1 - (i / max_val))
            # ret.append(1.0)
    return ret

def make_projection_matrix(rivectors):
    rimat = sp.csr_matrix(make_ri_matrix(rivectors))
    return np.dot(np.transpose(rimat), rimat)

####################################
# Similarity
####################################

def sim(word1, word2, model, vocab):
    return 1 - st.distance.cosine(model[vocab[word1][0]], model[vocab[word2][0]])

def corr_sim(word1, word2, projection_matrix, model, vocab):
    return st.distance.euclidean(model[vocab[word1][0]], model[vocab[word2][0]])

def synt_sim(word1, word2, rot, synt_matrix, model, vocab):
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

def run_test_suite(model, vocab):
    print("TOEFL - ", end='')
    vocabulary_test("/home/mange/data/tests/toefl.txt", model, vocab)
    print("ESL - ", end='')
    vocabulary_test("/home/mange/data/tests/esl.txt", model, vocab)
    print("SimLex-999 - ", end='')
    similarity_test("/home/mange/data/tests/simlex.txt", model, vocab)
    print("MEN - ", end='')
    similarity_test("/home/mange/data/tests/men.txt", model, vocab)
    print("RW - ", end='')
    similarity_test("/home/mange/data/tests/rw.txt", model, vocab)

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

def similarity_test_output(testfile, outfile, model, vocab):
    with open(testfile, 'r') as inf, open(outfile, 'w') as outf:
        for line in inf.readlines():
            word1, word2, sim = line.split()
            if (word1 in vocab) and (word2 in vocab):
                res = 1 - st.distance.cosine(model[get_index(word1)], model[get_index(word2)])
                outf.write(word1+' '+word2+' '+str(res)+'\n')
