import theano
import theano.tensor as T
from sklearn import metrics
import numpy as np


#####################################################
# Basic operators based on theano
#
def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape) * n, dtype=dtype, name=name)


#####################################################
# metrics: cosine, euclidean, nnc, nnd
#
def l2normalize(x, axis=1, e=1e-8, keepdims=True):
    return x / l2norm(x, axis=axis, e=e, keepdims=keepdims)


def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)


def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d


def euclidean(x, y, e=1e-8):
    xx = T.sqr(T.sqrt((x * x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y * y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist)
    return dist


A = T.matrix()
B = T.matrix()
ed = euclidean(A, B)
cd = cosine(A, B)
cosine_dist = theano.function([A, B], cd)
euclid_dist = theano.function([A, B], ed)


def gpu_nnc_predict(trX, trY, teX, metric='cosine', batch_size=4096):
    if metric == 'cosine':
        metric_fn = cosine_dist
    else:
        metric_fn = euclid_dist
    idxs = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        mb_idxs = []
        for j in range(0, len(trX), batch_size):
            dist = metric_fn(floatX(teX[i:i + batch_size]), floatX(trX[j:j + batch_size]))
            if metric == 'cosine':
                mb_dists.append(np.max(dist, axis=1))
                mb_idxs.append(j + np.argmax(dist, axis=1))
            else:
                mb_dists.append(np.min(dist, axis=1))
                mb_idxs.append(j + np.argmin(dist, axis=1))
        mb_idxs = np.asarray(mb_idxs)
        mb_dists = np.asarray(mb_dists)
        if metric == 'cosine':
            i = mb_idxs[np.argmax(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        else:
            i = mb_idxs[np.argmin(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        idxs.append(i)
    idxs = np.concatenate(idxs, axis=0)
    nearest = trY[idxs]
    return nearest


def gpu_nnd_score(trX, teX, metric='cosine', batch_size=4096):
    if metric == 'cosine':
        metric_fn = cosine_dist
    else:
        metric_fn = euclid_dist
    dists = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        for j in range(0, len(trX), batch_size):
            dist = metric_fn(floatX(teX[i:i + batch_size]), floatX(trX[j:j + batch_size]))
            if metric == 'cosine':
                mb_dists.append(np.max(dist, axis=1))
            else:
                mb_dists.append(np.min(dist, axis=1))
        mb_dists = np.asarray(mb_dists)
        if metric == 'cosine':
            d = np.max(mb_dists, axis=0)
        else:
            d = np.min(mb_dists, axis=0)
        dists.append(d)
    dists = np.concatenate(dists, axis=0)
    return float(np.mean(dists))


def nnc_score(trX, trY, teX, teY, metric='euclidean'):
    pred = gpu_nnc_predict(trX, trY, teX, metric=metric)
    acc = metrics.accuracy_score(teY, pred)
    return acc * 100.


def nnd_score(trX, teX, metric='euclidean'):
    return gpu_nnd_score(trX, teX, metric=metric)
