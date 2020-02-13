import numpy as np
from sklearn.utils.validation import column_or_1d
import json

from sklearn import preprocessing

def z_normalize(data, axis=1):
    return preprocessing.scale(data, axis=axis)


def dense_to_one_hot(labels_dense, num_classes):
    # check input
    labels_dense = column_or_1d(labels_dense, warn=True)
    if min(labels_dense) < 0 or max(labels_dense) >= num_classes:
        raise ValueError(
            "The value of label range in [{}, {}], which out of the valid range [0, {}-1]".format(
                min(labels_dense), max(labels_dense), num_classes)
        )
    # dense to one hot
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def distribute_dataset(X, y):
    y = column_or_1d(y, warn=True)
    res = {}
    for id in np.unique(y):
        res[id] = X[y==id]
    return res


def distribute_y(y):
    y = column_or_1d(y)
    res = {}
    for id in np.unique(y):
        res[id] = np.sum(y==id)
    return res


def distribute_y_json(y):
    y = column_or_1d(y)
    res = {}
    for id in np.unique(y):
        res[str(id)] = str(np.sum(y==id))
    return json.dumps(res)



def dic2json(in_dict):
    """
    :param in_dict: 
    :return: 
    """
    local_dict = {}
    for key in in_dict.keys():
        if type(key) is not str:
            try:
                local_dict[str(key)] = in_dict[key]
            except:
                try:
                    local_dict[str(key)] = in_dict[key]
                except:
                    raise TypeError("current key:{} can't convert to str type!".format(key))
        else:
            local_dict[key] = in_dict[key]
    return json.dumps(local_dict)

def shuffle_dataset(X, y=None):
    n = X.shape[0]
    inds = np.arange(n)
    np.random.shuffle(inds)
    if y is None:
        return y[inds]
    else:
        return X[inds], y[inds]


def split_random(X, ratio, y=None):
    n = X.shape[0]
    ind = np.arange(n)
    np.random.shuffle(ind)

    n_left = int(ratio * n)
    ind_left = ind[0:n_left]
    ind_right = ind[n_left::]

    if y is not None:
        return X[ind_left], y[ind_left], X[ind_right], y[ind_right]
    else:
        return X[ind_left], X[ind_right]

def split_stratified(X, y, ratio):
    distr = distribute_dataset(X, y)
    X_left, y_left = [], []
    X_right, y_right = [], []

    for key, X_batch in distr.items():
        n = X_batch.shape[0]
        i = int(np.ceil(n * ratio))
        X_left.append(X_batch[:i])
        y_left.extend([key]*i)
        X_right.append(X_batch[i:])
        y_right.extend([key]*(n-i))

    X_left, y_left = np.vstack(X_left), np.array(y_left)
    X_right, y_right = np.vstack(X_right), np.array(y_right)

    X_left, y_left = shuffle_dataset(X_left, y_left)
    X_right, y_right = shuffle_dataset(X_right, y_right)

    return X_left, y_left, X_right, y_right



