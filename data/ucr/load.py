import numpy as np
import os

from .. import base
from .. import utils


def load_ucr_concat(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """

    :param data_name: 
    :param data_dir_root: 
    :param encode_label:
    :param one_hot: 
    :return: 
    """
    data = load_ucr(data_name, data_dir_root, encode_label, one_hot, z_norm)
    if data.valid is None:
        X_all = np.concatenate([data.train.X, data.test.X], axis=0)
        y_all = np.concatenate([data.train.y, data.test.y])
    else:
        X_all = np.concatenate([data.train.X, data.test.X, data.valid.X], axis=0)
        y_all = np.concatenate([data.train.y, data.test.y, data.valid.y], axis=0)
    return X_all, y_all, data.nclass


def load_ucr_flat(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """

    :param data_name: 
    :param data_dir_root: 
    :param encode_label: 
    :param one_hot: 
    :return: 
    """
    data = load_ucr(data_name, data_dir_root, encode_label, one_hot, z_norm)
    if data.valid is None:
        return data.train.X, data.train.y, data.test.X, data.test.y, data.nclass
    else:
        return data.train.X, data.train.y, data.test.X, data.test.y, data.valid.X, data.valid.y, data.nclass


def save_ucr(data_name, data_dir_root, X_train, y_train, X_test=None, y_test=None):
    length = X_train.shape[1]
    str_fmt = '%d,' + '%.4f,' * (length)
    str_fmt = str_fmt[:(len(str_fmt) - 1)]

    dir_out = os.path.join(data_dir_root, data_name)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)

    trainset = np.concatenate([y_train[:, np.newaxis], X_train], axis=1)
    np.savetxt(
        os.path.join(dir_out, '{}_TRAIN'.format(data_name)),
        trainset, fmt=str_fmt, delimiter=',')
    if X_test is not None and y_test is not None:
        testset = np.concatenate([y_test[:, np.newaxis], X_test], axis=1)
        np.savetxt(os.path.join(dir_out, '{}_TEST'.format(data_name)),
                   testset, fmt=str_fmt, delimiter=',')


def load_ucr(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """

    :param data_name: 
    :param data_dir_root: 
    :param encode_label: 
    :param one_hot: 
    :return: 
    """
    # Note:
    # The validation set {}_VALID didn't exist in default data set.
    # So we take this scenario into account specially before loading data.
    filename_valid = '{}/{}/{}_VALID'.format(data_dir_root, data_name, data_name)
    if os.path.exists(filename_valid):
        return _load_ucr_with_valid(data_name, data_dir_root, encode_label, one_hot, z_norm)
    else:
        return _load_ucr_origin(data_name, data_dir_root, encode_label, one_hot, z_norm)


def _load_ucr_origin(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """
    
    :param data_name: 
    :param data_dir_root: 
    :param one_hot: 
    :return: 
    """
    # load from file
    filename_train = '{}/{}/{}_TRAIN'.format(data_dir_root, data_name, data_name)
    filename_test = '{}/{}/{}_TEST'.format(data_dir_root, data_name, data_name)
    data_train = np.genfromtxt(filename_train, delimiter=',', dtype=np.float32)
    data_test = np.genfromtxt(filename_test, delimiter=',', dtype=np.float32)

    # parse
    X_train = data_train[:, 1::]
    y_train = data_train[:, 0].astype(int)
    X_test = data_test[:, 1::]
    y_test = data_test[:, 0].astype(int)
    y_all = np.concatenate([y_train, y_test])
    classes, y_all = np.unique(y_all, return_inverse=True)
    n_class = len(classes)

    # z_norm
    if z_norm:
        X_train = utils.z_normalize(X_train)
        X_test = utils.z_normalize(X_test)

    # reconstruct label
    if encode_label or one_hot:
        # encode labels with value between 0 and n_classes-1.
        y_train = y_all[:X_train.shape[0]]
        y_test = y_all[X_train.shape[0]:]
        # convert to one hot label in need
        if one_hot == True:
            y_train = utils.dense_to_one_hot(y_train, n_class)
            y_test = utils.dense_to_one_hot(y_test, n_class)

    # pack data set
    train_set = base.Dataset(X=X_train, y=y_train)
    valid_set = None
    test_set = base.Dataset(X=X_test, y=y_test)
    return base.Datasets(train=train_set, valid=valid_set, test=test_set, nclass=n_class)


def _load_ucr_with_valid(data_name, data_dir_root, encode_label=True, one_hot=False, z_norm=False):
    """

    :param data_name: 
    :param data_dir_root: 
    :param encode_label:
    :param one_hot: 
    :return: 
    """
    # load from file
    filename_train = '{}/{}/{}_TRAIN'.format(data_dir_root, data_name, data_name)
    filename_test = '{}/{}/{}_TEST'.format(data_dir_root, data_name, data_name)
    filename_valid = '{}/{}/{}_VALID'.format(data_dir_root, data_name, data_name)
    data_train = np.genfromtxt(filename_train, delimiter=',', dtype=np.float32)
    data_test = np.genfromtxt(filename_test, delimiter=',', dtype=np.float32)
    data_valid = np.genfromtxt(filename_valid, delimiter=',', dtype=np.float32)

    # parse
    X_train = data_train[:, 1::]
    y_train = data_train[:, 0].astype(int)
    X_test = data_test[:, 1::]
    y_test = data_test[:, 0].astype(int)
    X_valid = data_valid[:, 1::]
    y_valid = data_valid[:, 0].astype(int)
    y_all = np.concatenate([y_train, y_test, y_valid])
    classes, y_all = np.unique(y_all, return_inverse=True)
    n_class = len(classes)

    # z_norm
    if z_norm:
        X_train = utils.z_normalize(X_train)
        X_test = utils.z_normalize(X_test)
        X_valid = utils.z_normalize(X_valid)

    # reconstruct label
    if encode_label or one_hot:
        # encode labels with value between 0 and n_classes-1.
        y_train = y_all[:y_train.shape[0]]
        y_test = y_all[y_train.shape[0]:(y_train.shape[0] + y_test.shape[0])]
        y_valid = y_all[(y_train.shape[0] + y_test.shape[0]):]
        # convert to one hot label in need
        if one_hot == True:
            y_train = utils.dense_to_one_hot(y_train, n_class)
            y_test = utils.dense_to_one_hot(y_test, n_class)
            y_valid = utils.dense_to_one_hot(y_valid, n_class)

    # pack data set
    train_set = base.Dataset(X=X_train, y=y_train)
    valid_set = base.Dataset(X=X_valid, y=y_valid)
    test_set = base.Dataset(X=X_test, y=y_test)
    return base.Datasets(train=train_set, valid=valid_set, test=test_set, nclass=n_class)


