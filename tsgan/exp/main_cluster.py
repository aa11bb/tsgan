"""
    Using rand index as metric
"""

from tsgan.exp.base import tag_path
from data import ucr
from tsgan.exp import base
from tsgan.exp.main_encoder import encode

import numpy as np
import os
import pandas as pd
from time import time
from sklearn.cluster import KMeans


def rand_index(y_actual, y_pred):
    """
    Test results on k-means-ed are consistent to the literature: Zakaria, Jesin, Abdullah Mueen, and Eamonn Keogh. "Clustering time series using unsupervised-shapelets." 2012 IEEE 12th International Conference on Data Mining. IEEE, 2012.
    The definition can be referred to : Paparrizos, John, and Luis Gravano. "k-shape: Efficient and accurate clustering of time series." Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. ACM, 2015.
    :param y_actual: 
    :param y_pred: 
    :return: 
    """
    n = len(y_actual)
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(n):
        for j in range(i, n):
            if y_actual[i] == y_actual[j] and y_pred[i] == y_pred[j]:
                tp += 1
            elif y_actual[i] == y_actual[j] and y_pred[i] != y_pred[j]:
                fn += 1
            elif y_actual[i] != y_actual[j] and y_pred[i] == y_pred[j]:
                fp += 1
            elif y_actual[i] != y_actual[j] and y_pred[i] != y_pred[j]:
                tn += 1

    return (tp + tn) / (tp + fp + fn + tn)

def kmeans(x, y, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    t_start = time()
    model.fit(x)
    t = time() - t_start
    ri = rand_index(y, model.labels_)
    return ri, t


def run_feature(feature_type, norm_type, data_name_list, dir_gan, dir_data, dir_out):
    print("******** kmeans over features")
    res = {'dataset': [], 'randIndex': [], 'time': []}
    n_datasets = len(data_name_list)
    for i, data_name in enumerate(data_name_list):
        print("******** [{}/{}] processing {}".format(i, n_datasets, data_name))
        ## load data
        features_tr, y_tr, features_te, y_te, n_classes = encode(
            data_name, dir_data, feature_type, norm_type, dir_gan)
        features = np.vstack([features_tr, features_te])
        y = np.hstack([y_tr, y_te])
        ri, t = kmeans(features, y, n_classes)
        print(data_name, ri, t)
        res['dataset'].append(data_name)
        res['randIndex'].append(ri)
        res['time'].append(t)
    ## save result
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(dir_out,'kmeans_{}_{}.csv'.format(feature_type, norm_type)), index=False)
    return df


def run_raw(data_name_list, dir_data, dir_out):
    print("******** kmeans over raw data")
    res = {'dataset': [], 'randIndex':[], 'time':[]}
    n_datasets = len(data_name_list)
    for i, data_name in enumerate(data_name_list):
        print("******** [{}/{}] processing {}".format(i, n_datasets, data_name))
        ## load data
        x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
        x = np.vstack([x_tr, x_te])
        y = np.hstack([y_tr, y_te])
        ## start to run
        ri, t = kmeans(x, y, n_classes)
        res['dataset'].append(data_name)
        res['randIndex'].append(ri)
        res['time'].append(t)
    ## save result
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(dir_out, 'kmeans_raw.csv'), index=False)
    return df

if __name__ == '__main__': # unit test
    dir_data = base.UCR_DIR
    tag = tag_path(os.path.abspath(__file__), 1)

    # data_name_list = ucr.get_data_name_list(dir_data)
    data_name_list = ['ArrowHead']

    ## for raw data
    dir_out = 'cache/{}/{}'.format(tag, 'raw')
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)
    run_raw(data_name_list, dir_data, dir_out)

    ## tsgan
    dir_gan = 'cache/main_gan_half'
    tag_exp = os.path.basename(dir_gan)
    dir_out = 'cache/{}/{}'.format(tag, tag_exp)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)
    run_feature('local-max', 'znorm', data_name_list, dir_gan, dir_data, dir_out)

