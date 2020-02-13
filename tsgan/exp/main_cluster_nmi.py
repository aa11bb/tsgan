"""
    Want to compare with the paper: ﻿Similarity preserving representation learning for time series clustering
    it takes the metrics of: ﻿normalized mutual information (NMI for short)
"""

from tsgan.exp.base import tag_path
from tsgan.exp import base
from tsgan.exp.main_encoder import encode

import numpy as np
import os
import pandas as pd
from time import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def kmeans(x, y, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    t_start = time()
    model.fit(x)
    t = time() - t_start
    nmi = normalized_mutual_info_score(y, model.labels_)
    return nmi, t


def run_feature(feature_type, norm_type, data_name_list, dir_gan, dir_data, dir_out):
    print("******** kmeans over features")
    res = {'dataset': [], 'score': [], 'time': []}
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
        res['score'].append(ri)
        res['time'].append(t)
    ## save result
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(dir_out,'kmeans_{}_{}.csv'.format(feature_type, norm_type)), index=False)
    return df

if __name__ == '__main__': # multiple runs
    dir_data = base.UCR_DIR
    tag = tag_path(os.path.abspath(__file__), 1)

    # data_name_list = ucr.get_data_name_list(dir_data)
    data_name_list = ['ArrowHead']

    dir_gan = 'cache/main_gan_half'
    tag_exp = os.path.basename(dir_gan)
    dir_out = 'cache/{}/{}'.format(tag, tag_exp)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)
    run_feature('local-max', 'znorm', data_name_list, dir_gan, dir_data, dir_out)

