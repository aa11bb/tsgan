"""
    Testify to train NN-based models with smaller labeled dataset.
    Note: the dataset should be large enough to be split for experiments.
"""

import matplotlib
matplotlib.use('Agg')

from tsgan.exp.base import tag_path
from tsgan.exp import base
from data import ucr, utils
from tsc.nn import ResNet, FCN
from tsgan.exp.main_encoder import encode
from tsgan.models import train
from tsgan.models.config import Config

import os
import pandas as pd
import numpy as np

import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def run_resnet(data_name, dir_data, ratios):
    x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
    x_tr = x_tr.reshape(x_tr.shape + (1,))
    x_te = x_te.reshape(x_te.shape + (1,))
    y_te_onehot = utils.dense_to_one_hot(y_te, n_classes)
    n_epochs = 200
    res = {'ratio': [], 'distr': [], 'acc': [], 'acc_te': []}
    for r in ratios:
        print("*** processing ratio={}".format(r))
        x_tr_cur, y_tr_cur, _, _ = utils.split_stratified(x_tr, y_tr, r)
        y_tr_cur_onehot = utils.dense_to_one_hot(y_tr_cur, n_classes)
        model = ResNet(x_tr_cur.shape[1:], n_classes)
        df_metrics = model.fit(x_tr_cur, y_tr_cur_onehot, n_epochs=n_epochs)
        acc_te = model.evaluate(x_te, y_te_onehot)
        last = df_metrics.loc[df_metrics.shape[0] - 1, :]
        res['ratio'].append(r)
        res['distr'].append(utils.distribute_y_json(y_tr_cur))
        res['acc'].append(last['acc'])
        res['acc_te'].append(acc_te)
    df_res = pd.DataFrame(res)
    return df_res

def run_fcn(data_name, dir_data, ratios):
    x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
    x_tr = x_tr.reshape(x_tr.shape + (1,))
    x_te = x_te.reshape(x_te.shape + (1,))
    y_te_onehot = utils.dense_to_one_hot(y_te, n_classes)
    n_epochs = 200
    res = {'ratio': [], 'distr': [], 'acc': [], 'acc_te': []}
    for r in ratios:
        print("*** processing ratio={}".format(r))
        x_tr_cur, y_tr_cur, _, _ = utils.split_stratified(x_tr, y_tr, r)
        y_tr_cur_onehot = utils.dense_to_one_hot(y_tr_cur, n_classes)
        model = FCN(x_tr_cur.shape[1:], n_classes)
        df_metrics = model.fit(x_tr_cur, y_tr_cur_onehot, n_epochs=n_epochs)
        acc_te = model.evaluate(x_te, y_te_onehot)
        last = df_metrics.loc[df_metrics.shape[0] - 1, :]
        res['ratio'].append(r)
        res['distr'].append(utils.distribute_y_json(y_tr_cur))
        res['acc'].append(last['acc'])
        res['acc_te'].append(acc_te)
    df_res = pd.DataFrame(res)
    return df_res

def run_tsgan(data_name, dir_data, ratios, tag):
    ## train gan
    tf.reset_default_graph()
    x_tr, y_tr, _, _, _ = ucr.load_ucr_flat(data_name, dir_data)
    x = x_tr
    x = np.reshape(x, x.shape + (1, 1))
    conf = Config(x, data_name, tag, x.shape[1], x.shape[2], x.shape[3], state='train')
    train.train(conf)

    ## classification
    dir_gan = 'cache/{}'.format(tag)
    feature_type = 'local-max'
    norm_type = 'tanh'
    model_name = 'LR'
    model = base.StandardClassifierDic[model_name]
    features_tr, y_tr, features_te, y_te, n_classes = encode(
        data_name, dir_data, feature_type, norm_type, dir_gan)
    res = {'ratio': [], 'distr': [], 'acc': [], 'acc_te': []}
    for r in ratios:
        features_tr_cur, y_tr_cur, _, _ = utils.split_stratified(features_tr, y_tr, r)
        acc, t = base.classify(model, features_tr_cur, y_tr_cur, features_te, y_te)
        res['ratio'].append(r)
        res['distr'].append(utils.distribute_y_json(y_tr_cur))
        res['acc'].append(acc[0])
        res['acc_te'].append(acc[1])
    df_res = pd.DataFrame(res)
    return df_res

def run_standard(data_name, dir_data, ratios):
    model_name = 'LR'
    model = base.StandardClassifierDic[model_name]
    x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
    res = {'ratio': [], 'distr': [], 'acc': [], 'acc_te': []}
    for r in ratios:
        x_tr_cur, y_tr_cur, _, _ = utils.split_stratified(x_tr, y_tr, r)
        acc, t = base.classify(model, x_tr_cur, y_tr_cur, x_te, y_te)
        res['ratio'].append(r)
        res['distr'].append(utils.distribute_y_json(y_tr_cur))
        res['acc'].append(acc[0])
        res['acc_te'].append(acc[1])
    df_res = pd.DataFrame(res)
    return df_res


if __name__ == '__main__':
    dir_data = base.UCR_DIR
    tag = tag_path(os.path.abspath(__file__), 1)
    dir_out = 'cache/{}'.format(tag)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)

    data_name_list = ['FordA', 'wafer']
    ratios = [1.0, 0.8, 0.6, 0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.01]
    n_runs = 10
    for data_name in data_name_list:
        dir_out_cur = os.path.join(dir_out, data_name)
        if os.path.exists(dir_out_cur) is False:
            os.makedirs(dir_out_cur)
        ## RestNet
        for i in range(n_runs):
            df_res = run_resnet(data_name, dir_data, ratios)
            df_res.to_csv(os.path.join(dir_out_cur, 'resnet_{}.csv'.format(i)), index=False)
        ## FCN
        # for i in range(n_runs):
        #     df_res = run_fcn(data_name, dir_data, ratios)
        #     df_res.to_csv(os.path.join(dir_out_cur, 'fcn_{}.csv'.format(i)), index=False)
        ## TSGAN
        # for i in range(n_runs):
        #     tag_cur = '{}/tsgan/run{}'.format(tag, i)
        #     df_res = run_tsgan(data_name, dir_data, ratios, tag_cur)
        #     df_res.to_csv(os.path.join(dir_out_cur, 'tsgan_{}.csv'.format(i)), index=False)
        # ## Standard
        # for i in range(n_runs):
        #     df_res = run_standard(data_name, dir_data, ratios)
        #     df_res.to_csv(os.path.join(dir_out_cur, 'standard_{}.csv'.format(i)), index=False)








