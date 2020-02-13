"""
    Script to train GAN.
"""

import matplotlib
matplotlib.use('Agg')

from data import ucr
from tsgan.exp.base import tag_path
from tsgan.models.config import Config
from tsgan.models import train
from tsgan.exp import base

import pandas as pd
import numpy as np
import os
import argparse
import shutil
import tensorflow as tf


def start_training(data_name_list, dir_data_root, mode, tag):
    n_datasets = len(data_name_list)
    for i, data_name in enumerate(data_name_list):
        print("******* [{}/{}] processing {}".format(i+1, n_datasets, data_name))
        tf.reset_default_graph()
        x_tr, _, x_te, _, _ = ucr.load_ucr_flat(data_name, dir_data_root)
        x_tr = np.reshape(x_tr, x_tr.shape + (1, 1))
        x_te = np.reshape(x_te, x_te.shape + (1, 1))
        if mode == 'half':
            x = x_tr
        elif mode == 'all':
            x = np.vstack([x_tr, x_te])
        else:
            raise ValueError("Can not find mode = {}".format(mode))
        conf = Config(x, data_name, tag, x.shape[1], x.shape[2], x.shape[3], state='train')
        train.train(conf)


def reduce_results(data_name_list, dir_log):
    import json
    ## make destination directory
    dir_dst = 'cache/results/{}'.format(os.path.basename(dir_log))
    if os.path.exists(dir_dst):
        shutil.rmtree(dir_dst)
    os.makedirs(dir_dst)

    ## copy the results to destination for each dataset
    for data_name in data_name_list:
        src = '{}/{}'.format(dir_log, data_name)
        dst = '{}/{}'.format(dir_dst, data_name)
        os.makedirs(dst)
        shutil.copytree('{}/logs'.format(src), '{}/logs'.format(dst))
        shutil.copytree('{}/samples'.format(src), '{}/samples'.format(dst))

    ## reduce the performance metrics, according to the last evaluation during training.
    res = {'dataset':[], 'nnd':[], 'mmd':[]}
    for data_name in data_name_list:
        path = os.path.join(dir_dst, data_name, 'logs', 'log_train.json')
        with open(path, 'r') as f:
            values = f.readlines()
        res['dataset'].append(data_name)
        nnd, mmd = 0, 0
        n_last = 3
        for i in range(1, n_last+1): # reduce the last 3
            row = json.loads(values[-i])
            nnd += np.float(row['nnd'])
            mmd += np.float(row['mmd'])
        nnd = nnd / n_last
        mmd = mmd / n_last
        res['nnd'].append(nnd)
        res['mmd'].append(mmd)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(dir_log, 'metrics_train.csv'), index=False)

    ## copy all .csv files
    csv_files = [f for f in os.listdir(dir_log) if f.endswith('.csv')]
    for f in csv_files:
        shutil.copy(os.path.join(dir_log, f), os.path.join(dir_dst, f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument(
        '--mode', type=str, default='half',
        help="'half', just use train split;"
             "'all', use all data (i.e. concat train and valid splits)"
    ) # example: python main_gan.py --mode=half

    ARGS, unparsed = parser.parse_known_args()
    mode = ARGS.mode
    dir_data_root = base.UCR_DIR
    tag = '{}_{}'.format(tag_path(os.path.abspath(__file__), 1), mode)

    # data_name_list = ucr.get_data_name_list(dir_data_root)
    data_name_list = ['ArrowHead']
    start_training(data_name_list, dir_data_root, mode, tag)
    # reduce_results(data_name_list, 'cache/{}'.format(tag))
