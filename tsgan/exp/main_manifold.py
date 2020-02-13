"""
    Using t-SNE to learn a low-dimensional embedding.
"""

from tsgan.exp.base import tag_path
from data import ucr
from tsgan.exp import base
from tsgan.exp.main_encoder import encode

import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 86
np.random.seed(SEED)


def run_manifold(x_tr, y_tr, x_te, y_te, dir_out):
    n_components = 2
    model = TSNE(n_components)

    manifold_tr = model.fit_transform(x_tr)
    plt.scatter(manifold_tr[:, 0], manifold_tr[:, 1], c=y_tr)
    plt.savefig(os.path.join(dir_out, 'tr.png'))
    plt.clf()

    manifold_te = model.fit_transform(x_te)
    plt.scatter(manifold_te[:, 0], manifold_te[:, 1], c=y_te)
    plt.savefig(os.path.join(dir_out, 'te.png'))
    plt.clf()

    x = np.vstack([x_tr, x_te])
    y = np.hstack([y_tr, y_te])
    manifold = model.fit_transform(x)
    plt.scatter(manifold[:, 0], manifold[:, 1], c=y)
    plt.savefig(os.path.join(dir_out, 'all.png'))
    plt.clf()

    np.savetxt(os.path.join(dir_out, 'tr'), manifold_tr, delimiter=',')
    np.savetxt(os.path.join(dir_out, 'te'), manifold_te, delimiter=',')

def run_tsgan(data_name, dir_data, dir_out):
    path_out = os.path.join(dir_out, data_name, 'tsgan')
    if os.path.exists(path_out) is False:
        os.makedirs(path_out)

    feature_type = 'local-max'
    norm_type = 'znorm'
    dir_gan = 'cache/main_gan_half'
    features_tr, y_tr, features_te, y_te, n_classes = encode(
        data_name, dir_data, feature_type, norm_type, dir_gan)

    np.savetxt(os.path.join(path_out, 'trx'), features_tr, delimiter=',')
    np.savetxt(os.path.join(path_out, 'try'), y_tr, delimiter=',')
    np.savetxt(os.path.join(path_out, 'tex'), features_te, delimiter=',')
    np.savetxt(os.path.join(path_out, 'tey'), y_te, delimiter=',')

    run_manifold(features_tr, y_tr, features_te, y_te, path_out)

def run_raw(data_name, dir_data, dir_out):
    path_out = os.path.join(dir_out, data_name, 'raw')
    if os.path.exists(path_out) is False:
        os.makedirs(path_out)

    x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)

    np.savetxt(os.path.join(path_out, 'try'), y_tr, delimiter=',')
    np.savetxt(os.path.join(path_out, 'tey'), y_te, delimiter=',')

    run_manifold(x_tr, y_tr, x_te, y_te, path_out)


if __name__ == '__main__':
    dir_data = base.UCR_DIR
    tag = tag_path(os.path.abspath(__file__), 1)
    dir_out = 'cache/{}'.format(tag)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)

    data_name_list = ucr.get_data_name_list(dir_data)
    data_name_list = ['ArrowHead']

    for data_name in data_name_list:
        run_tsgan(data_name, dir_data, dir_out)

    for data_name in data_name_list:
        run_raw(data_name, dir_data, dir_out)


