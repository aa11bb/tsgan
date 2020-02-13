from tsgan.exp.base import tag_path
from data import ucr
from tsgan.exp import base
from tsgan.models.model import TSGAN
from tsgan.models.config import Config
from tsgan.exp.main_encoder import TSGANEncoder

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
import matplotlib.pyplot as plt
import tensorflow as tf
tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True


def encode_on_batch(encoder, sess, X, batch_size):
    layer1, layer2 = encoder.get_features_local_max_detail() # fixed
    res1 = [[] for _ in range(len(layer1))]
    res2 = [[] for _ in range(len(layer2))]

    n_samples = len(X)
    n_batches = n_samples // batch_size
    for i in range(n_samples // batch_size):
        x_batch = X[i * batch_size:(i + 1) * batch_size]
        vals1, vals2 = sess.run([layer1, layer2], feed_dict={encoder.x: x_batch})
        for res, val in zip(res1, vals1):
            res.append(val)
        for res, val in zip(res2, vals2):
            res.append(val)
    n_samples_left = n_samples - n_batches * batch_size
    if n_samples_left > 0:
        x_left = X[-n_samples_left:]
        vals1, vals2 = sess.run([layer1, layer2], feed_dict={encoder.x: x_left})
        for res, val in zip(res1, vals1):
            res.append(val)
        for res, val in zip(res2, vals2):
            res.append(val)

    res1 = [np.vstack(r) for r in res1]
    res2 = [np.vstack(r) for r in res2]

    return res1, res2

def encode(data_name, dir_data, feature_type, dir_gan):
    tf.reset_default_graph()
    ## load data
    x_tr_2d, y_tr, _, _, n_classes = ucr.load_ucr_flat(data_name, dir_data)
    x_tr = np.reshape(x_tr_2d, x_tr_2d.shape + (1, 1))
    ## set up GAN
    dir_checkpoint = os.path.join(dir_gan, data_name, 'checkpoint')
    conf = Config(x_tr, '', '', x_tr.shape[1], x_tr.shape[2], x_tr.shape[3], state='test')
    gan = TSGAN(conf.dim_z, conf.dim_h, conf.dim_w, conf.dim_c, conf.random_seed,
                conf.g_lr, conf.d_lr, conf.g_beta1, conf.d_beta1, conf.gf_dim, conf.df_dim)
    ## start to run
    with tf.Session(config=tf_conf) as sess:
        isload, counter = gan.load(sess, dir_checkpoint)
        if not isload:
            raise Exception("[!] Train a model first, then run test mode")
        input_shape = [x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]]
        encoder = TSGANEncoder(gan, input_shape, type=feature_type)

        layer1, layer2 = encode_on_batch(encoder, sess, x_tr, conf.batch_size)

        return x_tr_2d, y_tr, layer1, layer2

def vis_fm(X_2d, y, d_fm_list, dir_out, tag="", i_channel=0, n_samples=2):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    n_colors = len(colors)
    n_classes = len(np.unique(y))
    if n_classes > n_colors:
        import warnings
        warnings.warn("Too classes to show. Such a bad visualization has been stopped.")
        return

    # construct data
    i_sample = 0
    rows = []
    convs = {} # convs[l] correspond to layer l
    for l in np.arange(len(d_fm_list)):
        convs[l] = []
    classes, classes_count = np.unique(y, return_counts=True)
    classes = list(classes)
    # find a class with multiple samples for visualization.
    clas1 = None
    for cla, co in zip(classes, classes_count):
        if co > 1:
            clas1 = cla
            break
    classes.remove(clas1)
    classes.insert(0, clas1)
    group = y == clas1
    xc = X_2d[group]
    rows.append(xc[:n_samples])
    for l, h in enumerate(d_fm_list):
        hc = h[group, :, :, i_channel]
        convs[l].append(hc[:n_samples])
    for c in classes: # for each class excepting the first
        if c == clas1:
            continue
        group = y == c
        xc = X_2d[group]
        rows.append(xc[i_sample][np.newaxis, :])
        for l, h in enumerate(d_fm_list):
            hc = h[group, :, :, i_channel]
            convs[l].append(hc[i_sample][np.newaxis, :])

    # plot row data
    plt_lines = []
    plt_labels = []
    for c, samples in zip(classes, rows):
        line = None
        for s in samples:
            line, = plt.plot(s, '.-', color=colors[c])
        plt_lines.append(line)
        plt_labels.append('class-{}'.format(c))
    # plt.legend(plt_lines, plt_labels, loc='best')
    plt.savefig('{}/{}_rows.png'.format(dir_out, tag))
    plt.clf()

    # plot feature maps
    for l, vals in convs.items():
        plt_lines = []
        plt_labels = []
        for c, samples in zip(classes, vals):
            line = None
            for s in samples:
                line, = plt.plot(s, '.-', color=colors[c])
            plt_lines.append(line)
            plt_labels.append('class-{}'.format(c))
        # plt.legend(plt_lines, plt_labels, loc='best')
        plt.savefig("{}/{}_convs_{}.png".format(dir_out, tag, l))
        plt.clf()
    plt.close('all')

if __name__ == '__main__':
    dir_data = base.UCR_DIR
    tag = tag_path(os.path.abspath(__file__), 1)
    dir_out = 'cache/{}'.format(tag)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)

    dir_gan = 'cache/main_gan_half'
    data_name = 'ArrowHead'
    # data_name_list = ucr.get_data_name_list(dir_data)
    data_name_list = ['ArrowHead']
    for data_name in data_name_list:
        x_2d, y_tr, layer1, layer2 = encode(data_name, dir_data, 'local-max', dir_gan)
        for i_channel in range(1, 64):
            path_out = os.path.join(dir_out, data_name, '{}'.format(str(i_channel).zfill(2)))
            if os.path.exists(path_out) is False:
                os.makedirs(path_out)
            vis_fm(x_2d, y_tr, layer1[:-2], path_out, tag='layer1', i_channel=i_channel)
            vis_fm(x_2d, y_tr, layer2[:-2], path_out, tag='layer2', i_channel=i_channel)




