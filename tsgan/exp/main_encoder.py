from data import ucr
from tsgan.lib.ops import *
from tsgan.models.model import TSGAN
from tsgan.models.config import Config
from tsgan.models.encoder import TSGANEncoder

import numpy as np
import os
import tensorflow as tf
tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True


def encode(data_name, dir_data, feature_type, norm_type, dir_gan):
    tf.reset_default_graph()
    ## load data
    x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
    x_tr = np.reshape(x_tr, x_tr.shape + (1, 1))
    x_te = np.reshape(x_te, x_te.shape + (1, 1))
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
        features_tr = encoder.encode(sess, x_tr, conf.batch_size, norm=norm_type)
        features_te = encoder.encode(sess, x_te, conf.batch_size, norm=norm_type)
        return features_tr, y_tr, features_te, y_te, n_classes



