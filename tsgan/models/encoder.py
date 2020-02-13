from data import ts
from ..lib.ops import *

import numpy as np
import tensorflow as tf
tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True

Feature_Types_Layers = ['h0', 'h1', 'h2', 'h3']
Feature_Types = ['flat', 'gap', 'gmp', 'local-avg', 'local-max', 'local-max-abs', 'local-max-int',
                 'local-max-stride']
Normalization_Types = [None, 'znorm', 'sigmoid', 'tanh']
class TSGANEncoder:
    def __init__(self, gan, input_shape, is_training=False, type='local-max'):
        self.gan = gan
        self.input_shape = input_shape
        self.is_training = is_training
        self.x = tf.placeholder(tf.float32, shape=[None] + self.input_shape, name='x')
        self.layers = self.gan.discriminator_layers(self.x, self.is_training)
        self.features = self.get_features(type)

    def encode(self, sess, X, batch_size, norm=None):
        res = []
        n_samples = len(X)
        n_batches = n_samples // batch_size
        for i in range(n_samples // batch_size):
            x_batch = X[i * batch_size:(i + 1) * batch_size]
            feature_batch = sess.run(self.features, feed_dict={self.x: x_batch})
            res.append(feature_batch)
        n_samples_left = n_samples - n_batches * batch_size
        if n_samples_left > 0:
            x_left = X[-n_samples_left:]
            feature_left = sess.run(self.features, feed_dict={self.x: x_left})
            res.append(feature_left)
        res = np.concatenate(res, axis=0)
        if norm is None:
            return res
        else:
            return self.normalize(res, norm)

    def get_features(self, type='h0'):
        ## flat feature map from specific layer
        if type.startswith('h'):
            index = int(type[1:])
            features = flatten(self.layers[index])
        ## flat and concat
        elif type == 'flat':
            features = self.get_features_flat()
        ## flat, concat, then pool
        elif type == 'local-max':
            features = self.get_features_local_max()
        elif type == 'local-max-abs':
            features = self.get_features_local_max_abs()
        elif type == 'local-max-int':
            features = self.get_features_local_max_int()
        elif type == 'local-max-stride':
            features = self.get_features_local_max_stride()
        elif type == 'local-avg':
            features = self.get_features_local_avg()
        elif type == 'gap':
            features = self.get_features_gap()
        elif type == 'gmp':
            features = self.get_features_gmp()
        else:
            raise ValueError('The type of features, {}, no found!'.format(type))

        return features

    def get_features_flat(self):
        h0, h1, h2, h3 = self.layers
        return tf.concat([flatten(h1), flatten(h2), flatten(h3)], axis=1)

    def get_features_local_max(self):
        _, h1, h2, _ = self.layers

        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value

        f1 = flatten(tf.nn.max_pool(h1, [1, dim_1 // 3, 1, 1], [1, 1, 1, 1], 'SAME'))
        f2 = flatten(tf.nn.max_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'SAME'))

        return tf.concat([f1, f2], axis=1)

    def get_features_local_max_detail(self):
        _, h1, h2, _ = self.layers
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value

        maxp1, maxpid1 = tf.nn.max_pool_with_argmax(h1, [1, dim_1 // 3, 1, 1], [1, 1, 1, 1], 'SAME')
        maxp2, maxpid2 = tf.nn.max_pool_with_argmax(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'SAME')

        f1 = flatten(maxp1)
        f2 = flatten(maxp2)

        return [h1, maxp1, maxpid1, f1], [h2, maxp2, maxpid2, f2]

    def get_features_local_max_abs(self):
        h0, h1, h2, h3 = self.layers
        dim_0 = h0.get_shape()[1].value
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value
        dim_3 = h3.get_shape()[1].value

        h1, h2, h3 = tf.abs(h1), tf.abs(h2), tf.abs(h3)
        f1 = flatten(tf.nn.max_pool(h1, [1, dim_1 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        f2 = flatten(tf.nn.max_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        f3 = flatten(tf.nn.max_pool(h3, [1, max(2, dim_3 // 4), 1, 1], [1, 1, 1, 1], 'VALID'))

        return tf.concat([f1, f2, f3], axis=1)

    def get_features_local_max_int(self):
        h0, h1, h2, h3 = self.layers
        dim_0 = h0.get_shape()[1].value
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value
        dim_3 = h3.get_shape()[1].value

        h1, h2, h3 = tf.cast(h1, tf.int32), tf.cast(h2, tf.int32), tf.cast(h3, tf.int32)
        f1 = flatten(tf.nn.max_pool(h1, [1, dim_1 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        f2 = flatten(tf.nn.max_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        f3 = flatten(tf.nn.max_pool(h3, [1, max(2, dim_3 // 4), 1, 1], [1, 1, 1, 1], 'VALID'))

        return tf.concat([f1, f2, f3], axis=1)

    def get_features_local_max_stride(self):
        h0, h1, h2, h3 = self.layers
        dim_0 = h0.get_shape()[1].value
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value
        dim_3 = h3.get_shape()[1].value
        w1 = dim_1 // 2
        w2 = dim_2 // 3
        w3 = max(2, dim_3 // 4)
        f1 = flatten(tf.nn.max_pool(h1, [1, w1, 1, 1], [1, min(5, w1), 1, 1], 'VALID'))
        f2 = flatten(tf.nn.max_pool(h2, [1, w2, 1, 1], [1, min(5, w2), 1, 1], 'VALID'))
        f3 = flatten(tf.nn.max_pool(h3, [1, w3, 1, 1], [1, min(5, w3), 1, 1], 'VALID'))

        return tf.concat([f1, f2, f3], axis=1)

    def get_features_local_avg(self):
        h0, h1, h2, h3 = self.layers
        dim_0 = h0.get_shape()[1].value
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value
        dim_3 = h3.get_shape()[1].value
        f1 = flatten(tf.nn.avg_pool(h1, [1, dim_1 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        f2 = flatten(tf.nn.avg_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        f3 = flatten(tf.nn.avg_pool(h3, [1, max(2, dim_3 // 4), 1, 1], [1, 1, 1, 1], 'VALID'))

        return tf.concat([f1, f2, f3], axis=1)

    def get_features_gap(self):
        h0, h1, h2, h3 = self.layers
        f1 = flatten(tf.layers.average_pooling2d(
            h1, pool_size=[h1.get_shape()[1].value, h1.get_shape()[2].value], strides=1))
        f2 = flatten(tf.layers.average_pooling2d(
            h2, pool_size=[h2.get_shape()[1].value, h2.get_shape()[2].value], strides=1))
        f3 = flatten(tf.layers.average_pooling2d(
            h3, pool_size=[h3.get_shape()[1].value, h3.get_shape()[2].value], strides=1))
        flat = tf.concat([f1, f2, f3], axis=1)
        return flat

    def get_features_gmp(self):
        h0, h1, h2, h3 = self.layers
        f1 = flatten(tf.layers.max_pooling2d(
            h1, pool_size=[h1.get_shape()[1].value, h1.get_shape()[2].value], strides=1))
        f2 = flatten(tf.layers.max_pooling2d(
            h2, pool_size=[h2.get_shape()[1].value, h2.get_shape()[2].value], strides=1))
        f3 = flatten(tf.layers.max_pooling2d(
            h3, pool_size=[h3.get_shape()[1].value, h3.get_shape()[2].value], strides=1))
        flat = tf.concat([f1, f2, f3], axis=1)
        return flat

    def normalize(self, features, mode='znorm'):
        # common-used: znorm, sigmoid, thanh
        return ts.normalize(features, mode)





