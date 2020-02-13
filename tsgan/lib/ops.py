
from .normalizations import GroupNormalization, InstanceNormalization, \
    LayerNormalization, BatchNormalization

import math

import tensorflow as tf

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

flatten = tf.contrib.layers.flatten
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
relu = tf.nn.relu


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

class Normalization(object):
    def __init__(self, name='layer_norm', mode='layer_norm', **kwargs):
        with tf.variable_scope('{}_{}'.format(name, mode)):
            self.name = name
            self.mode = mode
            if mode == 'layer_norm':
                self.normalizer = LayerNormalization(**kwargs)
            elif mode == 'instance_norm':
                self.normalizer = InstanceNormalization(**kwargs)
            elif mode == 'group_norm':
                self.normalizer = GroupNormalization(**kwargs)
            elif mode == 'batch_norm':
                ## do not forget to add update_ops dependencies manually.
                self.normalizer = BatchNormalization(momentum=0.9, epsilon=1e-5, **kwargs)
            else:
                raise ValueError("The normalization name {} can not be found!".format(name))
    def __call__(self, x, is_training=True):
        return self.normalizer(x, training=is_training)


class BatchNorm(object):
    """
    Realization based on tf.contrib.layers.batch_norm
    ------ References
    https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        Note: when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op. For example:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
              train_op = optimizer.minimize(loss)
        One can set updates_collections=None to force the updates in place, but that can have a speed penalty, especially in distributed settings.
        updates_collections: Collections to collect the update ops for computation. The updates_ops need to be executed with the train_op. If None, a control dependency would be added to make sure the updates are computed in place.
    https://towardsdatascience.com/pitfalls-of-batch-norm-in-tensorflow-and-sanity-checks-for-training-networks-e86c207548c8
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, is_training=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=is_training,
                                            scope=self.name)


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])],3)

def conv1d(input_, output_dim,
           k_h=1, k_w=5, d_h=1, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(
                                stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv, biases)
        return conv

def conv2d(input_, output_dim,
           k_h=10, k_w=10, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(
                                stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv, biases)
        return conv

def deconv1d(input_, output_shape,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d", with_w=False):
    in_c = input_.get_shape()[-1]
    out_c = output_shape[-1]
    init_w = tf.random_normal_initializer(stddev=stddev)
    init_b = tf.constant_initializer(0.0)
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, out_c, in_c], initializer=init_w)
        deconv = tf.nn.conv2d_transpose(input_, w,
                                        output_shape=tf.stack(output_shape),
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [out_c], initializer=init_b)
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def deconv2d(input_, output_shape,
             k_h=10, k_w=10, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    in_c = input_.get_shape()[-1]
    out_c = output_shape[-1]
    init_w = tf.random_normal_initializer(stddev=stddev)
    init_b = tf.constant_initializer(0.0)
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, out_c, in_c], initializer=init_w)
        deconv = tf.nn.conv2d_transpose(input_, w,
                                        output_shape=tf.stack(output_shape),
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [out_c], initializer=init_b)
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0,
           with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
