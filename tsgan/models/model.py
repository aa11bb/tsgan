from ..lib.ops import *
import os
from numpy.random import RandomState

def optimizer(loss, var_list, update_ops=None, lr=0.001, beta1=0.9):
    step = tf.Variable(0, trainable=False)
    with tf.variable_scope('optimizer'):
        if update_ops is None:
            opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)\
                .minimize(loss=loss, var_list=var_list, global_step=step)
        else:
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)\
                .minimize(loss=loss, var_list=var_list, global_step=step)
        return opt


class TSGAN(object):
    def __init__(self,
                 dim_z, dim_h, dim_w, dim_c, random_seed,
                 g_lr, d_lr, g_beta1, d_beta1, gf_dim=64, df_dim=64):
        # initialize batch normalization
        norm_mode = 'batch_norm'
        self.g_bn0 = Normalization(name='g_norm0', mode=norm_mode)
        self.g_bn1 = Normalization(name='g_norm1', mode=norm_mode)
        self.g_bn2 = Normalization(name='g_norm2', mode=norm_mode)
        self.g_bn3 = Normalization(name='g_norm3', mode=norm_mode)
        self.d_bn1 = Normalization(name='d_norm1', mode=norm_mode)
        self.d_bn2 = Normalization(name='d_norm2', mode=norm_mode)
        self.d_bn3 = Normalization(name='d_norm3', mode=norm_mode)
        # initialize hyper-parameters
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_c = dim_c
        self.random = RandomState(random_seed)
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_beta1 = g_beta1
        self.d_beta1 = d_beta1
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # set placeholder
        self.z = tf.placeholder(tf.float32, shape=[None, self.dim_z], name='z')
        self.x = tf.placeholder(tf.float32, shape=[None, self.dim_h, self.dim_w, self.dim_c], name='x')

        self.g = self.generator(self.z, reuse=False)
        self.d_real_prob, self.d_real_logits = self.discriminator(self.x)
        self.d_fake_prob, self.d_fake_logits = self.discriminator(self.g, reuse=True)

        # calculate loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real_logits, labels=tf.ones_like(self.d_real_prob)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_fake_logits, labels=tf.zeros_like(self.d_fake_prob)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_fake_logits, labels=tf.ones_like(self.d_fake_prob)))

        # get trainable variables
        var_list = tf.trainable_variables()
        self.d_vars = [v for v in var_list if v.name.startswith('D/')]
        self.g_vars = [v for v in var_list if v.name.startswith('G/')]
        self.d_filters = [v for v in self.d_vars if '_conv/w' in v.name]
        self.g_filters = [v for v in self.g_vars if '_deconv/w' in v.name]

        # set optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.d_update_ops = [o for o in update_ops if o.name.startswith('D/')]
        self.g_update_ops = [o for o in update_ops if o.name.startswith('G/')]
        self.d_opt = optimizer(self.d_loss, self.d_vars, self.d_update_ops, self.d_lr, self.d_beta1)
        self.g_opt = optimizer(self.g_loss, self.g_vars, self.g_update_ops, self.g_lr, self.g_beta1)

        # other tensors
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sampler = self.generator(self.z, is_training=False, reuse=True)
        self.d_features = tf.concat(self.get_features_discriminator(), axis=1)

    def discriminator(self, x, is_training=True, reuse=False, with_layers=False):
        with tf.variable_scope('D', reuse=reuse):
            h0 = lrelu(conv2d(x, self.df_dim, name='h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='h1_conv'), is_training))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='h2_conv'), is_training))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='h3_conv'), is_training))
            flat = flatten(h3)
            logits = linear(flat, 1, 'logits')
            prob = sigmoid(logits)

            if with_layers:
                layers = [h0, h1, h2, h3]
                return prob, logits, layers
            else:
                return prob, logits

    def discriminator_layers(self, x, is_training=True, reuse=True, mode='last'):
        with tf.variable_scope('D', reuse=reuse):
            h0_conv = conv2d(x, self.df_dim, name='h0_conv')
            h0 = lrelu(h0_conv)
            h1_conv = conv2d(h0, self.df_dim * 2, name='h1_conv')
            h1_bn = self.d_bn1(h1_conv, is_training)
            h1 = lrelu(h1_bn)
            h2_conv = conv2d(h1, self.df_dim * 4, name='h2_conv')
            h2_bn = self.d_bn2(h2_conv, is_training)
            h2 = lrelu(h2_bn)
            h3_conv = conv2d(h2, self.df_dim * 8, name='h3_conv')
            h3_bn = self.d_bn3(h3_conv, is_training)
            h3 = lrelu(h3_bn)

            if mode == 'last':
                return [h0, h1, h2, h3]
            elif mode == 'conv':
                return [h0_conv, h1_conv, h2_conv, h3_conv]
            elif mode == 'norm':
                return [h0_conv, h1_bn, h2_bn, h3_bn]
            else:
                raise ValueError("Can not find the mode={} !".format(mode))

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            s_h, s_w = self.dim_h, self.dim_w
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w,2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            batch_size = tf.shape(z)[0]

            z_ = linear(z, self.gf_dim*8*s_h16*s_w16, 'h0_lin')
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = relu(self.g_bn0(h0, is_training))

            h1 = deconv2d(h0,[batch_size, s_h8, s_w8, self.gf_dim*4], name='h1_deconv')
            h1 = relu(self.g_bn1(h1, is_training))

            h2 = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim*2], name='h2_deconv')
            h2 = relu(self.g_bn2(h2, is_training))

            h3 = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim*1], name='h3_deconv')
            h3 = relu(self.g_bn3(h3, is_training))

            h4 = deconv2d(h3, [batch_size, s_h, s_w, self.dim_c], name='h4_deconv')
            #out = tf.nn.sigmoid(h4) # !! use sigmoid as final layer will lead to divergence.

            return h4

    def get_layers_discriminator(self, is_training):
        layers = self.discriminator_layers(self.x, is_training)
        return layers

    def get_features_discriminator(self, is_training=False):
        layers = self.get_layers_discriminator(is_training)
        h0, h1, h2, h3 = layers
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value
        dim_3 = h3.get_shape()[1].value

        f1 = flatten(tf.nn.max_pool(h1, [1, dim_1 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        f2 = flatten(tf.nn.max_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        f3 = flatten(tf.nn.max_pool(h3, [1, max(2, dim_3 // 4), 1, 1], [1, 1, 1, 1], 'VALID'))

        return [f1, f2, f3]

    def sampler_noise(self, batch_size):
        return self.random.uniform(-1, 1, [batch_size, self.dim_z])

    def save(self, sess, dir_save, step):
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        prefix = os.path.join(dir_save, "model.ckpt")
        self.saver.save(sess, prefix, global_step=step)

    def load(self, sess, dir_load):
        import re
        print("[*]  Reading checkpoints ...")
        ckpt = tf.train.get_checkpoint_state(dir_load)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(dir_load, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0




