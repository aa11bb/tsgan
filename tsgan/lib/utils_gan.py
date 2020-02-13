import numpy as np
from six.moves import xrange

def extract_features(X, sess, conf, gan):
    """extract feature from the discriminator of GAN"""
    features = []
    n_samples = len(X)
    n_map = 0
    for i in xrange(n_samples // conf.batch_size):
        x_batch = X[i * conf.batch_size:(i + 1) * conf.batch_size]
        f_batch = sess.run(gan.d_features, feed_dict={gan.x: x_batch})
        features.append(f_batch)
        n_map += len(f_batch)
    n_left = n_samples - n_map
    if n_left > 0:
        x_left = X[-n_left:]
        f_left = sess.run(gan.d_features, feed_dict={gan.x: x_left})
        features.append(f_left)
    features = np.concatenate(features, axis=0)
    return features


def generate_samples(sess, gan, conf, n_sample):
    """generate samples with generator of GAN"""
    samples = []
    n_gen = 0
    for i in xrange(n_sample // conf.batch_size):
        z = gan.sampler_noise(conf.batch_size)
        samples_batch = sess.run(gan.sampler, feed_dict={gan.z: z})
        samples.append(samples_batch)
        n_gen += len(samples_batch)
    n_left = n_sample - n_gen
    if n_left > 0:
        z = gan.sampler_noise(n_left)
        samples_left = sess.run(gan.sampler, feed_dict={gan.z: z})
        samples.append(samples_left)
    return np.concatenate(samples, axis=0)