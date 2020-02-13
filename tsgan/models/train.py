from ..lib.utils_data import DataSet
from ..lib import vis
from ..lib import metrics
from ..lib.mmd import mix_rbf_mmd2
from ..lib import utils_gan
from ..lib.io import *

from . import model

from time import time
import numpy as np
from sklearn.metrics import accuracy_score

tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True

def get_acc(d_real_prob, d_fake_prob):
    real_prob = np.reshape(d_real_prob, (d_real_prob.shape[0],))
    fake_prob = np.reshape(d_fake_prob, (d_fake_prob.shape[0],))
    alpha = 0.5
    y_real = np.ones(len(real_prob))
    y_fake = np.zeros(len(fake_prob))

    y = np.concatenate([y_real, y_fake])
    y_pred_prob = np.concatenate([real_prob, fake_prob])
    y_pred = np.zeros(y_pred_prob.shape[0])
    for i, p in enumerate(y_pred_prob):
        if p >= alpha:
            y_pred[i] = 1
    acc = accuracy_score(y, y_pred)
    return acc

def train(conf):
    gan = model.TSGAN(
        conf.dim_z, conf.dim_h, conf.dim_w, conf.dim_c, conf.random_seed,
        conf.g_lr, conf.d_lr, conf.g_beta1, conf.d_beta1,
        conf.gf_dim, conf.df_dim
    )
    dataset = DataSet(conf.X, seed=conf.random_seed)

    # log ground truth
    vis_n_samples = min(6, conf.batch_size)
    vis_X = conf.X[:vis_n_samples]
    vis_X = vis_X.reshape([vis_X.shape[0], -1])
    vis.plot_series(vis_X, os.path.join(conf.dir_samples, "000_real.png"))
    # save variables to log
    f_log_train = open(os.path.join(conf.dir_logs,'log_train.json'), 'w')
    log_fields = [
        'n_epoches', 'n_updates', 'n_examples', 'n_seconds',
        'nnd', 'mmd', 'g_loss', 'd_loss_real', 'd_loss_fake'
    ]

    # set up tf session and train model
    with tf.Session(config=tf_conf) as sess:
        # initialize
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        n_updates = 0
        n_epochs = 0
        n_examples = 0
        g_losses, d_losses, d_losses_fake, d_losses_real = [], [], [], []
        d_real_prob_list, d_fake_prob_list, acc_list  = [], [], []
        ts_train = []
        nnds = []
        mmds = []
        mmd_bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        mmd_batchsize = min(conf.n_samples, conf.X.shape[0])
        mmd_real_t = tf.placeholder(tf.float32, [mmd_batchsize, conf.dim_h], name='mmd_real')
        mmd_sample_t = tf.placeholder(tf.float32, [mmd_batchsize, conf.dim_h], name='mmd_sample')
        mmd_loss_t = mix_rbf_mmd2(mmd_real_t, mmd_sample_t, sigmas=mmd_bandwidths)
        # train
        t_start = time()
        acc_last = 0.5
        for epoch in xrange(conf.n_epochs):
            g_loss, d_loss, d_loss_fake, d_loss_real = np.zeros(4)
            t_start_epoch = time()
            for i in xrange(dataset.n_samples // conf.batch_size):
                x = dataset.next_batch(conf.batch_size)
                z = gan.sampler_noise(conf.batch_size)

                if acc_last <= conf.acc_threshold:
                    _ = sess.run(gan.d_opt, feed_dict={gan.x: x, gan.z: z})
                _ = sess.run(gan.g_opt, feed_dict={gan.z: z})

                d_real_prob, d_fake_prob = sess.run([gan.d_real_prob, gan.d_fake_prob],
                                                    feed_dict={gan.x: x, gan.z: z})
                d_real_prob_mean, d_fake_prob_mean = np.mean(d_real_prob), np.mean(d_fake_prob)
                acc = get_acc(d_real_prob, d_fake_prob)
                acc_last = acc
                d_loss, d_loss_real, d_loss_fake, g_loss = sess.run(
                    [gan.d_loss, gan.d_loss_real, gan.d_loss_fake, gan.g_loss],
                    feed_dict={gan.x: x, gan.z: z})
                n_updates += 1
                n_examples += len(x)
            n_epochs += 1
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            d_losses_fake.append(d_loss_fake)
            d_losses_real.append(d_loss_real)
            d_real_prob_list.append(d_real_prob_mean)
            d_fake_prob_list.append(d_fake_prob_mean)
            acc_list.append(acc)
            ts_train.append(time() - t_start_epoch)

            # log
            print("Epoch: [{}/{}], "
                  "g_loss = {:.4f}, d_loss = {:.4f}, d_loss_fake = {:.4f}, d_loss_reak = {:.4f},"
                  "d_real_prob = {:.4f}, d_fake_prob = {:.4f}, acc = {:.4f}".
                format(epoch, conf.n_epochs, g_loss, d_loss, d_loss_fake, d_loss_real,
                       d_real_prob_mean, d_fake_prob_mean, acc))
            if epoch % conf.freq_log == 0 or epoch == conf.n_epochs-1:
                # eval
                gX = utils_gan.generate_samples(sess, gan, conf, conf.n_samples)
                gX = gX.reshape(len(gX), -1)
                teX = conf.X.reshape(len(conf.X), -1)
                nnd_ = metrics.nnd_score(gX[:mmd_batchsize], teX[:mmd_batchsize], metric='euclidean')
                nnds.append(nnd_)
                mmd_ = sess.run(mmd_loss_t,
                                feed_dict={mmd_real_t: teX[:mmd_batchsize],
                                           mmd_sample_t: gX[:mmd_batchsize]})
                mmds.append(mmd_)
                log_valus = [n_epochs, n_updates, n_examples, time()-t_start,
                             nnd_, float(mmd_), float(g_loss), float(d_loss_real), float(d_loss_fake)]
                f_log_train.write(json.dumps(dict(zip(log_fields, log_valus))) + '\n')
                f_log_train.flush()
                # save checkpoint
                gan.save(sess, conf.dir_checkpoint, n_updates)

            if epoch % conf.freq_plot == 0  or epoch == conf.n_epochs - 1:
                samples = utils_gan.generate_samples(sess, gan, conf, vis_n_samples)
                samples = samples.reshape([samples.shape[0],-1])
                img_path = os.path.join(conf.dir_samples, "train_{}.png".format(str(epoch+1).zfill(4)))
                txt_path = os.path.join(conf.dir_samples, "train_{}".format(str(epoch+1).zfill(4)))
                vis.plot_series(samples, img_path)
                np.savetxt(txt_path, samples, delimiter=',', newline='\n')

    metrics_dic = {
        'g_loss': np.array(g_losses),
        'd_loss': np.array(d_losses),
        'd_loss_fake': np.array(d_losses_fake),
        'd_loss_real': np.array(d_losses_real),
        'd_prob_real': np.array(d_real_prob_list),
        'd_prob_fake': np.array(d_fake_prob_list),
        'd_acc':np.array(acc_list),
        'nnd': np.array(nnds),
        'mmd': np.array(mmds),
        'time': np.array(ts_train)
    }
    metrics_save(metrics_dic, conf)
    metrics_vis(metrics_dic, conf)

    res = {
        'g_loss': g_losses[-1],
        'd_loss': d_losses[-1],
        'd_loss_fake': d_losses_fake[-1],
        'd_loss_real': d_losses_real[-1],
        'nnd': nnds[-1],
        'mmd': mmds[-1],
        'time': np.sum(ts_train)
    }
    return res