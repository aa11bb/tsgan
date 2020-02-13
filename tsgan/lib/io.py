from . import utils

from six.moves import xrange
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt


def metrics_save(metrics_dic, conf):
    metrics_out = {}
    for key, values in metrics_dic.items():
        metrics_out[key] = ['{:4}'.format(v) for v in values]
    with open(os.path.join(conf.dir_logs, 'metrics.json'), 'w') as f:
        f.write(json.dumps(metrics_out))


def metrics_vis(metrics_dic, conf):
    plt.figure()
    plt.plot(metrics_dic['g_loss'], label='g_loss')
    plt.plot(metrics_dic['d_loss'], label='d_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(conf.dir_logs, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(metrics_dic['g_loss'], label='g_loss')
    plt.plot(metrics_dic['d_loss'], label='d_loss')
    plt.plot(metrics_dic['d_loss_fake'], label='d_loss_fake')
    plt.plot(metrics_dic['d_loss_real'], label='d_loss_real')
    plt.legend(loc='best')
    plt.savefig(os.path.join(conf.dir_logs, 'loss_all.png'))
    plt.close()

    plt.figure()
    plt.plot(xrange(0, metrics_dic['mmd'].shape[0] * conf.freq_log, conf.freq_log),
             metrics_dic['mmd'], '^-')
    plt.savefig(os.path.join(conf.dir_logs, 'mmd.png'))
    plt.close()

    plt.figure()
    plt.plot(xrange(0, metrics_dic['nnd'].shape[0] * conf.freq_log, conf.freq_log),
             metrics_dic['nnd'], '^-')
    plt.savefig(os.path.join(conf.dir_logs, 'nnd.png'))
    plt.close()


def save_variables(conf, save_path):
    """save the configure and model variables"""
    vars_conf = utils.analyze_object_variables(conf, print_info=True)
    vars_tensor = utils.analyze_tensor_variables(tf.trainable_variables(), print_info=True)
    vars_list = list()
    vars_list.append("=" * 80)
    vars_list.append("variables of configure")
    vars_list.append("-" * 80)
    vars_list.extend(vars_conf)
    vars_list.append("=" * 80)
    vars_list.append("=" * 80 + "\n")
    vars_list.append("variables of tensor")
    vars_list.append("-" * 80)
    vars_list.extend(vars_tensor)

    utils.save_variables_to_file(save_path, vars_list)