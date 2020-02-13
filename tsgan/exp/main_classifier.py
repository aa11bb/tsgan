from tsgan.exp.base import tag_path
from tsgan.exp import base
from tsgan.exp.main_encoder import encode
from data import ucr

import pandas as pd
import os

def make_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

"""
    Standard Baselines on raw data
"""
def run_baselines(dir_data, data_name_list, dir_out_root):
    dir_out = os.path.join(dir_out_root, 'raw')
    n_runs = 3
    for ir in range(n_runs):
        dir_out_cur = os.path.join(dir_out, 'run{}'.format(ir))
        make_dir(dir_out_cur)
        for model_name, model in base.StandardClassifierDic.items():
            print("******** processing model {}".format(model_name))
            res = {'dataset': [], 'acc': [], 'acc_te': [], 'time': [], 'time_te': []}
            for data_name in data_name_list:
                x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
                acc, t = base.classify(model, x_tr, y_tr, x_te, y_te)
                res['dataset'].append(data_name)
                res['acc'].append(acc[0])
                res['acc_te'].append(acc[1])
                res['time'].append(t[0])
                res['time_te'].append(t[1])
                print(model_name, data_name, acc, t)
            df = pd.DataFrame(res)
            df.to_csv(os.path.join(dir_out_cur, '{}.csv'.format(model_name)), index=False)

"""
    TSGAN-encoder based classifiers
"""

def run_classifier(model, model_name, feature_type, norm_type, data_name_list, dir_gan, dir_data, dir_out):
    print("******** processing {}, {}".format(model_name, feature_type, norm_type))
    res = {'dataset': [], 'acc': [], 'acc_te': [], 'time': [], 'time_te': []}
    n_datasets = len(data_name_list)
    for j, data_name in enumerate(data_name_list):
        print("******** [{}/{}] processing {}".format(j, n_datasets, data_name))
        ## load data
        features_tr, y_tr, features_te, y_te, n_classes = encode(
            data_name, dir_data, feature_type, norm_type, dir_gan)
        acc, t = base.classify(model, features_tr, y_tr, features_te, y_te)
        res['dataset'].append(data_name)
        res['acc'].append(acc[0])
        res['acc_te'].append(acc[1])
        res['time'].append(t[0])
        res['time_te'].append(t[1])
        print(acc, t)
    ## save result
    df = pd.DataFrame(res)
    df.to_csv(
        os.path.join(dir_out, '{}_{}_{}.csv'.format(model_name, feature_type, norm_type)),
        index=False)
    return df

def run_standard_classifier(
        model_name, feature_type, norm_type, data_name_list, dir_gan, dir_data, dir_out):
    model = base.StandardClassifierDic[model_name]
    return run_classifier(model, model_name, feature_type, norm_type, data_name_list, dir_gan, dir_data, dir_out)

def run_tsgan(model_name, dir_out, data_name_list):
    path_gan_list = ['cache/main_gan_half']
    feature_type = 'local-max'
    norm_type = 'tanh'
    for path_gan in path_gan_list:
        path_out = os.path.join(dir_out, os.path.basename(path_gan))
        make_dir(path_out)
        run_standard_classifier(
            model_name, feature_type, norm_type, data_name_list, path_gan, dir_data, path_out)


if __name__ == '__main__':
    dir_data = base.UCR_DIR
    tag = tag_path(os.path.abspath(__file__), 1)
    dir_out = 'cache/{}'.format(tag)

    # data_name_list = ucr.get_data_name_list(dir_data)
    data_name_list = ['ArrowHead']

    #run_baselines(dir_data, data_name_list, dir_out)
    # run_tsgan('LR', dir_out, data_name_list)
    run_tsgan('LSVC', dir_out, data_name_list)
    #run_tsgan('1NN', dir_out, data_name_list)