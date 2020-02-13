#
# The util script to understand, visualize, and reconstruct UCR data set.
#
#
# ==============================================================================
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

from data import ucr
from data import utils


# ==============================================================================
#  get data name list

def get_data_name_list(data_dir_root):
    """
    
    :param data_dir_root: 
    :return: 
    """
    data_name_list = [f for f in os.listdir(data_dir_root)
                      if os.path.isdir(os.path.join(data_dir_root, f))]
    return np.sort(data_name_list)


def get_dataset_testset_larger_than_trainset(data_dir_root):
    """

    :param data_dir_root: 
    :return: 
    (2018-12-15, when data_dir_root=UCR_TS_Archive_2015)
        Number of data set: 79
        These data set are: 
        ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 
         'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X', 
         'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 
         'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 
         'Earthquakes', 'FISH', 'FaceAll', 'FaceFour', 'FacesUCR', 'FordA', 'FordB', 'Gun_Point', 
         'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 
         'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages', 
         'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 
         'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 
         'OSULeaf', 'OliveOil', 'Phoneme', 'Plane', 'ProximalPhalanxTW', 'RefrigerationDevices', 
         'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface', 
         'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 
         'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Two_Patterns', 
         'UWaveGestureLibraryAll', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'synthetic_control', 
         'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga']
    """
    data_name_list = get_data_name_list(data_dir_root)
    res = []
    for fname in data_name_list:
        datasets = ucr.load_ucr(fname, data_dir_root)
        if datasets.test.X.shape[0] >= datasets.train.X.shape[0]:
            res.append(fname)
    return res


def get_dataset_testset_double_than_trainset(data_dir_root):
    """

    :param data_dir_root: 
    :return:  
    (2018-12-15, when data_dir_root=UCR_TS_Archive_2015)
        Number of data set: 45
        These data set are:
        ['ArrowHead', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'DiatomSizeReduction', 
         'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 
         'ECG5000', 'ECGFiveDays', 'Earthquakes', 'FaceAll', 'FaceFour', 'FacesUCR', 'FordA', 'FordB', 
         'Gun_Point', 'HandOutlines', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MALLAT', 
         'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 
         'Phoneme', 'ShapeletSim', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 
         'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'TwoLeadECG', 'Two_Patterns', 
         'UWaveGestureLibraryAll', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'uWaveGestureLibrary_X', 
         'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga']
    """
    data_name_list = get_data_name_list(data_dir_root)
    res = []
    for fname in data_name_list:
        datasets = ucr.load_ucr(fname, data_dir_root)
        if datasets.test.X.shape[0] >= 2 * datasets.train.X.shape[0]:
            res.append(fname)
    return res


def get_dataset_testset_double_than_trainset_for_each_class(data_dir_root):
    """

    :param data_dir_root: 
    :return: 
    (2018-12-15, when data_dir_root=UCR_TS_Archive_2015)
        Number of data set: 33
        These data set are:
        ['ArrowHead', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'DiatomSizeReduction', 'ECG5000', 
         'ECGFiveDays', 'FacesUCR', 'FordA', 'FordB', 'Gun_Point', 'HandOutlines', 'InlineSkate', 
         'InsectWingbeatSound', 'ItalyPowerDemand', 'MALLAT', 'MoteStrain', 'ShapeletSim', 
         'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Symbols', 
         'ToeSegmentation1', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibraryAll', 'Worms', 
         'WormsTwoClass', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 
         'wafer', 'yoga']
    """
    data_name_list = get_data_name_list(data_dir_root)
    res = []
    for fname in data_name_list:
        datasets = ucr.load_ucr(fname, data_dir_root)
        distr_train = utils.distribute_y(datasets.train.y)
        distr_test = utils.distribute_y(datasets.test.y)
        is_pass = True
        for key_tr in distr_train.keys():
            num_tr = distr_train[key_tr]
            num_te = distr_test[key_tr]
            if num_te is None or num_te < 2 * num_tr:
                is_pass = False
                break
        if is_pass:
            res.append(fname)
    return res

# ==============================================================================
#

def category_dataset_exact_length(dir_data):
    data_categories_length = {}
    fname_list = ucr.get_data_name_list(dir_data)
    for fname in fname_list:
        data = ucr.load_ucr(fname, dir_data)
        length = data.train.X.shape[1]
        if (length in data_categories_length.keys()) is False:
            data_categories_length[length] = []
        data_categories_length[length].append(fname)
    for key in sorted(data_categories_length.keys()):
        print(key, data_categories_length[key])
    return data_categories_length

# ==============================================================================
#  category dataset get data name list
#  reference: http://timeseriesclassification.com/dataset.php
def category_dataset_trainsize(data_dir_root):
    name_list = get_data_name_list(data_dir_root)

    names_0_to_50 = []
    names_51_to_100 = []
    names_101_to_500 = []
    names_greater_500 = []
    for fname in name_list:
        dataset = ucr.load_ucr(fname, data_dir_root)
        n = dataset.train.X.shape[0]
        if n <= 50:
            names_0_to_50.append(fname)
        elif n <= 100:
            names_51_to_100.append(fname)
        elif n <= 500:
            names_101_to_500.append(fname)
        else:
            names_greater_500.append(fname)
    print("===== Category dataset by train size: ")
    print("form 0 to 50: ", len(names_0_to_50), names_0_to_50)
    print("from 51 to 100: ", len(names_51_to_100), names_51_to_100)
    print("from 101 to 500: ", len(names_101_to_500), names_101_to_500)
    print("greater than 500: ", len(names_greater_500), names_greater_500)
    print()


def category_dataset_testsize(data_dir_root):
    name_list = get_data_name_list(data_dir_root)

    names_0_to_300 = []
    names_301_to_1000 = []
    names_greater_1000 = []
    for fname in name_list:
        dataset = ucr.load_ucr(fname, data_dir_root)
        n = dataset.test.X.shape[0]
        if n <= 300:
            names_0_to_300.append(fname)
        elif n <= 1000:
            names_301_to_1000.append(fname)
        else:
            names_greater_1000.append(fname)
    print("===== Category dataset by test size: ")
    print("from 0 to 300: ", len(names_0_to_300), names_0_to_300)
    print("from 301 to 1000: ", len(names_301_to_1000), names_301_to_1000)
    print("greater than 1000: ", len(names_greater_1000), names_greater_1000)
    print()


def category_dataset_length(data_dir_root):
    name_list = get_data_name_list(data_dir_root)

    names_0_to_300 = []
    names_301_to_700 = []
    names_greater_700 = []
    for fname in name_list:
        dataset = ucr.load_ucr(fname, data_dir_root)
        length = dataset.train.X.shape[1]
        if length <= 300:
            names_0_to_300.append(fname)
        elif length <= 700:
            names_301_to_700.append(fname)
        else:
            names_greater_700.append(fname)
    print("===== Category dataset by length: ")
    print("from 0 to 300: ", len(names_0_to_300), names_0_to_300)
    print("from 301 to 700: ", len(names_301_to_700), names_301_to_700)
    print("greater than 700: ", len(names_greater_700), names_greater_700)
    print()


def category_dataset_nclass(data_dir_root):
    name_list = get_data_name_list(data_dir_root)

    names_0_to_10 = []
    names_11_to_30 = []
    names_greater_30 = []
    for fname in name_list:
        dataset = ucr.load_ucr(fname, data_dir_root)
        n_class = dataset.nclass
        if n_class <= 10:
            names_0_to_10.append(fname)
        elif n_class <= 30:
            names_11_to_30.append(fname)
        else:
            names_greater_30.append(fname)
    print("===== Category dataset by the number of class: ")
    print("from 0 to 10: ", len(names_0_to_10), names_0_to_10)
    print("from 11 to 30: ", len(names_11_to_30), names_11_to_30)
    print("greater than 30: ", len(names_greater_30), names_greater_30)
    print()


# ==============================================================================
#  understand data set by some statistics information and visualization

def get_data_info(data_dir_root, data_name_list=None, out_csv='./dataset_info.csv'):
    """

    :param data_dir_root:   
    :param data_name_list: 
    :param out_csv: 
    :return: 
    """
    if data_name_list is None:
        data_name_list = get_data_name_list(data_dir_root)

    res_col = ['DataSet', 'NClass', 'SequenceLength',
               'SizeAll', 'SizeTrain', 'SizeTest', 'SizeValid',
               'DistributionAll', 'DistributionTrain', 'DistributionTest', 'DistributionValid']
    res_df = pd.DataFrame(columns=res_col)
    for i, fname in enumerate(data_name_list):
        print("preprocessing dataset: {}".format(fname))
        # load data
        data = ucr.load_ucr(fname, data_dir_root)
        if data.valid is None:
            X_all = np.concatenate([data.train.X, data.test.X], axis=0)
            y_all = np.concatenate([data.train.y, data.test.y])
        else:
            X_all = np.concatenate([data.train.X, data.test.X, data.valid.X], axis=0)
            y_all = np.concatenate([data.train.y, data.test.y, data.valid.y])
        # get the information of specific data set
        res_df.loc[i, 'DataSet'] = fname
        res_df.loc[i, 'NClass'] = len(np.unique(y_all))
        res_df.loc[i, 'SequenceLength'] = X_all.shape[1]
        res_df.loc[i, 'SizeAll'] = X_all.shape[0]
        res_df.loc[i, 'SizeTrain'] = data.train.X.shape[0]
        res_df.loc[i, 'SizeTest'] = data.test.X.shape[0]
        if data.valid is not None:
            res_df.loc[i, 'SizeValid'] = data.valid.X.shape[0]
        res_df.loc[i, 'DistributionAll'] = utils.distribute_y_json(y_all)
        res_df.loc[i, 'DistributionTrain'] = utils.distribute_y_json(data.train.y)
        res_df.loc[i, 'DistributionTest'] = utils.distribute_y_json(data.test.y)
        if data.valid is not None:
            res_df.loc[i, 'DistributionValid'] = utils.distribute_y_json(data.valid.y)

    res_df.to_csv(out_csv)


def plot_data_by_class(data_dict, out_dir):
    """
    
    :param data_dict: 
    :param out_dir: 
    :return: 
    """
    # make directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # plot and save figure
    for key in data_dict.keys():
        samples = data_dict[key]
        n_sample = samples.shape[0]
        title = 'class-{}_nsample-{}'.format(key, n_sample)
        fname = title + '.png'
        n_plot = min(n_sample, 4)
        samples_plot = samples[:n_plot]
        f, axes = plt.subplots(n_plot // 2, 2)
        axes = axes.flat[:]
        f.suptitle(title)
        for i, ax in enumerate(axes):
            if i >= samples_plot.shape[0]:
                break
            ax.plot(samples_plot[i])
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def plot_data_by_class_batch(data_dir_root, data_name_list=None, out_dir='./cache/vis'):
    """
    
    :param data_dir_root: 
    :param data_name_list: 
    :param out_dir: 
    :return: 
    """
    if data_name_list is None:
        data_name_list = get_data_name_list(data_dir_root)

    for fname in data_name_list:
        print("preprocessing dataset: {}".format(fname))
        X_all, y_all = ucr.load_ucr_concat(fname, data_dir_root)
        data_dict = utils.distribute_dataset(X_all, y_all)
        out_dir_current = os.path.join(out_dir, fname)
        plot_data_by_class(data_dict, out_dir_current)



# ==============================================================================
#  reconstruct data set

def z_normalize(in_data_dir_root, out_data_dir_root, data_name_list=None):
    """
    
    :param in_data_dir_root: 
    :param out_data_dir_root: 
    :param data_name_list: 
    :return: 
    """
    if data_name_list is None:
        data_name_list = get_data_name_list(in_data_dir_root)

    for fname in data_name_list:
        X_train, y_train, X_test, y_test = ucr.load_ucr(fname, in_data_dir_root)

        X_train_norm = utils.z_normalize(X_train)
        X_test_norm = utils.z_normalize(X_test)

        data_train = np.hstack([y_train[:, np.newaxis], X_train_norm])
        data_test = np.hstack([y_test[:, np.newaxis], X_test_norm])

        out_path = os.path.join(out_data_dir_root, fname)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
        np.savetxt(os.path.join(out_path, "{}_TRAIN".format(fname)), data_train, delimiter=',', newline="\n")
        np.savetxt(os.path.join(out_path, "{}_TEST").format(fname), data_test, delimiter=',', newline="\n")

        print("Finish to process dataset {}".format(fname), data_train.shape, data_test.shape)

def split_test2valid_stratified(
        in_data_dir_root, out_dir_parent,
        out_dir_base='UCR_TS_Archive_2015_split-test-to-valid-stratified_valid-size-same-as-train'):
    """ Split the original test set to generate a validation set and the number samples of 
        each class in validation set is the same as the train set. 
    
    :param in_data_dir_root: this path shouldn't include Validation set.
    :param out_dir_parent: 
    :param out_dir_base:
    :param data_name_list: 
    :return: 
    """
    out_data_dir_root = os.path.join(out_dir_parent, out_dir_base)
    data_name_list = get_dataset_testset_double_than_trainset_for_each_class(in_data_dir_root)
    print("This program will process {} data sets.".format(len(data_name_list)))
    print("They are: ", data_name_list)
    print()
    for fname in data_name_list:
        print("processing data {}".format(fname))
        # prepare data
        X_train, y_train, X_test, y_test = ucr.load_ucr_flat(fname, in_data_dir_root)
        distr_train = utils.distribute_dataset(X_train, y_train)
        distr_test = utils.distribute_dataset(X_test, y_test)

        # split test set to validation set
        X_valid, y_valid = [], []
        X_test_new, y_test_new = [], []
        for key in distr_train.keys():
            num_tr = distr_train[key].shape[0]
            inds_te = np.arange(distr_test[key].shape[0])
            np.random.shuffle(inds_te)

            X_valid_temp = distr_test[key][inds_te[:num_tr]]
            X_valid.append(X_valid_temp)
            y_valid.append(np.ones([X_valid_temp.shape[0], 1], int)*key)

            X_test_new_temp = distr_test[key][inds_te[num_tr:]]
            X_test_new.append(X_test_new_temp)
            y_test_new.append(np.ones([X_test_new_temp.shape[0], 1], int)*key)
        X_valid = np.concatenate(X_valid, axis=0)
        y_valid = np.concatenate(y_valid, axis=0)
        X_test_new = np.concatenate(X_test_new, axis=0)
        y_test_new = np.concatenate(y_test_new, axis=0)

        # pack data set in UCR format
        trainset = np.concatenate([y_train[:,np.newaxis], X_train], axis=1)
        validset = np.concatenate([y_valid, X_valid], axis=1)
        testset = np.concatenate([y_test_new, X_test_new], axis=1)
        # make dir for specific data set
        out_path = os.path.join(out_data_dir_root, fname)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
        # output data to file
        str_fmt = '%path,' + '%.4f,' * (X_train.shape[1])
        str_fmt = str_fmt[:(len(str_fmt) - 1)]
        np.savetxt(os.path.join(out_path, '{}_TRAIN'.format(fname)), trainset, fmt=str_fmt, delimiter=',')
        np.savetxt(os.path.join(out_path, '{}_TEST'.format(fname)), testset, fmt=str_fmt, delimiter=',')
        np.savetxt(os.path.join(out_path, '{}_VALID'.format(fname)), validset, fmt=str_fmt,delimiter=',')


DATA_ROOT_ORIGIN = '../../dataset/UCR_TS_Archive_2015'

def run_list(): # copy any statement to main function to run
    get_data_name_list(DATA_ROOT_ORIGIN)
    get_dataset_testset_larger_than_trainset(DATA_ROOT_ORIGIN)
    get_dataset_testset_double_than_trainset(DATA_ROOT_ORIGIN)
    get_dataset_testset_double_than_trainset_for_each_class(DATA_ROOT_ORIGIN)

    get_data_info(DATA_ROOT_ORIGIN, get_data_name_list(DATA_ROOT_ORIGIN))

    plot_data_by_class_batch(DATA_ROOT_ORIGIN)

    z_normalize(DATA_ROOT_ORIGIN, '../../dataset/UCR_TS_Archive_2015_norm')
    split_test2valid_stratified(DATA_ROOT_ORIGIN, '../../dataset')


if __name__ == '__main__':
    category_dataset_nclass(DATA_ROOT_ORIGIN)


