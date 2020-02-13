import os

""" Constant parameters
"""
UCR_DIR = '../../dataset/UCR_TS_Archive_2015'


""" classifiers
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
StandardClassifierDic = {
    'LR': LogisticRegression(),
    'LSVC': LinearSVC(),
    '1NN': KNeighborsClassifier(n_neighbors=1)
}

from time import time
from sklearn import metrics
def classify(model, x_tr, y_tr, x_te, y_te):
    ## train
    t_start = time()
    model.fit(x_tr, y_tr)
    time_tr = time() - t_start
    acc_tr = metrics.accuracy_score(y_tr, model.predict(x_tr))
    ## test
    t_start = time()
    acc_te = metrics.accuracy_score(y_te, model.predict(x_te))
    time_te = time() - t_start

    return [acc_tr, acc_te], [time_tr, time_te]

"""
    For path construction
"""

def path_split(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders


def tag_path(path, nback=1):
    """
    example:
        tag_path(os.path.abspath(__file__), 1) # return file name
    :param path: 
    :param nback: 
    :return: 
    """
    folders = path_split(path)
    nf = len(folders)

    assert nback >= 1, "nback={} should be larger than 0.".format(nback)
    assert nback <= nf, "nback={} should be less than the number of folder {}!".format(nback, nf)

    tag = folders[-1].split('.')[0]
    if nback > 0:
        for i in range(2, nback+1):
            tag = folders[-i] + '_' + tag
    return tag