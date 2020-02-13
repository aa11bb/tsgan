from data import utils
from data import ucr


if __name__ == '__main__':
    DATA_ROOT = '../../dataset/UCR_TS_Archive_2015'
    filename = '50words'
    datasets = ucr.load_ucr(filename, DATA_ROOT, one_hot=False)
    X_train = datasets.train.X
    y_train = datasets.train.y
    X_test = datasets.test.X
    y_test = datasets.test.y

    distr = utils.distribute_dataset(X_train, y_train)
    print(distr.keys())