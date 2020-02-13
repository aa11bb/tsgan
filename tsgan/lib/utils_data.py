import numpy as np
import collections
from sklearn import utils as skutils

BaseDataSet = collections.namedtuple('BaseDataSet',
                                     ['X_train', 'y_train', 'X_test', 'y_test', 'n_classes'])

class DataSet(object):
    def __init__(self, X, y=None, seed=42):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_epochs_completed = 0
        self.i_in_epoch = 0
        self.seed = seed

    def next_batch(self, batch_size, with_shuffle=True):
        if self.y is None:
            return self._next_batch_X(batch_size, with_shuffle)
        else:
            return self._next_batch_X_y(batch_size, with_shuffle)

    def _next_batch_X(self, batch_size, with_shuffle=True):
        i_start = self.i_in_epoch
        # shuffle for the first epoch
        if self.n_epochs_completed == 0 and i_start == 0 and with_shuffle:
            self.X = shuffle(self.seed, self.X)
        # Go to the next batch
        if i_start + batch_size > self.n_samples:
            # Finished epoch
            self.n_epochs_completed += 1
            # Get the rest samples in this epoch
            n_samples_rest = self.n_samples - i_start
            X_rest = self.X[i_start:self.n_samples]
            # Shuffle the data
            if with_shuffle:
                self.X = shuffle(self.seed, self.X)
            # Start next epoch
            i_start = 0
            self.i_in_epoch = batch_size - n_samples_rest
            i_end = self.i_in_epoch
            X_new = self.X[i_start:i_end]
            X_batch = np.concatenate([X_rest, X_new], axis=0)
        else:
            self.i_in_epoch += batch_size
            i_end = self.i_in_epoch
            X_batch = self.X[i_start:i_end]

        return X_batch

    def _next_batch_X_y(self, batch_size, with_shuffle=True):
        i_start = self.i_in_epoch
        # shuffle for the first epoch
        if self.n_epochs_completed == 0 and i_start == 0 and with_shuffle:
            self.X, self.y = shuffle(self.seed,self.X, self.y)
        # Go the next batch
        if i_start + batch_size > self.n_samples:
            # Finished epoch
            self.n_epochs_completed += 1
            # Get the rest samples in this epoch
            n_samples_rest = self.n_samples - i_start
            X_rest = self.X[i_start:self.n_samples]
            y_rest = self.y[i_start:self.n_samples]
            # Shuffle the data
            if with_shuffle:
                self.X, self.y = shuffle(self.seed, self.X, self.y)
            # Start next epoch
            i_start = 0
            self.i_in_epoch = batch_size - n_samples_rest
            i_end = self.i_in_epoch
            X_new = self.X[i_start:i_end]
            y_new = self.y[i_start:i_end]

            X_batch = np.concatenate([X_rest, X_new], axis=0)
            y_batch = np.concatenate([y_rest, y_new], axis=0)
        else:
            self.i_in_epoch += batch_size
            i_end = self.i_in_epoch
            X_batch = self.X[i_start:i_end]
            y_batch = self.y[i_start:i_end]

        return X_batch, y_batch


def shuffle_list(seed, *data):
    from numpy.random import RandomState
    np_rng = RandomState(seed)
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shuffle(seed, *arrays):
    if isinstance(arrays[0][0], str):
        return shuffle_list(seed, *arrays)
    else:
        return skutils.shuffle(*arrays, random_state=seed)


def dense_to_one_hot(labels_dense, num_classes):
  """
    Convert class labels from scalars to one-hot vectors.
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def scale_image(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


def z_normalize(data):
    norm_data = data.copy()
    mean = np.mean(norm_data)
    std = np.std(norm_data)

    norm_data = norm_data - mean
    # The 1e-9 avoids dividing by zero
    norm_data = norm_data / (std + 1e-9)

    return norm_data
