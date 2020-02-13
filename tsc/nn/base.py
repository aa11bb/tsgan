"""Base class for all TSC models realized by Keras

References:
---------------------
https://github.com/hfawaz/dl-4-tsc
@article{fawaz2019deep,
  title={Deep learning for time series classification: a review},
  author={Fawaz, Hassan Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal={Data Mining and Knowledge Discovery},
  volume={33},
  number={4},
  pages={917--963},
  year={2019},
  publisher={Springer}
}

"""


from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import os
import numpy as np
import keras


class BaseClassifierDNNKeras(BaseEstimator):
    """Base class for the TSC model realized by Keras
    Notes:
        - tensor is a three-dimensional array corresponding multivariate time series.
        - y should be represented by one-hot vector
    """
    def __init__(self, input_shape, n_classes, verbose=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.verbose = verbose
        self.batch_size_tr = None
        self.model = None

    def build_model(self, **kwargs):
        raise NotImplementedError('This is an abstract method.')

    def fit(self, x,
            y,
            batch_size=None,
            n_epochs=None,
            validation_data=None,
            shuffle=True,
            **kwargs):
        raise  NotImplementedError('This is an abstract method.')

    def predict_proba(self, x, batch_size=None, **kwargs):
        probas = self.model.predict(x, batch_size, **kwargs)
        return probas

    def predict(self, x, batch_size=None, **kwargs):
        probas = self.predict_proba(x, batch_size)
        y_pred = np.argmax(probas, axis=1)
        return y_pred

    def evaluate(self, x, y, batch_size=None, **kwargs):
        y_pred = self.predict(x, batch_size)
        y_true = np.argmax(y, axis=1)
        acc = accuracy_score(y_true, y_pred)
        return acc

    def save(self, dir, fname):
        self.model.save(os.path.join(dir, '{}.hdf5'.format(fname)))

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)


