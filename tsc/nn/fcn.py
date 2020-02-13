from .base import BaseClassifierDNNKeras

import pandas as pd
import keras

class FCN(BaseClassifierDNNKeras):
    def __init__(self, input_shape, n_classes, verbose=1):
        super(FCN, self).__init__(input_shape, n_classes, verbose)

        # default parameters
        self.batch_size = 16
        self.n_epochs = 2000

        # set up model
        self.x = keras.Input(self.input_shape)
        self.output = self.build_model()
        self.model = keras.models.Model(inputs=self.x, outputs=self.output)
        if (self.verbose > 0):
            self.model.summary()

    def build_model(self):
        conv1 = keras.layers.Conv1D(128, 8, padding='same')(self.x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        conv2 = keras.layers.Conv1D(256, 5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, 3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)
        out = keras.layers.Dense(self.n_classes, activation='softmax')(gap)

        return out

    def fit(self, x,
            y,
            batch_size=None,
            n_epochs=None,
            validation_data=None,
            shuffle=True,
            **kwargs):
        # set parameters
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = min(x.shape[0] // 10, batch_size)  # default: wang2017time
        if n_epochs is None:
            n_epochs = self.n_epochs
            # n_epochs = 3 # for test

        # start to train
        optimizer = keras.optimizers.Adam()
        self.model.compile(
            loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        if validation_data is None:
            hist = self.model.fit(
                x, y, batch_size=batch_size, epochs=n_epochs,
                verbose=self.verbose, callbacks=[reduce_lr])
        else:
            hist = self.model.fit(
                x, y, batch_size=batch_size, epochs=n_epochs,
                verbose=self.verbose, validation_data=validation_data, callbacks=[reduce_lr])
        return pd.DataFrame(hist.history)



