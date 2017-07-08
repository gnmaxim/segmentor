import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM


class Blstm:
    def __init__(self, x, y, t_x, t_y):
        y    = self.__get_time_distributed_labels(y)
        t_y  = self.__get_time_distributed_labels(t_y)

        self.__train_X = self.__to_arrays(x)
        self.__train_Y = self.__to_arrays(y)
        self.__test_X  = self.__to_arrays(t_x)
        self.__test_Y  = self.__to_arrays(t_y)

        self.__get_max_sample_size()

        self.__train_X = self.__fill_with_null_vectors(self.__train_X)
        self.__train_Y = self.__fill_with_null_vectors(self.__train_Y)
        self.__test_X  = self.__fill_with_null_vectors(self.__test_X)
        self.__test_Y  = self.__fill_with_null_vectors(self.__test_Y)

        self.__input_shape = self.__get_input_shape(self.__train_X)
        self.__output_shape = self.__get_output_shape(self.__train_Y)

        return None


    def __get_time_distributed_labels(self, labels):
        labels = [l.assign(logic_not = lambda x: \
                np.logical_not(x[x.columns[0]]).astype(float)) \
                    for l in labels]
        labels = [l.assign(missing = lambda x: .0) for l in labels]


        return labels


    def __to_arrays(self, objects):
        array_list = []
        [array_list.append(np.asarray(obj)) for obj in objects]

        return array_list


    def __get_max_sample_size(self):
        sizes = [y.shape[0] for y in self.__train_Y]
        t_sizes = [y.shape[0] for y in self.__test_Y]

        sizes.extend(t_sizes)
        self.__max_sample_size = max(sizes)

        return None


    def __fill_with_null_vectors(self, seq):
        missing_vector_dim = np.zeros(seq[0].shape[1])

        seq = sequence.pad_sequences(seq,
                            dtype = 'float',
                            padding = 'post',
                            truncating = 'post',
                            value = missing_vector_dim,
                            maxlen = self.__max_sample_size)

        return seq


    def __get_input_shape(self, x):
        input_shape = self.__train_X[0].shape

        print (input_shape)

        return input_shape


    def __get_output_shape(self, x):
        output_shape = self.__train_Y[0].shape

        print (output_shape)

        return output_shape


    def build(self):
        self.__model = Sequential()
        self.__model.add(Bidirectional \
                            (LSTM(int(self.__max_sample_size / 2),
                                    return_sequences = True),
                                    input_shape = self.__input_shape))
        self.__model.add(Dropout(0.5))
        self.__model.add(TimeDistributed \
                            (Dense(self.__output_shape[1],
                                    activation = "softmax")))
        self.__model.compile(optimizer = "adam",
                                    metrics = ["accuracy"],
                                    loss = "binary_crossentropy")

        return None


    def train(self, val_static_split):
        self.__model.fit(self.__train_X, self.__train_Y,
                        validation_split = val_static_split,
                        batch_size = self.__BATCH_SIZE,
                        epochs = self.__EPOCHS)

        return None


    def get_keras_model(self):
        return self.__model


    def get_model_summary(self):
        return self.__model.summary()


    def predict(self):
        self.__predicted_Y = self.__model.predict(self.__test_X,
                                            batch_size = self.__BATCH_SIZE)

        return None


    __EPOCHS = 10
    __BATCH_SIZE = 10

    __model = None

    __train_X = None
    __train_Y = None
    __test_X  = None
    __test_Y  = None

    __input_shape = None
    __output_shape = None
    __max_sample_size = None

    __predicted_Y = None
