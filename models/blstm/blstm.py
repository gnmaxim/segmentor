import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM

import tensorflow as tf


BATCH_SIZE = 5

def data_frames_to_arrays(list_of_dfs):
    """
        Convert list of dataframes to list of numpy arrays
    """
    list_of_arrays = []
    [list_of_arrays.append(np.asarray(df)) for df in list_of_dfs]

    return list_of_arrays


def fill_with_zeros(X, Y):
    """
        Fill dataset inputs and outputs with null values or vectors, until they
        reach the dimension of the longer utterance.

        X, Y are lists of pandas dataframes

        returns a filled X, Y as numpy arrays and the max utterance size
    """

    max_size = max([dataframe.shape[0] for dataframe in Y])

    missing_features = np.zeros(X[0].shape[1])
    missing_labels = np.zeros(Y[0].shape[1])

    X = sequence.pad_sequences(X, maxlen = max_size,
                                dtype = 'float',
                                padding = 'post',
                                truncating = 'post',
                                value = missing_features)
    Y = sequence.pad_sequences(Y, maxlen = max_size,
                                dtype = 'float',
                                padding = 'post',
                                truncating = 'post',
                                value = missing_labels)

    return [X, Y]


def get_time_distributed_labels(Y):
    """
        Y should be a pandas DataFrame

        Adds columns to labels, this neural network model will require that:
        1. a vowel will be represented by triple <1 0 0>
        2. a non vowel will be represented by triple <0 1 0>
        3. a missing value will be represented by triple <0 0 1>
    """
    Y = [y.assign(not_vowel = lambda x: np.logical_not(x.vowel).astype(float)) for y in Y]
    Y = [y.assign(missing = lambda x: 0.0) for y in Y]

    return Y


def build_model(in_shape, out_len):
    """
        1. Bidirectional LSTM
        2. Dropout
        3. TimeDistributed
    """

    model = Sequential()

    model.add(Bidirectional(LSTM(int(in_shape[0] / 2),
                                    return_sequences = True),
                                    input_shape = in_shape))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(out_len, activation = 'softmax')))

    model.compile(loss = 'binary_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])
    print (model.summary())

    return model


def train(X, Y, validation_split):
    """
        Perform training on BLSTM based network model
    """

    # Extract utterance original lengths before filling with zeros
    lengths = [dataframe.shape[0] for dataframe in Y]
    max_length = max(lengths)

    print ("1.", type(X), type(X[0]))

    # Adapt labels to TimeDistributed wrapper
    Y = get_time_distributed_labels(Y)

    X_a = data_frames_to_arrays(X)
    Y_a = data_frames_to_arrays(Y)

    print ("2.", type(X_a), type(X_a[0]), np.shape(X_a), np.shape(X_a[0]))

    # Fill with missing values
    [X_a, Y_a] = fill_with_zeros(X_a, Y_a)
    print ("3.", type(X_a), type(X_a[0]), np.shape(X_a), np.shape(X_a[0]))

    model = build_model(np.shape(X_a[0]), len(Y_a[0][0]))
    # model.fit(X_a, Y_a,
    #             validation_split = validation_split,
    #             epochs = 10,
    #             batch_size = BATCH_SIZE)

    return model


def predict(model, X, Y):
    """
        Predict outputs with model and get some metrics:
        accuracy, precision, recall and fmeasure which combines precision
        and recall equally
    """
    print ("1.", type(X), type(X[0]))
    X_a = data_frames_to_arrays(X)
    Y_a = data_frames_to_arrays(Y)
    print ("2.", type(X_a), type(X_a[0]), np.shape(X_a), np.shape(X_a[0]))
    # Fill with missing values
    [X_a, Y_a] = fill_with_zeros(X_a, Y_a)
    print ("3.", type(X_a), type(X_a[0]), np.shape(X_a), np.shape(X_a[0]))
    predictions = model.predict(X_a, batch_size = BATCH_SIZE)

    a = accuracy(Y, predictions)
    p = precision(Y, predictions)
    r = recall(Y, predictions)
    f = fmeasure(Y, predictions)


    return a, p, r, f
