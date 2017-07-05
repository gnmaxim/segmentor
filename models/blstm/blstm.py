import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM



def fill_with_zeros(X, Y):
    """
        Fill dataset inputs and outputs with null values or vectors, until they
        reach the dimension of the longer utterance.

        X, Y are lists of pandas dataframes

        returns a filled X, Y as numpy arrays and the max utterance size
    """

    filled_X = []
    filled_Y = []

    max_size = max([dataframe.shape[0] for dataframe in Y])

    [filled_X.append(np.asarray(x)) for x in X]
    [filled_Y.append(np.asarray(y)) for y in Y]

    missing_features = np.zeros(X[0].shape[1])
    missing_labels = np.zeros(Y[0].shape[1])

    filled_X = sequence.pad_sequences(filled_X,
                            maxlen = max_size,
                            dtype = 'float',
                            padding = 'post',
                            truncating = 'post',
                            value = missing_features)
    filled_Y = sequence.pad_sequences(filled_Y,
                            maxlen = max_size,
                            dtype = 'float',
                            padding = 'post',
                            truncating = 'post',
                            value = missing_labels)

    return [filled_X, filled_Y]


def build_model(in_shape, out_len):
    """
        1. Bidirectional LSTM
        2. Dropout
        3. TimeDistributed
    """

    model = Sequential()

    model.add(Bidirectional(LSTM(in_shape[0],
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

    """
        Adding columns to Y label, this neural network model will require that:
        1. a vowel will be represented by triple <1 0 0>
        2. a non vowel will be represented by triple <0 1 0>
        3. a missing value will be represented by triple <0 0 1>
    """
    Y = [y.assign(not_vowel = lambda x: np.logical_not(x.vowel).astype(float)) for y in Y]
    Y = [y.assign(missing = lambda x: 0.0) for y in Y]

    # Fill with missing values
    [X, Y] = fill_with_zeros(X, Y)

    model = build_model(np.shape(X[0]), len(Y[0][0]))

    model.fit(X, Y, validation_split = validation_split, epochs = 50, batch_size = 5)



    return "wtf"



def test(test_X, test_Y):

    return "wtf"
