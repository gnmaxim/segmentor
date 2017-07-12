import os
import h5py
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM

from models import evaluator, callback_evaluator


class TimeDistributedBlstm:
    def __init__(self, train_x, train_y,
                        test_x, test_y,
                        info = False,):
        # Convert labels to time distributed format
        train_y = self.__to_time_distributed_labels(train_y)
        test_y  = self.__to_time_distributed_labels(test_y)

        self.__train_X = self.__to_arrays(train_x)
        self.__train_Y = self.__to_arrays(train_y)
        self.__test_X  = self.__to_arrays(test_x)
        self.__test_Y  = self.__to_arrays(test_y)

        self.__store_test_sample_sizes()
        self.__store_max_sample_size()

        # All sequences must have the same length
        self.__train_X = self.__input_null_vectors(self.__train_X)
        self.__train_Y = self.__output_null_vectors(self.__train_Y)
        self.__test_X  = self.__input_null_vectors(self.__test_X)
        self.__test_Y  = self.__output_null_vectors(self.__test_Y)

        self.__input_shape = self.__get_input_shape(self.__train_X)
        self.__output_shape = self.__get_output_shape(self.__train_Y)

        self.__build()

        self.__info = info

        if self.__info:
            self.__print_all_info()

        return None


    def __to_time_distributed_labels(self, labels):
        labels = [l.assign(logic_not = lambda x: \
                np.logical_not(x[x.columns[0]]).astype(float)) \
                    for l in labels]
        labels = [l.assign(missing = lambda x: .0) for l in labels]

        return labels


    def __to_arrays(self, objects):
        array_list = []
        [array_list.append(np.asarray(obj)) for obj in objects]

        return array_list


    def __store_test_sample_sizes(self):
        self.__test_sample_sizes = [y.shape[0] for y in self.__test_Y]

        return None


    def __store_max_sample_size(self):
        sizes = [y.shape[0] for y in self.__train_Y]
        t_sizes = [y.shape[0] for y in self.__test_Y]

        sizes.extend(t_sizes)
        self.__max_sample_size = max(sizes)

        return None


    def __input_null_vectors(self, sequences):
        missing_vector_dim = np.zeros(sequences[0].shape[1])

        sequences = sequence.pad_sequences(sequences,
                                        dtype = 'float',
                                        padding = 'post',
                                        truncating = 'post',
                                        value = missing_vector_dim,
                                        maxlen = self.__max_sample_size)

        return sequences


    def __output_null_vectors(self, sequences):
        missing_vector_dim = np.zeros(sequences[0].shape[1])
        missing_vector_dim[-1] = 1

        sequences = sequence.pad_sequences(sequences,
                                        dtype = 'float',
                                        padding = 'post',
                                        truncating = 'post',
                                        value = missing_vector_dim,
                                        maxlen = self.__max_sample_size)

        return sequences


    def __get_input_shape(self, x):
        input_shape = self.__train_X[0].shape

        return input_shape


    def __get_output_shape(self, x):
        output_shape = self.__train_Y[0].shape

        return output_shape


    def __build(self):
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

# LearningRateScheduler
# ReduceLROnPlateau
# CSVLogger
# LambdaCallback

    def train(self, val_static_split, session_path):
        savename = "{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}"
        bests_savename = "BEST-" + savename

        # Creating callbacks relatively to saving models
        savior = ModelCheckpoint(session_path + savename,
                                    period = 1)
        best_savior = ModelCheckpoint(session_path + bests_savename,
                                    monitor = "val_loss",
                                    save_best_only = True)

        # Callback for visualizing some graphs
        visual = TensorBoard(log_dir = session_path + "visual",
                            histogram_freq = 1,
                            write_graph = True,
                            write_grads = True,
                            write_images = True)

        # Callback for some .csv output
        to_csv = CSVLogger(filename = session_path + "out.csv",
                            separator = ",",
                            append = False)

        ce = callback_evaluator.ValidationMetrics()

        history = self.__model.fit(self.__train_X, self.__train_Y,
                                    validation_split = val_static_split,
                                    shuffle = False,
                                    batch_size = self.__BATCH_SIZE,
                                    epochs = self.__EPOCHS,
                                    callbacks = [savior,
                                                best_savior,
                                                visual,
                                                to_csv,
                                                ce])

        return history


    def get_keras_model(self):
        return self.__model


    def get_model_summary(self):
        return self.__model.summary()


    def predict(self):
        test_set_size = np.shape(self.__test_Y)[0]

        self.__predicted_Y = self.__model.predict(self.__test_X,
                                            batch_size = self.__BATCH_SIZE)

        # Transform predicted time distributed float values
        # into 1D dichotomized sequences of integer labels
        one_dim_predicted_Y = [np.argmin( \
                                self.__predicted_Y[i, :self.__test_sample_sizes[i], :-1],
                                axis = 1) \
                                    for i in range(test_set_size)]

        # Select original 1D test labels
        one_dim_Y = [self.__test_Y[i, :self.__test_sample_sizes[i], 0] \
                                    for i in range(test_set_size)]
        one_dim_Y = [sequence.astype(int) for sequence in one_dim_Y]

        binary_evaluator = evaluator.BinaryEvaluator(one_dim_predicted_Y, one_dim_Y)

        accuracy = binary_evaluator.accuracy()
        precision = binary_evaluator.precision()
        recall = binary_evaluator.recall()
        fscore = binary_evaluator.f_score()

        return accuracy, precision, recall, fscore


    def __print_all_info(self):
        print ("Model type: Time Distributed BLSTM with static input dimension")

        print ("Input shape:\t\t", self.__input_shape)
        print ("Output shape:\t\t", self.__output_shape)
        print ("Maximum Sample size:\t", self.__max_sample_size)

        print ("Missing input vector:\t", self.__train_X[0][-1])
        print ("Missing output vector:\t", self.__train_Y[0][-1])

        print ("\t", self.__EPOCHS, "epochs with batch size",self.__BATCH_SIZE)

        return None


    __EPOCHS = 70
    __BATCH_SIZE = 2

    __model = None

    __train_X = None
    __train_Y = None
    __test_X  = None
    __test_Y  = None

    __input_shape = None
    __output_shape = None
    __max_sample_size = None

    __predicted_Y = None
    __test_sample_sizes = None

    __info = None
    __SAVE_DIR = None
