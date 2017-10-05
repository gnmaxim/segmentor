import os
import h5py
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM

from models import NLPCallbacks, NLPMetrics


class TimeDistributedBlstm:
    def __init__(self, train_x, train_y,
                       test_x, test_y,
                       info = False):

        train_y = self.__to_time_distributed_labels(train_y)
        test_y  = self.__to_time_distributed_labels(test_y)

        self.__train_X = self.__to_arrays(train_x)
        self.__train_Y = self.__to_arrays(train_y)
        self.__test_X  = self.__to_arrays(test_x)
        self.__test_Y  = self.__to_arrays(test_y)

        self.__store_max_sample_size()

        self.__train_X = self.__input_null_vectors(self.__train_X)
        self.__train_Y = self.__output_null_vectors(self.__train_Y)
        self.__test_X  = self.__input_null_vectors(self.__test_X)
        self.__test_Y  = self.__output_null_vectors(self.__test_Y)

        self.__input_shape = self.__get_input_shape(self.__train_X)
        self.__output_shape = self.__get_output_shape(self.__train_Y)

        self.__extract_lens = lambda set: [len([l for l in sequence if l[-1] != 1])
                                                    for sequence in set]
        self.__to_one_dimension_label = lambda seqs, sizes, set_size: \
                                [np.argmin(seqs[i, :sizes[i], :-1], axis = 1) \
                                                    for i in range(set_size)]

        self.__test_sample_sizes = self.__extract_lens(self.__test_Y)

        self.__build()

        self.__info = info

        if self.__info:
            self.__print_all_info()

        return

    def __to_time_distributed_labels(self, labels):
        """
            # Convert labels to time distributed format:
                [1, 0, 0] means 1
                [0, 1, 0] means 0
                [0, 0, 1] means void (element of sequence is absent)
        """
        labels = [l.assign(logic_not = lambda x: \
                np.logical_not(x[x.columns[0]]).astype(float)) \
                    for l in labels]
        labels = [l.assign(missing = lambda x: .0) for l in labels]

        return labels


    def __to_arrays(self, objects):
        array_list = []
        [array_list.append(np.asarray(obj)) for obj in objects]

        return array_list


    def __store_max_sample_size(self):
        sizes = [y.shape[0] for y in self.__train_Y]
        t_sizes = [y.shape[0] for y in self.__test_Y]

        sizes.extend(t_sizes)
        self.__max_sample_size = max(sizes)

        return


    def __input_null_vectors(self, sequences):
        """
            # Insert null vectors at the end of input data in such a way that all
            samples will have the same size
        """
        missing_vector_dim = np.zeros(sequences[0].shape[1])

        sequences = sequence.pad_sequences(sequences,
                                        dtype = 'float',
                                        padding = 'post',
                                        truncating = 'post',
                                        value = missing_vector_dim,
                                        maxlen = self.__max_sample_size)

        return sequences

    def __output_null_vectors(self, sequences):
        """
            # Insert [0, 0, 1] at the end of output data in such a way that all
            samples will have the same size, where [0, 0, 1] means indeed that a
            element of sequence is absent
        """
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

        # self.__model.add(Bidirectional( \
        #                     LSTM(int(self.__max_sample_size / 2),
        #                           return_sequences = True),
        #                           input_shape = (self.__input_shape)))
        # Stateful
        self.__model.add(Masking( \
                            mask_value = np.zeros(self.__train_X[0].shape[1]),
                            batch_input_shape = (self.__BATCH_SIZE,
                                                self.__input_shape[0],
                                                self.__input_shape[1])))
        self.__model.add(Bidirectional( \
                            LSTM(int(self.__max_sample_size / 2),
                                  stateful = True,
                                  return_sequences = True)))
        self.__model.add(Dropout(0.5))
        self.__model.add(TimeDistributed \
                            (Dense(self.__output_shape[1],
                                   activation = "softmax")))
        self.__model.compile(optimizer = "adam",
                                   metrics = ["categorical_accuracy"],
                                   loss = "binary_crossentropy")

        return


    def train(self, val_static_split, session_path):
        """
            # Before training the custom metrics class will be instantiated as a
            callback, and others like (best) save model callback, TensorBoard
            callback for session monitoring, a csv statistic writer. Then perform
            model training

            # Returns training history
        """
        nlp_metrics = NLPCallbacks.CallbackBinaryEvaluator(
                                    batch_size = self.__BATCH_SIZE,
                                    len_counter = self.__extract_lens,
                                    label_adaptor = self.__to_one_dimension_label)

        acc = nlp_metrics.get_accuracy_dict_name()
        fscore = nlp_metrics.get_fscore_dict_name()

        savename = "{epoch:02d}-{val_loss:.2f}-{" + acc + "}"
        bests_savename = "BEST-" + savename

        save_all = NLPCallbacks.get_saver(session_path + savename)
        save_best = NLPCallbacks.get_bests_saver(session_path + savename, fscore)
        train_board = NLPCallbacks.get_monitor_board(session_path + "visual")
        train_csv = NLPCallbacks.get_csv_logger(session_path + "statistics.csv")
        lr_cooler = NLPCallbacks.learning_rate_cooler("val_loss")

        train_history = self.__model.fit(self.__train_X, self.__train_Y,
                                    validation_split = val_static_split,
                                    shuffle = True,
                                    batch_size = self.__BATCH_SIZE,
                                    epochs = self.__EPOCHS,
                                    callbacks = [nlp_metrics,
                                                 save_all,
                                                 save_best,
                                                 train_board,
                                                 train_csv,
                                                 lr_cooler])

        return train_history


    def get_model_summary(self):
        return self.__model.summary()


    def predict(self):
        test_set_size = np.shape(self.__test_Y)[0]

        self.__predicted_Y = self.__model.predict(self.__test_X,
                                    batch_size = self.__BATCH_SIZE)

        one_dim_predicted_Y = self.__to_one_dimension_label(self.__predicted_Y,
                                                        self.__test_sample_sizes,
                                                        len(self.__predicted_Y))
        one_dim_Y = self.__to_one_dimension_label(self.__test_Y,
                                                        self.__test_sample_sizes,
                                                        len(self.__test_Y))
        one_dim_Y = [sequence.astype(int) for sequence in one_dim_Y]

        binary_evaluator = NLPMetrics.BinarySequenceEvaluator(one_dim_predicted_Y,
                                                        one_dim_Y)

        accuracy = binary_evaluator.accuracy()
        precision = binary_evaluator.precision()
        recall = binary_evaluator.recall()
        fscore = binary_evaluator.f_score()

        return accuracy, precision, recall, fscore


    def __print_all_info(self):
        print ("\nModel type: Time Distributed BLSTM with static input dimension")

        print ("\tInput shape:\t\t", self.__input_shape)
        print ("\tOutput shape:\t\t", self.__output_shape)
        print ("\tMaximum Sample size:\t", self.__max_sample_size)

        print ("\tMissing input vector:\t", self.__train_X[0][-1])
        print ("\tMissing output vector:\t", self.__train_Y[0][-1])

        print ("\n\t", self.__EPOCHS, "epochs with batch size",self.__BATCH_SIZE)

        return


    __EPOCHS = 50
    __BATCH_SIZE = 4

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

    __extract_lens = None
    __to_one_dimension_label = None

    __info = None
    __SAVE_DIR = None
