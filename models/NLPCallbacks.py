from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau

from models import NLPMetrics

import numpy as np


class CallbackBinaryEvaluator(Callback):
    def __init__(self, batch_size,
                       len_counter,
                       label_adaptor):

        self.__len_counter = len_counter
        self.__label_adaptor = label_adaptor
        self.__BATCH_SIZE = batch_size

        return


    def on_train_begin(self, logs = {}):
        self.s_val_accuracy = []
        self.s_val_precision = []
        self.s_val_recall = []
        self.s_val_fscore = []

        # Adding metrics to dictionary
        self.params['metrics'].append(self.__NLP_ACCURACY_NAME)
        self.params['metrics'].append(self.__NLP_PRECISION_NAME)
        self.params['metrics'].append(self.__NLP_RECALL_NAME)
        self.params['metrics'].append(self.__NLP_FSCORE_NAME)

        return


    def on_epoch_end(self, epoch, logs = {}):
        if epoch == 0:
            self.__val_size = len(self.validation_data[1])
            self.__val_sample_sizes = self.__len_counter(self.validation_data[1])
            self.__val_one_dim_labels = self.__label_adaptor(
                                                        self.validation_data[1],
                                                        self.__val_sample_sizes,
                                                        self.__val_size)

        self.__p_Y = self.model.predict(self.validation_data[0],
                                        batch_size = self.__BATCH_SIZE)
        self.__p_Y = self.__label_adaptor(self.__p_Y,
                                        self.__val_sample_sizes,
                                        self.__val_size)

        bin_evaluator = NLPMetrics.BinarySequenceEvaluator(self.__p_Y,
                                        self.__val_one_dim_labels)

        accuracy = bin_evaluator.accuracy()
        precision = bin_evaluator.precision()
        recall = bin_evaluator.recall()
        fscore = bin_evaluator.f_score()

        # Set metrics to dictionary for output
        logs[self.__NLP_ACCURACY_NAME] = np.array(accuracy)
        logs[self.__NLP_PRECISION_NAME] = np.array(precision)
        logs[self.__NLP_RECALL_NAME] = np.array(recall)
        logs[self.__NLP_FSCORE_NAME] = np.array(fscore)

        # Append metrics
        self.s_val_accuracy.append(accuracy)
        self.s_val_precision.append(precision)
        self.s_val_recall.append(recall)
        self.s_val_fscore.append(fscore)

        return


    def get_accuracy_dict_name(self):
        return self.__NLP_ACCURACY_NAME


    def get_precision_dict_name(self):
        return self.__NLP_PRECISION_NAME


    def get_recall_dict_name(self):
        return self.__NLP_RECALL_NAME


    def get_fscore_dict_name(self):
        return self.__NLP_FSCORE_NAME


    __p_Y = None

    __NLP_ACCURACY_NAME = "nlp_val_accuracy"
    __NLP_PRECISION_NAME = "nlp_val_precision"
    __NLP_RECALL_NAME = "nlp_val_recall"
    __NLP_FSCORE_NAME = "nlp_val_fscore"

    __val_size = None
    __val_sample_sizes = None
    __val_one_dim_labels = None

    __len_counter = None
    __label_adaptor = None


def get_saver(filepath):
    saver = ModelCheckpoint(filepath, period = 1)

    return saver



def get_bests_saver(filepath, feature_to_monitor):
    bests_saver = ModelCheckpoint(filepath,
                            monitor = feature_to_monitor,
                            save_best_only = True)

    return bests_saver


def get_monitor_board(log_directory):
    monitor = TensorBoard(log_dir = log_directory,
                            histogram_freq = 1,
                            write_graph = True,
                            write_grads = True,
                            write_images = True)

    return monitor


def get_csv_logger(filepath):
    csv_l = CSVLogger(filename = filepath,
                            separator = ",",
                            append = False)

    return csv_l


def learning_rate_cooler(feature_to_monitor, patience = 5):
    reduce_lr = ReduceLROnPlateau(monitor = feature_to_monitor,
                                patience = patience,
                                factor = 0.1,
                                min_lr = 0.001,
                                verbose = 1)

    return reduce_lr
