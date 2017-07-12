import numpy as np
from keras.callbacks import Callback


class NLPMetrics(Callback,
                    validation_sample_lengths,
                    test_sample_lengths):
    def on_train_begin(self, logs={}):
        self.s_val_accuracy = []
        self.s_val_precision = []
        self.s_val_recall = []
        self.s_val_fscore = []

        # Adding metrics to dictionary
        self.params['metrics'].append(self.__ACC_DICT_NAME)
        self.params['metrics'].append(self.__PREC_DICT_NAME)
        self.params['metrics'].append(self.__REC_DICT_NAME)
        self.params['metrics'].append(self.__F_DICT_NAME)

        # Here at the beggining of training session the validationset
        # and testset orignal lengths will be extracted, then, at each
        # epoch end this values will be used for calculating metrics

        return


    def on_epoch_end(self, epoch, logs = {}):
        # batch_size?
        self.__p_Y = self.model.predict(self.validation_data[0])

        # Transform predicted time distributed float values
        # into 1D dichotomized sequences of integer labels
        one_dim_predicted_Y = [np.argmin( \
                                self.__p_Y[i, :self.__test_sample_sizes[i], :-1],
                                axis = 1) \
                                    for i in range(test_set_size)]

        # Select original 1D test labels
        one_dim_Y = [self.__test_Y[i, :self.__test_sample_sizes[i], 0] \
                                    for i in range(test_set_size)]
        one_dim_Y = [sequence.astype(int) for sequence in one_dim_Y]

        # Count outcome types

        # Get metrics

        # Set metrics to dictionary for output
        # logs[self.__ACC_DICT_NAME] =
        # logs[self.__PREC_DICT_NAME] =
        # logs[self.__REC_DICT_NAME] =
        # logs[self.__F_DICT_NAME] =

        # Append metrics
        # self.s_val_accuracy.append()
        # self.s_val_precision.append()
        # self.s_val_recall.append()
        # self.s_val_fscore()

        return

    __p_Y = None

    __ACC_DICT_NAME = "s_val_accuracy"
    __PREC_DICT_NAME = "s_val_precision"
    __REC_DICT_NAME = "s_val_recall"
    __F_DICT_NAME = "s_val_fscore"
