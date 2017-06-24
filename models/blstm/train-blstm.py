'''
    Prosodic prominence detection in Italian continuous speech using BLSTMs

    Coded by: Maxim Gaina, maxim.gaina@yandex.ru
    Datasets provided by: Fabio Tamburini, fabio.tamburini@unibo.it
'''

from keras.models import Sequential, load_model
from keras_contrib.layers import crf
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Masking, Reshape
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import ZeroPadding1D
from keras.preprocessing import sequence
from keras import backend as K
from random import shuffle

import pandas
import numpy
import sys


# Dataset path
PATH = '../corpus/NSYLxWork.csv'

# Features per syllable and its evaluation
COLUMNS = ['nucleus-duration',
            'spectral-emphasis',
            'pitch-movements',
            'overall-intensity',
            'syllable-duration',
            'prominent-syllable']

LEARNING_PHASES = 1

#TRAINSET_LEGTH = 85; VALIDATIONSET_LENGHT = 15; TESTSET_LENGTH = 19
TRAIN_INDEXES = numpy.arange(85)
VALIDATION_INDEXES = numpy.arange(85, 100)
TESTSET_INDEXES = numpy.arange(100, 120)


dataFrame = pandas.read_csv(PATH, delim_whitespace = True,
                            header = None,
                            names = COLUMNS,
                            skip_blank_lines = False)

x_dataset = []; y_dataset = []
x_utterance = []; y_utterance = []
utterance_length = []

prominent_syllable = [0., 1.]
not_prominent_syllable = [1., 0.]
missing_features = [0., 0., 0., 0., 0.]

max_utterance_length = 0

print ("\nExtracting utterances from", PATH)

for index, row in dataFrame.iterrows():
    # Extracting feature vector per syllable and its prominence evaluation
    features = row['nucleus-duration':'syllable-duration'].values
    prominence = row['prominent-syllable']

    # If vector contiains NaN then the expression is finished
    if numpy.isnan(prominence):
        length = len(x_utterance)
        utterance_length.append(length)

        if max_utterance_length < length:
            max_utterance_length = length

        # Append expression to dataset
        x_dataset.append(x_utterance)
        y_dataset.append(y_utterance)

        x_utterance = []
        y_utterance = []
    else:
        if prominence == 0:
            prominence = not_prominent_syllable
        else:
            prominence = prominent_syllable
        # Append syllable features to current expression
        x_utterance.append(features)
        y_utterance.append(prominence)

total_expressions = len(y_dataset)


print ("... extracted ", total_expressions, "utterances. Dataset split:")
print ("\tTraining set", len(TRAIN_INDEXES), "expressions")
print ("\tValidation set", len(VALIDATION_INDEXES), "expressions")
print ("\tTest set", len(TESTSET_INDEXES), "expressions")
print ("\nLongest expression composed by", max_utterance_length, "syllables.")

print ("Filling shorter expressions with zeroes...")
x_dataset = sequence.pad_sequences(x_dataset,
                                maxlen = max_utterance_length,
                                dtype = 'float',
                                padding = 'post',
                                truncating = 'post',
                                value = missing_features)
y_dataset = sequence.pad_sequences(y_dataset,
                                maxlen = max_utterance_length,
                                dtype = 'float',
                                padding = 'post',
                                truncating = 'post',
                                value = prominent_syllable)

x_dataset = numpy.asarray(x_dataset)
y_dataset = numpy.asarray(y_dataset)
utterance_length = numpy.asarray(utterance_length)


indexes = numpy.arange(total_expressions)
scores = []
lf_manual_accuracy = []; lf_manual_precision = []
lf_manual_recall = []; lf_manual_fscore = []

numpy.set_printoptions(threshold = sys.maxsize)
numpy.set_printoptions(threshold = sys.maxsize)

for learning_phase in range(LEARNING_PHASES):
    numpy.random.shuffle(indexes)
    x_dataset = x_dataset[indexes]
    y_dataset = y_dataset[indexes]
    utterance_length = utterance_length[indexes]

    print ("Dataset shuffled")

    # Building Model
    model = Sequential()
    #model.add(ZeroPadding1D(input_shape = (max_utterance_length, len(COLUMNS) - 1)))

    model.add(Masking(input_shape = (max_utterance_length, len(COLUMNS) - 1),
                        mask_value = missing_features))

    model.add(LSTM(17, return_sequences = True))
    model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(len(prominent_syllable), activation = 'softmax')))

    model.compile(loss = 'binary_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])
    print (model.summary())

    # No known nb_epoch and batch_size gives better metrics than this
    model.fit(x_dataset[TRAIN_INDEXES], y_dataset[TRAIN_INDEXES],
                validation_data = (x_dataset[VALIDATION_INDEXES], y_dataset[VALIDATION_INDEXES]),
                epochs = 50, batch_size = 2)

    metrics = model.evaluate(x_dataset[TESTSET_INDEXES],
                            y_dataset[TESTSET_INDEXES],
                            verbose = 1)

    scores.append(metrics)

    predictions = model.predict(x_dataset[TESTSET_INDEXES], batch_size = 2)
    predictions = numpy.argmax(predictions, axis = 2)
    y = numpy.argmax(y_dataset[TESTSET_INDEXES], axis = 2)

    zeros = numpy.zeros(max_utterance_length)
    ones = numpy.ones(max_utterance_length)

    # number of correct non prominency forecastings:    a
    # number of incorrect prominency forecastings:      b
    # number of incorrect non prominency forecastings:  c
    # numbero of correct prominency forecastings:       d
    a = []; b = []; c = []; d = []
    accuracy = []; precision = []; recall = []; fmeasure = []

    for i in range(len(TESTSET_INDEXES)):
        dataFrame = pandas.DataFrame(data = numpy.transpose([
                            predictions[i][:utterance_length[TESTSET_INDEXES][i]],
                            y[i][:utterance_length[TESTSET_INDEXES][i]]]))
        # print (dataFrame)

        ''' couples order: (0, 0) (0, 1) (1, 0) (1, 1) '''
        pairs = numpy.bincount(dataFrame.dot([2, 1]), minlength = 4)
        print (pairs)
        a = pairs[0]; b = pairs[2]; c = pairs[1]; d = pairs[3]

        accuracy.append((a + d) / (a + b + c + d))
        precision.append(d / (b + d))
        recall.append(d / (c + d))
        fmeasure.append(2 * ((precision[-1] * recall[-1]) / (precision[-1] + recall[-1])))

    lf_manual_accuracy.append(numpy.mean(accuracy))
    lf_manual_precision.append(numpy.mean(precision))
    lf_manual_recall.append(numpy.mean(recall))
    lf_manual_fscore.append(numpy.mean(fmeasure))

    print ("Learning phase", learning_phase + 1, "scores:")
    print ("\tKeras Accuracy %.2f%%" % (metrics[1] * 100))
    print ("\tManual Accuracy %.2f%%" % (lf_manual_accuracy[-1] * 100))
    # print ("\tManual Precision %.2f%%" % lf_manual_precision[-1])
    # print ("\tManual Recall %.2f%%" % lf_manual_recall[-1])
    # print ("\tManual F-measure F1 %.2f%%" % lf_manual_fscore[-1])

    del model

overall_scores = numpy.mean(scores, axis = 0)
print ("Overall scores:")
print ("\tKeras Accuracy %.2f%%" % (overall_scores[1] * 100))

numpy.set_printoptions(threshold = sys.maxsize)

# NOTA: col vecchio modello meno sono le sillabe migliori sono le metriche,
# più direct hit... in media, 3 sillabe mancanti in più aggiungono 0,5% di A
