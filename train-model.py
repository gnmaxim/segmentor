import configparser as cp
import itertools as it
import pandas as pd
import numpy as np
import argparse
import timeit
import sys
import os

from multiprocessing import Pool
from functools import partial

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

from models import blstm_v1
import matplotlib.pyplot as plt


PROC = 6
VAL_PROP = 0.2
SESSIONS_PATH = "sessions/"


def unpack_dataset(dataset):
    raw_dataset = pd.read_csv(dataset, delimiter = ";",
                                    skip_blank_lines = False)
    features = list(raw_dataset.columns.values)

    # get indexes of rows that are frames and not blank lines
    is_frame = raw_dataset.loc[:, features[0]].notnull()

    # give the same odd index to frames that are in the same utterance
    utterance_partition = (is_frame != is_frame.shift()).cumsum()

    # select frames from dataframe and group them by utterance,
    # the result is a set of dataframe per expression
    grouped_by_utterance = raw_dataset[is_frame].groupby(utterance_partition)

    # extract the maximum known length
    utterance_lengths = grouped_by_utterance.apply(len)
    max_utterance_length = np.max(utterance_lengths)

    keys = grouped_by_utterance.groups.keys()
    groups = [grouped_by_utterance.get_group(key) for key in keys]

    X = [group[features[0: -1]] for group in groups]
    Y = [group[features[-1]].to_frame() for group in groups]

    return [X, Y, max_utterance_length]


if __name__ == "__main__":
    """
        The main purpose of this parallel labeling is to optimize time, so we don't
        care about memory usage here.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--session_name",
                        required = True,
                        help = "Name of folder where models will be saved")
    parser.add_argument("-m", "--model_type",
                        required = True,
                        choices = ["td-blstm"],
                        help = "Specify model type, \'td-blstm\' for time \
                                distributed blstm, \'mtd-blstm\' for \
                                time distributed blstm with masked input")
    parser.add_argument("-t", "--trainset",
                        required = True,
                        help = "path to trainset .csv file")
    parser.add_argument("-v", "--static_validation",
                        required = True,
                        type = float,
                        choices = np.arange(.0, .4, .05),
                        help = "use static validation set shrinked from trainset")
    parser.add_argument("-s", "--testset",
                        required = True,
                        help = "path to testset .csv file")
    parser.add_argument("-c", "--cores",
                        type = int,
                        choices = range(1, 8),
                        help = "Number of physical core to use")
    args = parser.parse_args()


    learning_session = SESSIONS_PATH \
                        + args.model_type + "/" \
                        + args.session_name + "/"

    if os.path.exists(learning_session):
        raise ValueError("Session with name", learning_session, "already exists")
    else:
        os.makedirs(learning_session)

        phys_cores = PROC
        if args.cores:
            phys_cores = args.cores

        validation_split = VAL_PROP
        if args.static_validation:
            validation_split = args.static_validation


        pool = Pool(processes = phys_cores)

        start_time = timeit.default_timer()
        data = pool.map(unpack_dataset, [args.trainset, args.testset])
        elapsed_time = timeit.default_timer() - start_time
        print ("\nData unpacking performed in\t", elapsed_time, "seconds")

        max_utterance_length = max([length[2] for length in data])
        print ("Longest utterance has\t\t", max_utterance_length, "frames")

        train_X = data[0][0]
        train_Y = data[0][1]
        test_X = data[1][0]
        test_Y = data[1][1]

        td_blstm = blstm_v1.TimeDistributedBlstm(train_X, train_Y,
                                                    test_X, test_Y,
                                                    info = True)
        print (td_blstm.get_model_summary())
        history = td_blstm.train(validation_split,
                                    session_path = learning_session)

        scores = td_blstm.predict()
        print (scores)
