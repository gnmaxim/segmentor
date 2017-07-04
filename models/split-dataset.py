import configparser as cp
import pandas as pd
import numpy as np
import argparse
import timeit
import sys
import os
from multiprocessing import Pool
from functools import partial



PHYS_CORES = 4

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

    # convert utterance dataframes to numpy 3D array, that is
    # <utterance, frame, feature>
    keys = grouped_by_utterance.groups.keys()
    groups = [grouped_by_utterance.get_group(key) for key in keys]

    X = [group.as_matrix([features[0: -1]]) for group in groups]
    Y = [group.as_matrix([features[-1]]) for group in groups]

    return X, Y, max_utterance_length


if __name__=="__main__":
    """
        The main purpose of this parallel labeling is to optimize time, so we don't
        care about memory usage here.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainset", help = "path to trainset .csv file")
    parser.add_argument("-v", "--validationset", help = "path to validationset .csv file")
    parser.add_argument("-s", "--testset", help = "path to testset .csv file")
    parser.add_argument("-c", "--cores", type = int, help = "Number of physical core to use")
    args = parser.parse_args()

    if args.trainset and args.testset: #args.validationset and :
        phys_cores = PHYS_CORES
        if args.cores:
            phys_cores = args.cores


        pool = Pool(processes = phys_cores)
        start_time = timeit.default_timer()
        data = pool.map(unpack_dataset, [args.trainset,
                                                    args.testset])
                                         #args.validationset])
        elapsed_time = timeit.default_timer() - start_time
        print ("\nData unpacking performed in\t", elapsed_time, "seconds")

        data = np.asmatrix(data)
        max_utterance_length = int(max(data[:, 2]))
        print ("Longest utterance has\t\t", max_utterance_length, "frames")

        # Must parallelize data padding
    else:
        parser.print_help()


"""
    https://github.com/fchollet/keras/issues/1711

    execute-timit.sh
        extract-corpus.sh
        prepare-dataset-parallel.py
        train-model.py
            blstm.py
            convolutiona.py

"""
