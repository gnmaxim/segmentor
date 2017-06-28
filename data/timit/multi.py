import configparser as cp
import itertools as it
import pandas as pd
import numpy as np
import argparse
import timeit
import sys
import os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial

"""
    These global constants will not be modified
"""
# Number of processes to create
PROC = 4

# Some file extensions used here
WAV_EXT = ".WAV"
PHN_EXT = ".PHN"
CSV_EXT = ".csv"

# Phones path and <frame, features>.csv path, a single .csv is an utterance
PHN_PATH = "phn/"
CSV_PATH = "csv/"

# List of vowels in english (TIMIT classification)
VOWELS = "vowels.csv"

CONFIG_PATH = "../../configs/features.conf"
PART_PATH = "../../configs/partitioning.conf"
OUTPUT = "../../models/"
SETS = ["trainset", "validationset", "testset"]

VOWEL = 1
NONVOWEL = 0
ADEGUACY = 0.75
SAMPLE_FREQUENCY = 0.0000625 # 16kHz

PHONE_LABELS = ["start", "end", "type"]
FRAME_TIME = "frameTime"



def set_labels(utterance_filenames, input_dir, frame_size, frame_step):
    """
        Takes a chunk of filenames (one per each utterance) containing features.
        A labeled set of utterances (frame by frame) will be returned, specifying
        if that frame is part of a vowel or not.
    """

    labeled_utterances = []

    for feature_file, i in zip(utterance_filenames, np.arange(len(utterance_filenames))):
        #if feature_file.endswith("30.csv"):
        # Load utterance frames with features
        expr_features = pd.read_csv(input_dir + CSV_PATH + feature_file,
                                    delimiter = ";")
        print (feature_file,
                "\tgetting labeled by pid:", os.getpid(),
                "\t", i, "th utterance")

        # Load phones of the same utterance
        phone_file = feature_file[: -len(CSV_EXT)] + PHN_EXT
        expr_phones = pd.read_csv(input_dir + PHN_PATH + phone_file,
                                        delim_whitespace = True,
                                        header = None,
                                        names = PHONE_LABELS)

        tot_frames = expr_features.shape[0]
        tot_phones = expr_phones.shape[0]

        # Convert sample number to time (seconds)
        expr_phones.loc[:, "start":"end"] *= SAMPLE_FREQUENCY
        # expr_time_length = expr_phones.iloc[-1, expr_phones.columns.get_loc("end")]

        # Extract start points of each frame
        frame_start_points = np.asarray(expr_features.loc[:, FRAME_TIME])

        labels = []
        for phone in range(tot_phones):
            for frame in (k for k in range(tot_frames) if
                (frame_start_points[k] >= expr_phones.loc[phone, "start"]) and
                (frame_start_points[k] < expr_phones.loc[phone, "end"])):

                frame_ending = frame_start_points[frame] + frame_size

                if (frame_ending <= expr_phones.loc[phone, "end"]):
                    # if current frame is totally a part of one single phone
                    # ending boundary INCLUDED, then label the frame with phone type
                    labels.append(int(expr_phones.loc[phone, "type"] in vowels))
                else:
                    # time interval of current frame over next phone
                    nextPhonePart = (frame_ending -
                                        expr_phones.loc[phone, "end"]) / frame_size

                    if (phone + 1 < tot_phones \
                        and nextPhonePart >= ADEGUACY):
                        # more than 1-tollerance * 100 % of this frame is on the next phone
                        labels.append(int(expr_phones.loc[phone + 1, "type"] in vowels))
                    else:
                        # less than 1-tollerance * 100 % of this frame is on the next phone
                        labels.append(int(expr_phones.loc[phone, "type"] in vowels))


        expr_label_data = pd.DataFrame(labels, index = None, columns = ["vowel"])
        labeled_utterance = pd.concat([expr_features, expr_label_data], axis = 1)

        final_features = list(labeled_utterance.columns.values)
        nans = np.full([len(final_features)], np.nan)
        s = pd.Series(nans, index = final_features)
        labeled_utterance = labeled_utterance.append(s, ignore_index = True)

        labeled_utterances.append(labeled_utterance)

    return labeled_utterances



if __name__=="__main__":
    """
        The main purpose of this parallel labeling is to optimize time, so we don't
        care about memory usage here.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
                        help = "Path containing wav, phn and csv directories.")
    parser.add_argument("-v", "--validationset",
                        help = "Transform a slice of dataset into validationset, \
                        with propoprions that are specified in " + PART_PATH + ". \
                        \nNote that the rest of the dataset content will be created \
                        as trainset.",
                        action = "store_true")
    args = parser.parse_args()

    if args.directory:
        input_dir = args.directory
        feature_files = os.listdir(input_dir + CSV_PATH)
        tot_files = len(feature_files)

        sets = []
        if args.validationset:
            train_files, val_files, w, v = train_test_split(feature_files,
                                                np.zeros(len(feature_files)),
                                                test_size = 0.2,
                                                train_size = 0.8)

            # Sets must be appended in this order: train, validation, test
            sets.append(train_files)
            sets.append(val_files)
            del w, v
        else:
            sets.append(feature_files)


        featureConfig = cp.ConfigParser(comment_prefixes = ("/", ";", "#"),
                                        strict = True)
        featureConfig.read(CONFIG_PATH)
        print ("\nDetected openSMILE components:")
        for section in featureConfig.sections():
            print ("\t", section)
            if "cFramer" in section:
                frame_size = featureConfig.getfloat(section, "frameSize")
                frame_step = featureConfig.getfloat(section, "frameStep")

        print ("\nDetected frame size:", frame_size, "s")
        print ("Detected frame step:", frame_step, "s")


        vowels = np.asarray(pd.read_csv(VOWELS, header = None), dtype = str)
        vowels = np.reshape(vowels, len(vowels))
        print ("\n", len(vowels),"vowel symbols: ", *vowels, sep = " ")
    else:
        print ("No input directory provided.")

    for set_type, set_name in zip(sets, SETS):
        chunks = np.array_split(set_type, PROC)

        pool = Pool(processes = PROC)

        start_time = timeit.default_timer()
        labeled_utterances = pool.starmap(set_labels, zip(chunks,
                                                    it.repeat(input_dir),
                                                    it.repeat(frame_size),
                                                    it.repeat(frame_step)))
        elapsed_time = timeit.default_timer() - start_time
        print (set_name, elapsed_time)

        # Remove .csv from previous computations, if exists
        os.remove(OUTPUT + set_name + CSV_EXT)

        features = list(labeled_utterances[0][0].columns.values)
        for utterances in labeled_utterances:
            for utterance in utterances:
                if not os.path.isfile(OUTPUT + set_name + CSV_EXT):
                    # if file does not exist write header
                    utterance.to_csv(OUTPUT + set_name + CSV_EXT,
                                            header = features,
                                            index = False,
                                            sep = ";")
                else:
                    # if it exists then simply append
                    utterance.to_csv(OUTPUT + set_name + CSV_EXT,
                                            mode = "a",
                                            header = False,
                                            index = False,
                                            sep = ";")
