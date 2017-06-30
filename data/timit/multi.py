"""
    Script that assigns label to each extracted frame vector

    consider this before training:
        1. enabled/disabled noHeader openSMILE configuration
        2. enabled/disabled voiced cutoff configuration
            2a. enabled/disabled voicing probability
            NOTED that a lot of cuted off frames are actually vowels
        3. erase h#
        4. tollerance (0.25 or 0)

    done:
        - eliminate ffmpeg dependecy (BETTER ffmpeg than without)
        - multiprocess labeling
"""

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
# Number of core to use by default
PHYS_CORES = 4

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
OUTPUT = "../../models/"

VOWEL = 1
NONVOWEL = 0
ADEGUACY = 0.75
SAMPLE_FREQUENCY = 0.0000625 # 16kHz

PHONE_LABELS = ["start", "end", "type"]
FRAME_TIME = "frameTime"



def perform_frame_labeling(utterance_filenames,
                            input_dir,
                            frame_size,
                            frame_step):
    """
        Takes a chunk of filenames (one file per each utterance) containing features.
        A labeled set of utterances (frame by frame) will be returned, specifying
        if that frame is part of a vowel or not.
    """

    labeled_utterances = []

    for feature_file, i in zip(utterance_filenames, np.arange(len(utterance_filenames))):
        # if feature_file.endswith("30.csv"):
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

                    if phone + 1 < tot_phones and nextPhonePart >= ADEGUACY:
                        # more than ADEGUACY*100% of this frame covers the next phone
                        labels.append(int(expr_phones.loc[phone + 1, "type"] in vowels))
                    else:
                        # less than ADEGUACY*100% of this frame covers the next phone
                        labels.append(int(expr_phones.loc[phone, "type"] in vowels))


        expr_label_data = pd.DataFrame(labels, index = None, columns = ["vowel"])
        labeled_utterance = pd.concat([expr_features, expr_label_data], axis = 1)

        # Throwing away frame start info which is no more relevant
        labeled_utterance.drop(labeled_utterance.columns[0], axis = 1, inplace = True)

        # In the final .csv utterances will be separated by a blank line,
        # so at the end of utterance's DataFrame a vector of NaNs must be inserted
        final_features = list(labeled_utterance.columns.values)
        nans = np.full([len(final_features)], np.nan)
        s = pd.Series(nans, index = final_features)
        labeled_utterance = labeled_utterance.append(s, ignore_index = True)

        labeled_utterances.append(labeled_utterance)

    return labeled_utterances



if __name__ == "__main__":
    """
        The main purpose of this parallel labeling is to optimize time, so we don't
        care about memory usage here.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
                        help = "Path containing wav, phn and csv directories.")
    parser.add_argument("-o", "--output",
                        help = "Name of the output file containing labeled data")
    parser.add_argument("-c", "--cores", type = int,
                        help = "Number of physical core to use")
    args = parser.parse_args()

    if args.directory and args.output:
        phys_cores = PHYS_CORES
        if args.cores:
            phys_cores = args.cores

        input_dir = args.directory
        output_file = args.output

        if not output_file.endswith(CSV_EXT):
            output_file += CSV_EXT

        feature_files = os.listdir(input_dir + CSV_PATH)
        tot_files = len(feature_files)

        # Searching for frameSize and frameStep infos inside specified
        # openSMILE configuration file
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

        # Slicing data for some parallel processing
        chunks = np.array_split(feature_files, phys_cores)

        pool = Pool(processes = phys_cores)

        start_time = timeit.default_timer()
        labeled_utterances = pool.starmap(perform_frame_labeling, zip(chunks,
                                                    it.repeat(input_dir),
                                                    it.repeat(frame_size),
                                                    it.repeat(frame_step)))
        elapsed_time = timeit.default_timer() - start_time
        print (output_file, elapsed_time)

        # Remove .csv from previous computations, if exists
        if os.path.exists(OUTPUT + output_file):
            os.remove(OUTPUT + output_file)

        features = list(labeled_utterances[0][0].columns.values)
        for utterances in labeled_utterances:
            for utterance in utterances:
                if not os.path.isfile(OUTPUT + output_file):
                    # if file does not exist write header
                    utterance.to_csv(OUTPUT + output_file,
                                            header = features,
                                            index = False,
                                            sep = ";")
                else:
                    # if it exists then simply append
                    utterance.to_csv(OUTPUT + output_file,
                                            mode = "a",
                                            header = False,
                                            index = False,
                                            sep = ";")
    else:
        parser.print_help()
