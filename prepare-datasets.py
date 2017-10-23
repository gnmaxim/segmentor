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
PROC = 6

# Some file extensions used here
WAV_EXT = ".WAV"
PHN_EXT = ".PHN"
CSV_EXT = ".csv"

# Phones path and <frame, features>.csv path, a single .csv is an utterance
PHN_PATH = "phn/"
CSV_PATH = "csv/"

# List of vowels in english (TIMIT classification)
VOWELS = "rawdata/timit/vowels"

CONFIG_PATH = "configs/features.conf"
OUTPUT = "datasets/"

VOWEL = 1
NONVOWEL = 0
ADEGUACY = 1
SAMPLE_FREQUENCY = 0.0000625 # 16kHz

PHONE_LABELS = ["start", "end", "type"]
FRAME_TIME = "frameTime"


def energy_contributions():


    return


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

        # Energy profile creation code
        expr_features['energy_profile'] = expr_features.loc[:, expr_features.columns[0]] * expr_features.loc[:, expr_features.columns[1]].values

        # Convert sample number to time (seconds)
        expr_phones.loc[:, "start":"end"] *= SAMPLE_FREQUENCY

        # Extract start points of each frame
        frame_starts = np.asarray(expr_features.loc[:, FRAME_TIME])
        frame_endings = frame_starts + frame_size
        frame_labels = np.zeros(len(frame_starts))

        is_vowel = [int(phone in vowels)
                        for phone in expr_phones.loc[:, "type"]]

        abs_diff = np.abs(np.diff(is_vowel))
        vowel_ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
        vowel_ranges[:, 0] += 1

        for vowel_range in vowel_ranges:
            indexes = [i for i in range(tot_frames) \
                        if frame_starts[i] >= expr_phones.loc[vowel_range[0], "start"] \
                            and frame_endings[i] <= expr_phones.loc[vowel_range[1], "end"]]
            frame_labels[indexes] = VOWEL


        # Label each frame with starting and ending frame phone for viewing better the outputs
        # Complex labeling code, increases a lot computational time
        
        #frame_start_phone = []
        #frame_end_phone = []
        #for phone in range(tot_phones):
            #s = [(expr_phones.loc[phone, "type"] + str(int(expr_phones.loc[phone, "type"] in vowels))) for k in range(tot_frames) \
                    #if frame_starts[k] >= expr_phones.loc[phone, "start"] \
                        #and frame_starts[k] < expr_phones.loc[phone, "end"]]
            #frame_start_phone.extend(s)

            #e = [(expr_phones.loc[phone, "type"] + str(int(expr_phones.loc[phone, "type"] in vowels))) for k in range(tot_frames) \
                    #if frame_starts[k] >= expr_phones.loc[phone, "start"] \
                        #and frame_endings[k] <= expr_phones.loc[phone, "end"]]
            #frame_end_phone.extend(e)
            #e1 = [(expr_phones.loc[phone + 1, "type"] + str(int(expr_phones.loc[phone + 1, "type"] in vowels))) for k in range(tot_frames) \
                    #if frame_starts[k] >= expr_phones.loc[phone, "start"] \
                        #and frame_starts[k] < expr_phones.loc[phone, "end"] \
                        #and frame_endings[k] > expr_phones.loc[phone, "end"] \
                        #and phone + 1 < tot_phones]
            #frame_end_phone.extend(e1)
        #frame_start_phoned = pd.DataFrame(frame_start_phone, index = None, columns = ["start_phone"])
        #frame_end_phoned = pd.DataFrame(frame_end_phone, index = None, columns = ["end_phone"])


        expr_label_data = pd.DataFrame(frame_labels, index = None, columns = ["vowel"])
        
        # In case you uncomment right above uncomment this and comment right below
        #labeled_utterance = pd.concat([expr_features, expr_label_data, \
                                        #frame_start_phoned, frame_end_phoned], axis = 1)
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
        phys_cores = PROC
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
