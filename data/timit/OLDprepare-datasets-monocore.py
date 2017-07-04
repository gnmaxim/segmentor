"""
    Script that assigns label to each extracted frame vector

    TODOs:
        1. eliminate FRAME_TIME dependecy
        2. data partitioning
            2a. extend to test

    VARIABLES TO CONSIDER BEFORE TRAINING:
        1. enabled/disabled noHeader openSMILE configuration
        2. enabled/disabled voiced cutoff configuration
            2a. enabled/disabled voicing probability
            NOTED that a lot of cuted off frames are actually vowels
        3. erase h#
        4. tollerance (0.25 or 0)

    DONE:
        1. eliminate ffmpeg dependecy (BETTER ffmpeg than without)
"""

from sklearn.model_selection import train_test_split
import argparse
import configparser as cp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np
import sys
import os



WAV_EXT = ".WAV"
PHN_EXT = ".PHN"
CSV_EXT = ".csv"
PHN_PATH = "phn/"
CSV_PATH = "csv/"
VOWELS = "vowels.csv"
CONFIG_PATH = "../../configs/features.conf"
PART_PATH = "../../configs/partitioning.conf"
OUTPUT = "../../models/"
# Add test
SETS = ["trainset", "validationset"]

VOWEL = 1
NONVOWEL = 0
TOLLERANCE = 0.25
SAMPLE_FREQUENCY = 0.0000625 # 16kHz

PHONE_LABELS = ["start", "end", "type"]
FRAME_TIME = "frameTime"



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory",
                    help = "Path containing wav, phn and csv directories.")
# parser.add_argument("-t", "--testset",
#                     help = "Transform a slice of dataset into testset, \
#                     with propoprions that are specified in " + PART_PATH + ".",
#                     action = "store_true")
parser.add_argument("-v", "--validationset",
                    help = "Transform a slice of dataset into validationset, \
                    with propoprions that are specified in " + PART_PATH + ". \
                    \nNote that the rest of the dataset content will be created \
                    as trainset.",
                    action = "store_true")
args = parser.parse_args()


if args.directory:
    inputDir = args.directory

    featureFiles = os.listdir(inputDir + CSV_PATH)
    totFiles = len(featureFiles)


    pc = cp.ConfigParser(comment_prefixes = ("/", ";", "#"),
                                    strict = True)
    pc.read(PART_PATH)

    # temporary code
    sets = []
    if args.validationset:
        trainFiles, valFiles, z1, z2 = train_test_split(featureFiles,
                                                np.zeros(len(featureFiles)),
                                                test_size = 0.2,
                                                train_size = 0.8)
        sets.append(trainFiles)
        sets.append(valFiles)
    else:
        sets.append(featureFiles)

    featureConfig = cp.ConfigParser(comment_prefixes = ("/", ";", "#"),
                                    strict = True)
    featureConfig.read(CONFIG_PATH)
    print ("\nDetected openSMILE components:")
    for section in featureConfig.sections():
        print ("\t", section)
        if "cFramer" in section:
            frameSize = featureConfig.getfloat(section, "frameSize")
            frameStep = featureConfig.getfloat(section, "frameStep")

    print ("\nDetected frame size:", frameSize, "s")
    print ("Detected frame step:", frameStep, "s")


    vowels = np.asarray(pd.read_csv(VOWELS, header = None), dtype = str)
    vowels = np.reshape(vowels, len(vowels))
    print ("\n", len(vowels),"vowel symbols: ", *vowels, sep = " ")


    overallPhoneDurations = []
    overallVowelDurations = []
    overallNonVowelDurations = []

    # Frame number of longest utterance
    setCount = 0
    maxTotFrames = 0
    maxTotPhones = 0
    np.set_printoptions(threshold = sys.maxsize)

    for setType in sets:
        #for featureFile in featureFiles:
        for featureFile in setType:
            if featureFile.endswith("30.csv"):
            #if featureFile.startswith("_home_maxim_Desktop_prominator_corpus_timit_TIMIT_TEST_DR1_FAKS0"):

                exprFeatures = pd.read_csv(inputDir + CSV_PATH + featureFile,
                                                delimiter = ";")

                # Extracting number of frames for current utterance
                totFrames = exprFeatures.shape[0]

                #if maxTotFrames == 0:
                    # extracting header
                    # features = list(exprFeatures.columns.values)

                if totFrames > maxTotFrames:
                    maxTotFrames = totFrames


                # Load phones of the same utterance
                phoneFile = featureFile[: -len(CSV_EXT)] + PHN_EXT
                exprPhones = pd.read_csv(inputDir + PHN_PATH + phoneFile,
                                                delim_whitespace = True,
                                                header = None,
                                                names = PHONE_LABELS)

                # Phone number of current utterance
                totPhones = exprPhones.shape[0]

                if totPhones > maxTotPhones:
                    maxTotPhones = totPhones

                print ("\nLabeling utterance made of", totFrames, "frames and", totPhones, "phones")
                print ("\t", featureFile)

                # Convert sample number to time (seconds)
                exprPhones.loc[:, "start":"end"] *= SAMPLE_FREQUENCY
                exprTimeLength = exprPhones.iloc[-1, exprPhones.columns.get_loc("end")]
                # print (exprPhones)

                # Preparing arrays of phone durations to use evenually use them for statistics
                # overallPhoneDurations.extend(
                #     np.asarray(exprPhones.loc[:, "end"] - exprPhones.loc[:, "start"]))
                # overallVowelDurations.extend(
                #     np.asarray(exprPhones.loc[exprPhones["type"].isin(vowels), "end"] -
                #                 exprPhones.loc[exprPhones["type"].isin(vowels), "start"]))
                # overallNonVowelDurations.extend(
                #     np.asarray(exprPhones.loc[exprPhones["type"].isin(vowels) == False, "end"] -
                #                 exprPhones.loc[exprPhones["type"].isin(vowels) == False, "start"]))

                frStartPoints = np.asarray(exprFeatures.loc[:, FRAME_TIME])

                # print ("frames", totFrames, "frStartPoints", len(frStartPoints))

                labels = []
                for phone in range(totPhones):
                    # print (exprPhones.loc[phone, "type"], "\t",
                    #             exprPhones.loc[phone, "type"] in vowels, "\t",
                    #             exprPhones.loc[phone, "end"] - exprPhones.loc[phone, "start"])

                    for frame in (k for k in range(totFrames) if
                        (frStartPoints[k] >= exprPhones.loc[phone, "start"]) and
                        (frStartPoints[k] < exprPhones.loc[phone, "end"])):

                        frameEnd = frStartPoints[frame] + frameSize

                        if (frameEnd <= exprPhones.loc[phone, "end"]):
                            # if current frame is totally a part of one single phone
                            # ending boundary INCLUDED, then label the frame with phone type
                            labels.append(int(exprPhones.loc[phone, "type"] in vowels))
                        else:
                            # time interval of current frame over new phone
                            nextPhonePart = (frameEnd -
                                                exprPhones.loc[phone, "end"]) / frameSize

                            if (phone + 1 < totPhones \
                                and nextPhonePart >= 1 - TOLLERANCE):
                                # more than 1-tollerance * 100 % of this frame is on the next phone
                                labels.append(int(exprPhones.loc[phone + 1, "type"] in vowels))
                            else:
                                # less than 1-tollerance * 100 % of this frame is on the next phone
                                labels.append(int(exprPhones.loc[phone, "type"] in vowels))

                    # print (labels)
                    # print ("Frames Number: ", totFrames)
                    # print ("Label Vector: ", len(labels))


                exprLabelData = pd.DataFrame(labels, index = None, columns = ["vowel"])

                utteranceData = pd.concat([exprFeatures, exprLabelData], axis = 1)

                finalFeatures = list(utteranceData.columns.values)
                nans = np.full([len(finalFeatures)], np.nan)
                s = pd.Series(nans, index = finalFeatures)
                utteranceData = utteranceData.append(s, ignore_index = True)


                if not os.path.isfile(OUTPUT + SETS[setCount] + CSV_EXT):
                    # if file does not exist write header
                    utteranceData.to_csv(OUTPUT + SETS[setCount] + CSV_EXT,
                                            header = finalFeatures,
                                            index = False,
                                            sep = ";")
                else:
                    # if it exists so append without writing the header
                    utteranceData.to_csv(OUTPUT + SETS[setCount] + CSV_EXT,
                                            mode = "a",
                                            header = False,
                                            index = False,
                                            sep = ";")

        setCount += 1

    # overallPhoneDurations = [p for p in overallPhoneDurations if p < 0.5]
    # overallNonVowelDurations = [p for p in overallNonVowelDurations if p < 0.5]
    #
    # meanPhoneLength = np.mean(overallPhoneDurations)
    # maxPhoneSize = np.max(overallPhoneDurations)
    # minPhoneLength = np.min(overallPhoneDurations)
    # varPhoneLength = np.var(overallPhoneDurations)
    # stdPhoneLength = np.std(overallPhoneDurations)
    # print ("\nPHONE STATS")
    # print ("Mean duration:", meanPhoneLength, "s")
    # print ("Duration variance:", varPhoneLength, "s")
    # print ("Min duration:", minPhoneLength, "s\n",
    #         "Max duration:", maxPhoneSize, "s")
    #
    # meanVowelLength = np.mean(overallVowelDurations)
    # maxVowelLength = np.max(overallVowelDurations)
    # minVowelLength = np.min(overallVowelDurations)
    # varVowelLength = np.var(overallVowelDurations)
    # stdVowelLength = np.std(overallVowelDurations)
    # print ("\nVOWEL STATS")
    # print ("Mean duration", meanVowelLength, "s")
    # print ("Duration variance", varVowelLength, "s")
    # print ("Min duration:", minVowelLength, "s\n",
    #         "Max duration:", maxVowelLength, "s")
    #
    # meanNonVowelLength = np.mean(overallNonVowelDurations)
    # varNonVowelLength = np.var(overallNonVowelDurations)
    # stdNonVowelLength = np.std(overallNonVowelDurations)
    # print ("\nNON-VOWEL STATS")
    # print ("Mean duration", meanNonVowelLength, "s")
    # print ("Duration variance", varNonVowelLength, "s")
    #
    # xp = meanPhoneLength + stdPhoneLength * np.asarray(overallPhoneDurations)
    # xv = meanVowelLength + stdVowelLength * np.asarray(overallVowelDurations)
    # xn = meanNonVowelLength + stdNonVowelLength * np.asarray(overallNonVowelDurations)
    #
    # num_bins = 300
    # fig, ax = plt.subplots()
    #
    # ax.hist(xn, num_bins, normed = 0)
    # ax.hist(xv, num_bins, normed = 0)
    #
    # plt.xticks(np.arange(min(xn), max(xv), 0.002))
    # fig.tight_layout()
    # plt.show()

    print ("\nLongest utterance made of:")
    print ("\tframes", maxTotFrames)
    print ("\tphones", maxTotPhones, "\n")

else:
    print ("\n\tTap \"--help\" on your beatiful keyboard.\n")
