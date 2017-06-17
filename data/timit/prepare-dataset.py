'''
    Script that assigns label to each extracted frame vector
'''

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np
import sys
import os


EXT_CSV = '.csv'; EXT_WAV = '.WAV'; EXT_PHN = '.PHN'
PATH_CSV = 'csv/'; PATH_PHN = 'phn/';
PATH_OUTPUT = '../../nn/'; PATH_OUTPUT_FILE = 'dataset.csv'
VOWELS = 'vowels.csv'
FRAME_TIME = 'frameTime'; FRAME_SIZE = 0.0025
PHONE_LABELS = ['start', 'end', 'type']
VOWEL = 1; NOT_VOWEL = 0
SAMPLE_FREQUENCY = 1 / 16000    # 16KHz
TOLLERANCE = 0.25


vowels = np.asarray(pd.read_csv(VOWELS, header = None), dtype = str)
vowels = np.reshape(vowels, len(vowels))
print (vowels)


frameLabels = []
overallPhoneDurations = []
overallVowelDurations = []
overallNonVowelDurations = []

maxFrameSize = 0    # Frame number of longest utterance
np.set_printoptions(threshold = sys.maxsize)


for featureFile in os.listdir(PATH_CSV):
    #if featureFile.endswith('.csv'):
    if featureFile.startswith("_home_maxim_Desktop_prominator_timit_TIMIT_TEST_DR1_FAKS0"):

        exprSmileData = pd.read_csv(PATH_CSV + featureFile,
                                        delimiter = ';')
        print (np.asarray(exprSmileData))

        # Watch out: deleting first column which is useless, but if you change
        # openSmile .conf file you may don't need this anymore
        exprSmileData.drop('name', axis = 1, inplace = True)

        exprFrameSize = exprSmileData.shape[0]

        if maxFrameSize == 0:
            features = list(exprSmileData.columns.values)
            if (FRAME_TIME in features) and exprFrameSize > 3:
                frameStep = exprSmileData[FRAME_TIME][2] - exprSmileData[FRAME_TIME][1]
            else:
                raise ValueError('ERROR: there is no', FRAME_TIME,
                                    'column name. Or insufficient number of frames')

        if exprFrameSize > maxFrameSize:
            maxFrameSize = exprFrameSize


        # To ensure that was loaded corresponding .PHN file
        phoneFile = featureFile[: -len(EXT_CSV)] + EXT_PHN
        exprPhoneData = pd.read_csv(PATH_PHN + phoneFile,
                                        delim_whitespace = True,
                                        header = None,
                                        names = PHONE_LABELS)

        # Phone number of current utterance
        exprPhoneSize = exprPhoneData.shape[0]

        # Convert sample number to time (seconds)
        exprPhoneData.loc[:, 'start':'end'] *= SAMPLE_FREQUENCY
        exprTimeLength = exprPhoneData.iloc[-1, exprPhoneData.columns.get_loc('end')]
        # print (exprPhoneData)

        # Preparing arrays of phone durations to use evenually use them for statistics
        overallPhoneDurations.extend(
            np.asarray(exprPhoneData.loc[:, 'end'] - exprPhoneData.loc[:, 'start']))
        overallVowelDurations.extend(
            np.asarray(exprPhoneData.loc[exprPhoneData['type'].isin(vowels), 'end'] -
                        exprPhoneData.loc[exprPhoneData['type'].isin(vowels), 'start']))
        overallNonVowelDurations.extend(
            np.asarray(exprPhoneData.loc[exprPhoneData['type'].isin(vowels) == False, 'end'] -
                        exprPhoneData.loc[exprPhoneData['type'].isin(vowels) == False, 'start']))

        # Generaing frames start point
        frStartPoints = np.asarray(exprSmileData.loc[:, FRAME_TIME])

        # print ('frames', exprFrameSize, 'frStartPoints', len(frStartPoints))

        labels = []
        for phone in range(exprPhoneSize):
            print (exprPhoneData.loc[phone, 'type'], '\t',
                        exprPhoneData.loc[phone, 'type'] in vowels, '\t',
                        exprPhoneData.loc[phone, 'end'] - exprPhoneData.loc[phone, 'start'])

            for frame in (k for k in range(exprFrameSize) if
                (frStartPoints[k] >= exprPhoneData.loc[phone, 'start']) and
                (frStartPoints[k] < exprPhoneData.loc[phone, 'end'])):

                frameEnd = frStartPoints[frame] + FRAME_SIZE

                if (frameEnd <= exprPhoneData.loc[phone, 'end']):
                    # if current frame is totally a part of one single phone
                    # ending boundary INCLUDED, then label the frame with phone type
                    labels.append(int(exprPhoneData.loc[phone, 'type'] in vowels))
                else:
                    # time interval of current frame over new phone
                    nextPhonePart = (frameEnd -
                                        exprPhoneData.loc[phone, 'end']) / FRAME_SIZE

                    if nextPhonePart >= (1 - TOLLERANCE):
                        # more than 1-tollerance * 100 % of this frame is on the next phone
                        labels.append(int(exprPhoneData.loc[phone + 1, 'type'] in vowels))
                    else:
                        # less than 1-tollerance * 100 % of this frame is on the next phone
                        labels.append(int(exprPhoneData.loc[phone, 'type'] in vowels))

            # print (labels)
            print ('Frames Number: ', exprFrameSize)
            print ('Label Vector: ', len(labels))


        exprLabelData = pd.DataFrame(labels, index = None, columns = ['vowel'])

        utteranceData = pd.concat([exprSmileData, exprLabelData], axis = 1)

        finalFeatures = list(utteranceData.columns.values)
        nans = np.full([len(finalFeatures)], np.nan)
        s = pd.Series(nans, index = finalFeatures)
        utteranceData = utteranceData.append(s, ignore_index = True)

        if not os.path.isfile(PATH_OUTPUT + PATH_OUTPUT_FILE):
            # if file does not exist write header
            utteranceData.to_csv(PATH_OUTPUT + PATH_OUTPUT_FILE,
                                    header = finalFeatures,
                                    index = False,
                                    sep = ';')
        else:
            # if it exists so append without writing the header
            utteranceData.to_csv(PATH_OUTPUT + PATH_OUTPUT_FILE,
                                    mode = 'a',
                                    header = False,
                                    index = False,
                                    sep = ';')


# overallPhoneDurations = [p for p in overallPhoneDurations if p < 0.5]
# overallNonVowelDurations = [p for p in overallNonVowelDurations if p < 0.5]
#
# meanPhoneLength = np.mean(overallPhoneDurations)
# maxPhoneSize = np.max(overallPhoneDurations)
# minPhoneLength = np.min(overallPhoneDurations)
# varPhoneLength = np.var(overallPhoneDurations)
# stdPhoneLength = np.std(overallPhoneDurations)
# print ('\nPHONE STATS')
# print ('Mean duration:', meanPhoneLength, 's')
# print ('Duration variance:', varPhoneLength, 's')
# print ('Min duration:', minPhoneLength, 's\n',
#         'Max duration:', maxPhoneSize, 's')
#
# meanVowelLength = np.mean(overallVowelDurations)
# maxVowelLength = np.max(overallVowelDurations)
# minVowelLength = np.min(overallVowelDurations)
# varVowelLength = np.var(overallVowelDurations)
# stdVowelLength = np.std(overallVowelDurations)
# print ('\nVOWEL STATS')
# print ('Mean duration', meanVowelLength, 's')
# print ('Duration variance', varVowelLength, 's')
# print ('Min duration:', minVowelLength, 's\n',
#         'Max duration:', maxVowelLength, 's')
#
# meanNonVowelLength = np.mean(overallNonVowelDurations)
# varNonVowelLength = np.var(overallNonVowelDurations)
# stdNonVowelLength = np.std(overallNonVowelDurations)
# print ('\nNON-VOWEL STATS')
# print ('Mean duration', meanNonVowelLength, 's')
# print ('Duration variance', varNonVowelLength, 's')
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
