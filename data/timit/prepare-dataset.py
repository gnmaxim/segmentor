'''
    Script that assigns label to each extracted frame vector
'''

import pandas as pd
import numpy as np
import sys
import os


PATH_CSV = 'csv/'
PATH_PHN = 'phn/'
FEATURES = []


maxFrameNumber = 0
np.set_printoptions(threshold = sys.maxsize)

for featureFile in os.listdir(PATH_CSV):
    if featureFile.endswith(".csv"):
        utteranceData = pd.read_csv(PATH_CSV + featureFile, delimiter = ';')
        # print (np.asarray(dataFrame))

        frameNumber = utteranceData.shape[0]
        if frameNumber > maxFrameNumber:
            maxFrameNumber = frameNumber

        # for index, row in utteranceData.iterrows():
        #    if index == 1:
        #        print (np.asarray(row.values))


# PADDING BASED ON maxFrameNumber

'''
A. a single utterance .csv file

    frame_time [vector of feature]

B. starting and ending of a phone

    (phone, duration) ------> (frame, phone)

C. Unifying A and B
'''
