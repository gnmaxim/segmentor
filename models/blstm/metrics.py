import numpy as np


"""
    On binary labels we have that:
        number of correct 0 negative forecastings:      a
        number of incorrect 1 positive forecastings:    b
        number of incorrect 0 negative forecastings:    c
        number of correct 1 positive forecastings:      d

    accuracy = (a + d) / (a + b + c + d)
    precision = d / (b + d)
    recall = d / (c + d)
    fmeasure = 2 * ((precision * recall) / (precision + recall))
"""

def get_accuracy(pred_Y, Y):
    return "a"


def get_precision(pred_Y, Y):
    return "a"


def get_recall(pred_Y, Y):
    return "a"


def get_fmeasure(pred_Y, beta):
    return "a"

"""
    >>> np.array([0, 0]).dot(1 << np.arange(a.size)[:-1])
    0
    >>> np.array([1, 0]).dot(1 << np.arange(a.size)[:-1])
    1
    >>> np.array([0, 1]).dot(1 << np.arange(a.size)[:-1])
    2
    >>> np.array([1, 1]).dot(1 << np.arange(a.size)[:-1])
    3

        1. convert Y p_Y to arrays
        2. np.dstack((l4[:,:,0], l5[:,:,0]))
            array([[[0, 1],
                [1, 2],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 2],
                [1, 2],
                [0, 1]],

               [[0, 1],
                [1, 2],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 2],
                [1, 2],
                [0, 1]]])

        it only requires to substitute l and l1 with p_Y and Y
"""

def get_prediction_scores(p_Y, Y):
    true_neg = 0    # a
    false_pos = 0   # b
    false_neg = 0   # c
    true_pos = 0    # d

    ind = [true_neg, false_pos, false_neg, true_pos]

    p_Y = np.asarray(p_Y)
    Y = np.asarray(Y)

    np.concatenate((p_Y[]), axis = )




    for i in range(len(Y)):
        couples = []
        # TRANSFORM IN MATRIX
        [couples.append(np.array([p_Y[i][frame][0], Y[i][frame][0]])) \
                        for frame in range Y[i] if Y[i][frame][-1] == 0]
        [ind[pos = lambda x: couple.dot(1 << np.arange(a.size)[:-1])] \
                        for couple in couples]

    accuracy = get_accuracy
