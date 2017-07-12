import numpy as np


class BinaryEvaluator:
    """
        Given a 1D sequence of predicted and original labels in {0, 1},
        BinaryEvaluator can be used to compute Accuracy, Precision, Recall and
        F-score with beta = 1 parameter.
    """
    def __init__(self, predicted_labels, labels):
        self.__p_Y = predicted_labels
        self.__Y = labels

        self.__count_outcome_types()

        return None


    def __count_outcome_types(self):
        """
            There are 4 result types that are observable from predictions.
            If on the first column are placed predicted labels and original
            labels on the other, then:
                (0, 0)    denote true negative predictions
                (1, 0)    denote false positive predictions
                (0, 1)    denote false negative predictions
                (1, 1)    denote positive predictions

            But the above tuples if converted to decimal are integers from 0 to 3,
            which can be the index of an iterable. This property is used below.
        """
        results = [0, 0, 0, 0]

        predicted_labels = np.concatenate(self.__p_Y)
        labels = np.concatenate(self.__Y)

        # The i-th predicted label and non are merged into a tuple
        # of form < predicted_label, label >
        comparisons = np.dstack((predicted_labels, labels))[0]

        # It's time to count result types as described in the top comment
        for k in range(len(comparisons)):
            results[int(comparisons[k].dot(1 << np.arange(comparisons[k].size)[:]))] += 1

        self.__true_neg, self.__false_pos, \
        self.__false_neg, self.__true_pos = results

        return None


    def accuracy(self):
        accuracy = (self.__true_neg + self.__true_pos) \
                    / (self.__true_neg + self.__false_pos \
                        + self.__false_neg + self.__true_pos)

        return accuracy


    def precision(self):
        precision = self.__true_pos / (self.__false_pos + self.__true_pos)

        return precision


    def recall(self):
        recall = self.__true_pos / (self.__false_neg + self.__true_pos)

        return recall


    def f_score(self, beta = 1):
        f_score = (1 + pow(beta, 2)) \
                    * (self.precision() * self.recall()) \
                    / ((pow(beta, 2) * self.precision()) + self.recall())

        return f_score


    __p_Y = None
    __Y = None

    __true_pos  = 0
    __false_pos = 0
    __false_neg = 0
    __true_neg  = 0
