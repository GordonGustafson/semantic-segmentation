import numpy as np

from evaluation import confusion_matrix, precision_and_recall, f1_score

import unittest

diagonal = np.array([[1, 0],
                     [0, 1]])
flipped_diagonal = np.array([[0, 1],
                             [1, 0]])
columns = np.array([[0, 1],
                    [0, 1]])
all_ones = np.ones((2, 2))

all_correct_confusion_matrix = np.array([[2, 0],
                                         [0, 2]])
all_wrong_confusion_matrix = np.array([[0, 2],
                                       [2, 0]])

class TestEvaluation(unittest.TestCase):
    def test_confusion_matrix(self):

        self.assertTrue(np.array_equal(confusion_matrix(diagonal, diagonal), all_correct_confusion_matrix))
        self.assertTrue(np.array_equal(confusion_matrix(diagonal, flipped_diagonal), all_wrong_confusion_matrix))
        self.assertTrue(np.array_equal(confusion_matrix(diagonal, columns), all_ones))
        self.assertTrue(np.array_equal(confusion_matrix(diagonal, all_ones), np.array([[0, 2],
                                                                                       [0, 2]])))
        self.assertTrue(np.array_equal(confusion_matrix(all_ones, diagonal), np.array([[0, 0],
                                                                                       [2, 2]])))

    def test_precision_and_recall(self):
        all_correct_result = precision_and_recall(all_correct_confusion_matrix)
        self.assertTrue(np.allclose(all_correct_result["precision"], np.array([1.0, 1.0])))
        self.assertTrue(np.allclose(all_correct_result["recall"], np.array([1.0, 1.0])))

        all_wrong_result = precision_and_recall(all_wrong_confusion_matrix)
        self.assertTrue(np.allclose(all_wrong_result["precision"], np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(all_wrong_result["recall"], np.array([0.0, 0.0])))

        result_1234 = precision_and_recall(np.array([[1, 2],
                                                     [3, 4]]))
        self.assertTrue(np.allclose(result_1234["precision"], np.array([0.25, 0.66666667])))
        self.assertTrue(np.allclose(result_1234["recall"], np.array([0.33333333, 0.57142857])))

    def test_f1_score(self):
        values = np.array([0, 0.25, 0.5, 1.0])
        precision = np.kron(np.ones((4,)), values)
        recall = np.kron(values, np.ones((4,)))
        expected_f1 = np.array([np.nan,      0.0,      0.0, 0.0,
                                   0.0,     0.25, 0.33333333, 0.4,
                                   0.0, 0.33333333,      0.5, 0.66666667,
                                   0.0,      0.4, 0.66666667, 1.0])

        print(expected_f1)
        print(f1_score(precision, recall))
        self.assertTrue(np.allclose(f1_score(precision, recall), expected_f1, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
