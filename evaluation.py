import numpy as np
import sklearn.metrics


def confusion_matrix(ground_truth, estimated) -> np.ndarray:
    """
    Returns a num_classes by num_classes array, where entry ij represents the
    number of pixels of class i predicted to belong to class j.
    """
    return sklearn.metrics.confusion_matrix(ground_truth.reshape(-1),
                                            estimated.reshape(-1))

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def precision_and_recall(confusion_mat):
    true_positives = np.diagonal(confusion_mat)

    confusion_mat_zero_diagonal = np.copy(confusion_mat)
    np.fill_diagonal(confusion_mat_zero_diagonal, val=0)
    false_negatives = np.sum(confusion_mat_zero_diagonal, axis=1)
    false_positives = np.sum(confusion_mat_zero_diagonal, axis=0)

    precision_by_class = true_positives / (true_positives + false_positives)
    recall_by_class = true_positives / (true_positives + false_negatives)

    return {
        "precision": precision_by_class,
        "recall": recall_by_class,
    }
