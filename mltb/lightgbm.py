"""LightGBM tools."""

from sklearn.metrics import f1_score
import numpy as np


def lightgbm_multi_class_f1_score_factory(num_classes, average):
    """Factory for LightGBM multi class F1 Score function.

    Parameters
    ----------
    num_classes : int
        Number of classes to classify.
    average : string string, ['micro' or 'macro']
        ``'micro'`` :
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'`` :
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.

    See Also
    --------
    * `sklearn.metrics.f1_score: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`
    * `LightGBM Training API: <https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api>`
    """
    if average != 'macro' and average != 'micro':
        raise ValueError("average should be 'macro' or 'micro'")

    eval_name = 'f1_' + average

    def lightgbm_multi_class_f1_score(y_pred, data):
        y_true = data.get_label()
        y_pred = y_pred.reshape((num_classes, -1))
        y_pred = np.transpose(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        return eval_name, f1_score(y_true, y_pred, average=average), True

    return lightgbm_multi_class_f1_score
