"""LightGBM tools."""

from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def multi_class_f1_score_factory(num_classes, average):
    """Factory for LightGBM multi class F1-score function.

    Parameters
    ----------
    num_classes : int
        Number of classes to classify.
    average : string, 'micro' or 'macro'
        ``'micro'``
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``
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

    def multi_class_f1_score(y_pred, data):
        y_true = data.get_label()
        y_pred = y_pred.reshape((num_classes, -1))
        y_pred = np.transpose(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        return eval_name, f1_score(y_true, y_pred, average=average), True

    return multi_class_f1_score


def binary_class_f1_score(y_pred, data):
    """LightGBM binary class F1-score function.

    Parameters
    ----------
    y_pred
        LightGBM predictions.
    data
        LightGBM ``'Dataset'``.

    Returns
    -------
    (eval_name, eval_result, is_higher_better)
        ``'eval_name'`` :  string
            is always 'f1' - the name of the metric
        ``'eval_result'`` : float
            is the result of the metric
         ``'is_higher_better'`` : bool
            is always 'True' because higher F1 score is better

    See Also
    --------
    * `sklearn.metrics.f1_score: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`
    * `LightGBM Training API: <https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api>`
    """
    y_true = data.get_label()
    y_pred = np.round(y_pred)
    return 'f1', f1_score(y_true, y_pred), True


def multi_class_accuracy_score_factory(num_classes):
    """Factory for LightGBM multi class accuracy-score function.

    Parameters
    ----------
    num_classes : int
        Number of classes to classify.

    See Also
    --------
    * `sklearn.metrics.accuracy_score: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`
    * `LightGBM Training API: <https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api>`
    """

    def multi_class_accuracy_score(y_pred, data):
        y_true = data.get_label()
        y_pred = y_pred.reshape((num_classes, -1))
        y_pred = np.transpose(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        return 'accuracy', accuracy_score(y_true, y_pred), True

    return multi_class_accuracy_score


def binary_class_accuracy_score(y_pred, data):
    """LightGBM binary class accuracy-score function.

    Parameters
    ----------
    y_pred
        LightGBM predictions.
    data
        LightGBM ``'Dataset'``.

    Returns
    -------
    (eval_name, eval_result, is_higher_better)
        ``'eval_name'`` : string
            is always 'accuracy' - the name of the metric
        ``'eval_result'`` : float
            is the result of the metric
         ``'is_higher_better'`` : bool
            is always 'True' because higher accuracy score is better

    See Also
    --------
    * `sklearn.metrics.accuracy_score: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`
    * `LightGBM Training API: <https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api>`
    """
    y_true = data.get_label()
    y_pred = np.round(y_pred)
    return 'accuracy', accuracy_score(y_true, y_pred), True
