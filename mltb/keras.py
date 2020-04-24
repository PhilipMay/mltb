"""Keras tools."""
from enum import Enum

import sklearn.metrics
import numpy as np
import tensorflow.keras as keras

from . import metrics as metrics_utils


class DefaultMetrics:

    @staticmethod
    def val_roc_auc(y_true, y_pred, pos_label):
        return sklearn.metrics.roc_auc_score(y_true, y_pred)

    @staticmethod
    def val_average_precision(y_true, y_pred, pos_label):
        return sklearn.metrics.average_precision_score(y_true, y_pred, pos_label=pos_label)

    @staticmethod
    def val_f1(y_true, y_pred, pos_label):
        round_y_pred = np.rint(y_pred)
        return sklearn.metrics.f1_score(y_true, round_y_pred, pos_label=pos_label)

    @staticmethod
    def val_acc(y_true, y_pred, pos_label):
        round_y_pred = np.rint(y_pred)
        return sklearn.metrics.accuracy_score(y_true, round_y_pred)

    @staticmethod
    def val_mcc(y_true, y_pred, pos_label):
        round_y_pred = np.rint(y_pred)
        return sklearn.metrics.matthews_corrcoef(y_true, round_y_pred)


DEFAULT_METRICS_BY_NAME = {function_name: getattr(DefaultMetrics, function_name) for function_name in
                           dir(DefaultMetrics) if not function_name.startswith('__')}


class UnsupportedMetrics(ValueError):
    def __init__(self, metric_names):
        super().__init__('Unsupported metrics: {}'.format(','.join(metric_names)))


class BinaryClassifierMetricsCallback(keras.callbacks.Callback):
    """Keras callback to calculate metrics of a binary classifier for each epoch.

    Attributes
    ----------
    val_data
        The validation data.
    val_labels
        The validation labels.
    pos_label : int, optional
        The positive label number. The default is 1.
    metrics : List[Union[str, Callable[[List[float], List[float], int], float]]], optional
        The list of metrics to compute. Defaults metrics:
         - val_roc_auc
         - val_average_precision
         - val_f1
         - val_acc
         - val_best_f1
         - val_best_f1_threshold
    """

    def __init__(self, val_data, val_labels, pos_label=1, metrics=None):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.pos_label = pos_label

        self.metrics = metrics or ['val_roc_auc', 'val_average_precision', 'val_f1', 'val_acc']
        self.__validate_metrics(self.metrics)
        self.metric_functions = self.__convert_metrics_to_functions(self.metrics)

    def __is_valid_metric(self, metric):
        valid_metric_names = DEFAULT_METRICS_BY_NAME.keys()
        if callable(metric):
            return True
        if isinstance(metric, str) and metric in valid_metric_names:
            return True
        return False

    def __validate_metrics(self, metrics):
        if not metrics or not isinstance(metrics, list):
            raise ValueError('Invalid metric list. It must be a list of custom metrics or str.')

        invalid_metrics = list(filter(lambda x: not self.__is_valid_metric(x), metrics))

        if len(invalid_metrics) > 0:
            raise UnsupportedMetrics(list(map(lambda x: str(x), invalid_metrics)))

    def __convert_metrics_to_functions(self, metrics):
        return list(map(lambda x: DEFAULT_METRICS_BY_NAME[x] if isinstance(x, str) else x, metrics))

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        y_pred = self.model.predict(self.val_data)
        y_true, pos_label = self.val_labels, self.pos_label

        for metric_function in self.metric_functions:
            metric_name = metric_function.__name__
            logs[metric_name] = metric_function(y_true, y_pred, pos_label)

        # DEPRECATED: Those metrics should be replaced by custom metrics
        val_best_f1, val_best_f1_threshold = metrics_utils.best_f1_score(
            y_true, y_pred, pos_label)
        logs["val_best_f1"] = val_best_f1
        logs["val_best_f1_threshold"] = val_best_f1_threshold
