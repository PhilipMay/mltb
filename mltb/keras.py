"""Keras tools."""
import sklearn.metrics
import numpy as np
import tensorflow.keras as keras

from . import metrics


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
    """
    def __init__(self, val_data, val_labels, pos_label=1):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.pos_label = pos_label

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        predict_results = self.model.predict(self.val_data)

        round_predict_results = np.rint(predict_results)

        val_roc_auc = sklearn.metrics.roc_auc_score(self.val_labels,
                                                predict_results)
        logs["val_roc_auc"] = val_roc_auc

        val_average_precision = sklearn.metrics.average_precision_score(
                self.val_labels, predict_results,
                pos_label=self.pos_label)

        logs['val_average_precision'] = val_average_precision

        val_f1 = sklearn.metrics.f1_score(self.val_labels, round_predict_results,
                                      pos_label=self.pos_label)
        logs["val_f1"] = val_f1

        accuracy = sklearn.metrics.accuracy_score(self.val_labels,
                                                  round_predict_results)
        logs["val_acc"] = accuracy

        val_best_f1, val_best_f1_threshold = metrics.best_f1_score(
                self.val_labels, predict_results, self.pos_label)
        logs["val_best_f1"] = val_best_f1
        logs["val_best_f1_threshold"] = val_best_f1_threshold
