"""Keras tools."""
import sklearn.metrics
import numpy as np
import keras
import tensorflow as tf

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

    def on_epoch_end(self, batch, logs={}):
        logs = logs or {}
        predict_results = self.model.predict(self.val_data)

        round_predict_results = np.rint(predict_results)

        roc_auc = sklearn.metrics.roc_auc_score(self.val_labels, predict_results)
        logs["roc_auc"] = roc_auc

        f1 = sklearn.metrics.f1_score(self.val_labels, round_predict_results)
        logs["f1"] = f1

        accuracy = sklearn.metrics.accuracy_score(self.val_labels, round_predict_results)
        logs["accuracy"] = accuracy

        best_f1, best_f1_threshold = metrics.best_f1_score(self.val_labels, predict_results, self.pos_label)
        logs["best_f1"] = best_f1
        logs["best_f1_threshold"] = best_f1_threshold


def set_gpu_mem_growth():
    """Allocate only as much GPU memory as needed for Keras.

    Based on runtime allocations: it starts out allocating very little memory,
    and as Sessions get run and more GPU memory is needed, we extend the GPU
    memory region needed by the TensorFlow process. Note that we do not release
    memory, since that can lead to even worse memory fragmentation.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
