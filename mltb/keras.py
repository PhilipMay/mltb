"""Keras tools."""
import sklearn.metrics
import numpy
import keras

from . import metrics
from . import tools 

class BinaryClassifierMetricsCallback(keras.callbacks.Callback):
    """Keras callback to calculate metrics of a binary classifier for each epoch.
    
    # Arguments
        val_data: The validation data.
        val_labels: Validation labels.
        pos_label: Positive label (default is `1`).

    # Properties
        metrics: Dictionary with list of metric data for each epoch.
    """
    
    def __init__(self, val_data, val_labels, pos_label=1):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.pos_label = pos_label
        self.metrics = {}
    
    def on_epoch_end(self, batch, logs={}):       
        predict_results = self.model.predict(self.val_data)
        
        round_predict_results = numpy.rint(predict_results)
        
        roc_auc = sklearn.metrics.roc_auc_score(self.val_labels, predict_results)
        self.metrics.setdefault("roc_auc", []).append(roc_auc)
        
        f1 = sklearn.metrics.f1_score(self.val_labels, round_predict_results)
        self.metrics.setdefault("f1", []).append(f1)

        accuracy = sklearn.metrics.accuracy_score(self.val_labels, round_predict_results)
        self.metrics.setdefault("accuracy", []).append(accuracy)

        best_f1, best_f1_threshold = metrics.best_f1_score(self.val_labels, predict_results, self.pos_label)
        self.metrics.setdefault("best_f1", []).append(best_f1)
        self.metrics.setdefault("best_f1_threshold", []).append(best_f1_threshold)
