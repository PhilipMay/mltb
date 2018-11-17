"""Keras tools."""
import sklearn.metrics
import numpy as np
import keras

from . import metrics
from . import tools 

class BinaryClassifierMetricsCallback(keras.callbacks.Callback):
    """Keras callback to calculate metrics of a binary classifier for each epoch.
    
    # Arguments
        val_data: The validation data.
        val_labels: Validation labels.
        pos_label: Positive label (default is `1`).
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
