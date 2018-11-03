import sklearn
import numpy
import keras

from . import metrics
from . import tools 

class BinaryClassifierMetricsCallback(keras.callbacks.Callback):
    """Keras callback to calculate metrics of a binary classifier for each epoch."""
    
    def __init__(self, val_data, val_labels, pos_label):
        self.val_data = val_data
        self.val_labels = val_labels
        self.pos_label = pos_label
        self.hist = {}
    
    def on_epoch_end(self, batch, logs={}):        
        predict_results = self.model.predict(self.val_data)
        
        round_predict_results = numpy.rint(predict_results)
        
        roc_auc = sklearn.metrics.roc_auc_score(self.val_labels, predict_results)
        tools.append_to_dict(self.hist, "roc_auc", roc_auc)
        
        f1 = sklearn.metrics.f1_score(self.val_labels, round_predict_results)
        tools.append_to_dict(self.hist, "f1", f1)

        accuracy = sklearn.metrics.accuracy_score(self.val_labels, round_predict_results)
        tools.append_to_dict(self.hist, "accuracy", accuracy)

        best_f1, best_f1_threshold = metrics.best_f1_score(self.val_labels, predict_results, self.pos_label)
        tools.append_to_dict(self.hist, "best_f1", best_f1)
        tools.append_to_dict(self.hist, "best_f1_threshold", best_f1_threshold)
