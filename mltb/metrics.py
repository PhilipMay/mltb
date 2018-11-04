"""Metrics."""
import sklearn

def f1_from_roc(fpr, tpr, pos, neg):
    """Calculate f1 score from roc values.
    
    Args:
        fpr (float): false positive rate
        tpr (float): true positive rate
        pos (int): number of positive labels
        neg (int): number of negative labels
        
    Returns:
        The f1 score.
    """
    fp = fpr * neg
    fn = (1 - tpr) * pos
    tp = pos - fn
    f1 = tp / (tp + ((fn + fp) / 2))
    return f1

def pos_neg(labels, pos_label):
    pos = sum(label == pos_label for label in labels)
    neg = sum(label != pos_label for label in labels)
    return pos, neg        
        
def best_f1_score(labels, predictions, pos_label):
    """Calculate best f1 score with its threshold.
    
    Args:
        labels: 
    
    Returns:
        
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions, pos_label=pos_label)
    pos, neg = pos_neg(labels, 1)
    
    best_f1 = -1
    best_f1_threshold = -1
    
    for fpr_value, tpr_value, t in zip(fpr, tpr, thresholds):
        f1 = f1_from_roc(fpr_value, tpr_value, pos, neg)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = t
    return best_f1, best_f1_threshold        
        
