import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score

def ood_auc(label,socre):
    return roc_auc_score(label, socre)

def ood_aupr(label,score):
    return average_precision_score(label,score)

def fpr95(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    target = np.where(tpr >= 0.95)[0][0]
    return fpr[target]
