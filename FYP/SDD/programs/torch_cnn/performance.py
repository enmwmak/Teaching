'''
Compute performance matrices.

Weight Accuracy (WA): Sum of correct / No. of predictions

Unweighted Accuracy (UA) also called Unweighted Average Recall (UAR): It is 
the sum of class-wise accuracy (recall) divided by the number of classes

For the explanations, precision, recall, and F1 score, see 
see https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

'''
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

def get_accuracy(true_labels, pred_labels):
    wa = accuracy_score(true_labels, pred_labels)
    ua = recall_score(true_labels, pred_labels, average='macro')
    return wa, ua

def comp_loso_cv_f1(results):
    tp = tn = fp = fn = 0
    for result in results:
        tp += result["tp"] ; tn += result["tn"] ; fp += result["fp"] ; fn += result["fn"]
    f1 = 2*tp/(2*tp + fp + fn)    
    return f1

def get_precision(tp, fp):
    '''
    Precision = No. of true positives / No. of samples predicted positive
    '''
    pre = tp/(tp + fp)
    return pre

def get_recall(tp, fn):
    '''
    Recall = No. of true positives / No. of actual positives
    Recall is also called sensitivity or true positive rate
    '''
    rec = tp/(tp + fn)
    return rec

def get_f1_score(tp, fp, fn):
    '''
    F1 = (2 * Precision * Recall)/(Precision + Recall) 
       = 2*tp/(2*tp + fp + fn)
    '''
    f1 = 0 if (2*tp + fp + fn == 0) else 2*tp/(2*tp + fp + fn)
    return f1

def get_dec_count(true, pred, pos=1, neg=0): 
    tp = np.sum(np.logical_and(pred==pos, true==pos))
    tn = np.sum(np.logical_and(pred==neg, true==neg))
    fp = np.sum(np.logical_and(pred==pos, true==neg))
    fn = np.sum(np.logical_and(pred==neg, true==pos))
    return tp, tn, fp, fn


if __name__ == "__main__":
    '''
    Compute the chance level of Precision, Recall, F1 score, and Accuracy for
    a binary classifier that produces random decision following a Bernoulli distribution
    with a given probability for the positive class
    '''

    N = 100000               # No. of samples (0 or 1) to generate
    prior1 = 0.28           # Prior prob for positive class (decision = 1)

    true = bernoulli.rvs(prior1, size=N)
    _, counts = np.unique(true, return_counts=True)
    print(f"No. of '0': {counts[0]}, No. of '1': {counts[1]}")

    pred = bernoulli.rvs(prior1, size=N)            # Prediction of the random classifier
    _, counts = np.unique(pred, return_counts=True)
    print(f"No. of '0': {counts[0]}, No. of '1': {counts[1]}")

    tp, tn, fp, fn = get_dec_count(true, pred, pos=1, neg=0)
    f1 = get_f1_score(tp, fp, fn)
    print(f'F1 = {f1:0.4f}')
    
    wa, ua = get_accuracy(true, pred)
    print(f'WA = {wa*100:.2f}%')
    print(f'UA = {ua*100:.2f}%')

    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.show()

    