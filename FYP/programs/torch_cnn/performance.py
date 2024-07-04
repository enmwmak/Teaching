'''
Compute performance matrices.

Weight Accuracy (WA): Sum of correct / No. of predictions

Unweighted Accuracy (UA) also called Unweighted Average Recall (UAR): It is 
the sum of class-wise accuracy (recall) divided by the number of classes

'''
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_accuracy(true_labels, pred_labels):
    wa = accuracy_score(true_labels, pred_labels)
    ua = recall_score(true_labels, pred_labels, average='macro')
    return wa, ua


if __name__ == "__main__":
    true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    pred = [1, 1, 1, 1, 2, 3, 3, 2, 2, 2, 3, 3, 1, 1]
    wa, ua = get_accuracy(true, pred)
    print(f'WA = {wa*100:.2f}%')
    print(f'UA = {ua*100:.2f}%')

    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3])
    disp.plot()
    plt.show()


    