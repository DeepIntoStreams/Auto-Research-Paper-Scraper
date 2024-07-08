import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics


def draw_confusion_matrix(true_values, predicted_values, labels_on_axes, title=""):
    """
    Creates and displays a confusion matrix with annotations.
    """
    confusion_matrix = metrics.confusion_matrix(true_values, predicted_values)
    cm_sum = np.sum(confusion_matrix, axis=1).reshape(-1, 1)
    cm_percentage = confusion_matrix / cm_sum.astype(float) * 100

    annot = np.empty_like(confusion_matrix).astype(str)
    n_rows, n_cols = confusion_matrix.shape

    for i in range(n_rows):
        for j in range(n_cols):
            p = cm_percentage[i, j]
            c = confusion_matrix[i, j]
            if i == j:
                s = cm_sum[i][0]
                annot[i, j] = f'{p:.1f}%\n{c}/{s}'
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{p:.1f}%\n{c}'

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=annot, fmt='', ax=ax, cmap="coolwarm", linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix {title}")
    ax.xaxis.set_ticklabels(labels_on_axes)
    ax.yaxis.set_ticklabels(labels_on_axes)
    return
