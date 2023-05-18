import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import mlflow
import mlflow.sklearn
from mlflow.keras import log_model
import logging
from IPL_Utils import *

import matplotlib.pyplot as plt
def view_loss(history):
  plt.clf()
  plt.semilogy(history.history['loss'], label="train_loss")
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend()
  return plt

def generate_confusion_matrix(cm, classNames, labels):
    plt.figure(figsize=(12, 12))
    ax = plt.subplot()
    df_cm = pd.DataFrame(cm)
    sns.heatmap(df_cm, annot=True, ax=ax, cmap="Blues", fmt='g',
                annot_kws={"fontsize": 13, "style": "italic", "weight": "bold"},
                square=True, linewidths=2, linecolor="black");  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels', color='black', weight="bold");
    ax.set_ylabel('True labels', color='black', weight="bold");
    ax.set_title('Confusion Matrix  \n', weight="bold");
    ax.xaxis.set_ticklabels(classNames, rotation=90, fontsize="12", va="top", color='black', weight="bold");
    ax.yaxis.set_ticklabels(classNames, rotation=0, fontsize="12", va="center", color='black', weight="bold");
    return ax

def generate_roc_curve(classNames, testing_predictions, y_test):
    n_classes = 10
    n_colors = ['green', 'red', 'purple', 'orange', 'black',
                'brown','pink', 'magenta', 'yellow', 'orange']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], testing_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7, 7))
    for i in range(0, n_classes):
        plt.plot(fpr[i], tpr[i], color=n_colors[i], label=classNames[i])
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', weight="bold")
    plt.ylabel('True Positive Rate', weight="bold")
    plt.title('ROC Curve \n', weight="bold")
    plt.legend(loc="lower right")

    return plt


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    x , y, classNames = generate_training_testing_data()
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    mlflow.end_run()
    experiment_name = 'IPL_MS_DS_Demo'
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # lr_model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
        lr_model = RandomForestClassifier(random_state=42, n_estimators=8, max_depth=11)
        lr_model.fit(X_train, y_train)

        predictions = lr_model.predict(X_test)
        model_accuracy = accuracy_score(predictions, y_test)
        model_f1_score = f1_score(predictions, y_test, average='macro')
        model_precision= precision_score(predictions, y_test, average='macro')
        model_recall = recall_score( predictions, y_test, average='weighted')

        mlflow.log_metric("Accuracy", model_accuracy)
        mlflow.log_metric("Precision", model_precision)
        mlflow.log_metric("F1-Score", model_f1_score)
        mlflow.log_metric("Recall", model_recall)

        actual_values = [np.argmax(t) for t in y_test]
        test_predictions = [np.argmax(t) for t in predictions]
        cm = confusion_matrix(test_predictions, actual_values)
        labels = [0, 1, 2,3,4,5,6,7,8,9]

        fig = generate_confusion_matrix(cm, classNames, labels)
        fig.figure.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        roc_fig = generate_roc_curve(classNames, predictions, y_test)
        roc_fig.savefig('ROC_curve.png')
        mlflow.log_artifact('ROC_curve.png')


    mlflow.end_run()

