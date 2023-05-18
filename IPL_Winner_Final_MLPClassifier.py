import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
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
    ax.set_title('Confusion Matrix for MLPClassifier  \n', weight="bold");
    ax.xaxis.set_ticklabels(classNames, rotation=90, fontsize="12", va="top", color='black', weight="bold");
    ax.yaxis.set_ticklabels(classNames, rotation=0, fontsize="12", va="center", color='black', weight="bold");
    return ax

def generate_roc_curve()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    x , y, classNames = generate_training_testing_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mlflow.end_run()
    experiment_name = 'IPL_MS_DS_Demo'
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlp_model = MLPClassifier(random_state=2)
        mlp_model.fit(X_train, y_train)

        predictions = mlp_model.predict(X_test)
        model_accuracy = accuracy_score(predictions, y_test)
        model_f1_score = f1_score(predictions, y_test, average='macro')
        model_precision= precision_score(predictions, y_test, average='macro')
        model_recall = recall_score( predictions, y_test, average='weighted')

        mlflow.log_metric("Accuracy", model_accuracy)
        mlflow.log_metric("Precision", model_precision)
        mlflow.log_metric("F1-Score", model_f1_score)
        mlflow.log_metric("Recall", model_recall)

        cm = confusion_matrix(predictions, y_test)
        labels = [0, 1, 2,3,4,5,6,7,8,9]

        fig = generate_confusion_matrix(cm, classNames, labels)

        # save loss picture, use log_artifact() to track it in MLFLow UI
        fig.figure.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

    mlflow.end_run()

