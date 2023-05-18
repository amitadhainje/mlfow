import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import mlflow
import mlflow.sklearn
from mlflow.keras import log_model
import logging
from IPL_Utils import *

import matplotlib.pyplot as plt
def plot_accuracy_history(history):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    epochs = range(len(acc))
    plt.figure(figsize=(7, 7))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    return plt

def plot_loss_history(history):
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(figsize=(7, 7))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    return plt

def ann_model():
  return Sequential([Dense(20, input_dim=9, activation='relu'),
                    Dense(10, activation='softmax')])

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
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    n_classes = 10
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    # dl_model = ann_model()
    # print (dl_model.summary())

    # supply training algo settings (optimizer, evaluation metrics)
    compile_kwargs = {
        "optimizer": 'adam',
        "loss": 'categorical_crossentropy',
        "metrics": "accuracy",
    }
    # dl_model.compile(**compile_kwargs)

    # supply training data and hyper parameters
    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        'validation_data': (X_test, y_test),
        "epochs": 10,
        "verbose": 2
    }

    # model fit
    # history = dl_model.fit(**fit_kwargs)


    mlflow.end_run()
    experiment_name = 'IPL_MS_DS_Demo'
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        dl_model = ann_model()
        dl_model.compile(**compile_kwargs)
        history = dl_model.fit(**fit_kwargs)

        for param_key, param_value in {**compile_kwargs, **fit_kwargs}.items():
            if param_key not in ['x', 'y']:
                # use log_param() to track hyper-parameters (except training dataset x,y)
                if (param_key != "validation_data"):
                    mlflow.log_param(param_key, param_value)

        for key, values in history.history.items():
            for i, v in enumerate(values):
                # use log_metric() to track evaluation metrics
                mlflow.log_metric(key.title(), v, step=i)

        for i, layer in enumerate(dl_model.layers):
            # use log_param() to track model.layer (details of each CNN layer)
            mlflow.log_param(f'hidden_layer_{i}_units', layer.output_shape)

        # use log_model() to track output Keras model (accessible through the MLFlow UI)
        # log_model(dl_model, 'keras_model')

        predictions = dl_model.predict(X_test)
        t_predictions = dl_model.predict(X_train)

        actual_values = [np.argmax(t) for t in y_test]
        test_predictions = [np.argmax(t) for t in predictions]

        actual_train_vals = [np.argmax(t) for  t in y_train]
        train_predictions = [np.argmax(t) for t in t_predictions]

        overall_test_accuracy = accuracy_score(test_predictions, actual_values)
        overall_train_accuracy = accuracy_score(train_predictions, actual_train_vals)
        model_f1_score = f1_score(test_predictions, actual_values, average='macro')
        model_precision = precision_score(test_predictions, actual_values, average='macro')
        model_recall = recall_score(test_predictions, actual_values, average='weighted')

        mlflow.log_metric("Overall Testing Accuracy", overall_test_accuracy)
        mlflow.log_metric("Overall Training Accuracy", overall_train_accuracy)
        mlflow.log_metric("Precision", model_precision)
        mlflow.log_metric("F1-Score", model_f1_score)
        mlflow.log_metric("Recall", model_recall)

        actual_values = [np.argmax(t) for t in y_test]
        test_predictions = [np.argmax(t) for t in predictions]
        cm = confusion_matrix(test_predictions, actual_values)
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        fig = generate_confusion_matrix(cm, classNames, labels)
        fig.figure.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        roc_fig = generate_roc_curve(classNames, predictions, y_test)
        roc_fig.savefig('ROC_curve.png')
        mlflow.log_artifact('ROC_curve.png')

        ##(3) sketch loss
        loss_fig = plot_loss_history(history)
        loss_fig.savefig('train-validation-loss.png')
        mlflow.log_artifact('train-validation-loss.png')

        acc_fig = plot_accuracy_history(history)
        acc_fig.savefig('train-validation-accuracy.png')
        mlflow.log_artifact('train-validation-accuracy.png')

        # return MLFLow run context
        # this run variable contains the runID and experimentID that is essential to
        # retrieving our training outcomes programatically
        # mlp_model = MLPClassifier(random_state=2)
        # mlp_model.fit(X_train, y_train)
        #
        # predictions = mlp_model.predict(X_test)
        # model_accuracy = accuracy_score(predictions, y_test)
        # print ("Accuracy ===", model_accuracy)
        # mlflow.log_metric("Accuracy", model_accuracy)


    mlflow.end_run()

