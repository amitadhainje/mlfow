import os
import warnings
import sys

import pandas as pd
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
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

def ann_model():
  return Sequential([Dense(20, input_dim=9, activation='relu'),
                    Dense(10, activation='softmax')])

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    x , y = generate_training_testing_data()
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    n_classes = 10
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    dl_model = ann_model()
    print (dl_model.summary())

    # supply training algo settings (optimizer, evaluation metrics)
    compile_kwargs = {
        "optimizer": 'adam',
        "loss": 'categorical_crossentropy',
        "metrics": "accuracy",
    }
    dl_model.compile(**compile_kwargs)

    # supply training data and hyper parameters
    fit_kwargs = {
        "x": X_train_scaled,
        "y": y_train,
        "epochs": 5,
        "verbose": 2
    }

    # model fit
    history = dl_model.fit(**fit_kwargs)

    # display training epoch results
    for key, values in history.history.items():
        for i, v in enumerate(values):
            print(f'{key}: {v} (Step: {i})')


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
                mlflow.log_param(param_key, param_value)

        for key, values in history.history.items():
            for i, v in enumerate(values):
                # use log_metric() to track evaluation metrics
                mlflow.log_metric(key, v, step=i)

        for i, layer in enumerate(dl_model.layers):
            # use log_param() to track model.layer (details of each CNN layer)
            mlflow.log_param(f'hidden_layer_{i}_units', layer.output_shape)

        # use log_model() to track output Keras model (accessible through the MLFlow UI)
        # log_model(dl_model, 'keras_model')

        ##(3) sketch loss
        fig = view_loss(history)

        # save loss picture, use log_artifact() to track it in MLFLow UI
        fig.savefig('train-validation-loss.png')
        mlflow.log_artifact('train-validation-loss.png')

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

