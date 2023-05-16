import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    matches_df = pd.read_csv("IPL_Matches_2008_2022_updated.csv")
    balls_df = pd.read_csv("IPL_Ball_by_Ball_2008_2022_updated.csv")

    matches_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    balls_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    matches_df.columns = [x.lower() for x in matches_df.columns]

    balls_df.columns = [x.lower() for x in balls_df.columns]
    print(matches_df.columns)
    print(balls_df.columns)
    print(matches_df.shape)
    print(balls_df.shape)

    matches_df['season'] = matches_df['season'].str.replace('2020/21', '2021')
    matches_df['season'] = matches_df['season'].str.replace('2009/10', '2010')
    matches_df['season'] = matches_df['season'].str.replace('2007/08', '2008')
    print (matches_df['season'].value_counts())

    print("Removing the matches with No-Results")
    no_results_id = list(set(matches_df[matches_df['wonby'] == "NoResults"]['id'].values))
    print(no_results_id)
    matches_df = matches_df[matches_df['wonby'] != "NoResults"]
    balls_df = balls_df[~balls_df['id'].isin(no_results_id)]
    print(matches_df.shape)
    print(balls_df.shape)

    current_ipl_teams = ['Chennai Super Kings',
                         'Delhi Capitals',
                         'Mumbai Indians',
                         'Gujarat Titans',
                         'Kolkata Knight Riders',
                         'Lucknow Super Giants',
                         'Punjab Kings',
                         'Rajasthan Royals',
                         'Royal Challengers Bangalore',
                         'Sunrisers Hyderabad']

    team_hash = {}
    for x in range(0, len(current_ipl_teams)):
        team_hash[current_ipl_teams[x]] = x + 1
    print(team_hash)

    matches_df['team1'] = matches_df['team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
    matches_df['team1'] = matches_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    matches_df['team1'] = matches_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')

    matches_df['team2'] = matches_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    matches_df['team2'] = matches_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    matches_df['team2'] = matches_df['team2'].str.replace('Kings XI Punjab', 'Punjab Kings')

    matches_df['winningteam'] = matches_df['winningteam'].str.replace('Kings XI Punjab', 'Punjab Kings')
    matches_df['winningteam'] = matches_df['winningteam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    matches_df['winningteam'] = matches_df['winningteam'].str.replace('Delhi Daredevils', 'Delhi Capitals')

    matches_df['tosswinner'] = matches_df['tosswinner'].str.replace('Kings XI Punjab', 'Punjab Kings')
    matches_df['tosswinner'] = matches_df['tosswinner'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    matches_df['tosswinner'] = matches_df['tosswinner'].str.replace('Delhi Daredevils', 'Delhi Capitals')

    balls_df['winningteam'] = balls_df['winningteam'].str.replace('Kings XI Punjab', 'Punjab Kings')
    balls_df['battingteam'] = balls_df['battingteam'].str.replace('Kings XI Punjab', 'Punjab Kings')
    balls_df['bowlingteam'] = balls_df['bowlingteam'].str.replace('Kings XI Punjab', 'Punjab Kings')

    print(matches_df.shape)
    print(balls_df.shape)
    final_df = matches_df.merge(balls_df, how='inner', on='id')
    print (final_df.shape)

    var_mod = ['tossdecision', 'city_x', 'venue', 'umpire1', 'wonby']
    le = LabelEncoder()
    for i in var_mod:
        final_df[i] = le.fit_transform(final_df[i])

    final_df = final_df.replace({'team1': team_hash})
    final_df = final_df.replace({'team2': team_hash})
    final_df = final_df.replace({'tosswinner': team_hash})
    final_df = final_df.replace({'winningteam_x': team_hash})
    print (final_df.head())

    x = final_df[['team1', 'team2', 'tossdecision', 'tosswinner', 'city_x', 'venue', 'season', 'wonby', 'umpire1']]
    y = final_df[['winningteam_x']]
    print (x.head())

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)
    experiment_name = 'IPL_MS_DS_Demo'
    mlflow.set_experiment(experiment_name)

    mlflow.end_run()
    artifact_path = mlflow.get_artifact_uri()
    uri = mlflow.tracking.get_tracking_uri()
    print(artifact_path)
    print(uri)


    mlflow.end_run()
    with mlflow.start_run():
        # mlflow.set_experiment(experiment_name="IPL_MS_DS_Demo")
        lr_model = LogisticRegression()

        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)


        print (accuracy_score(y_test, y_pred))

        mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr_model, "model", registered_model_name="LogisticRegression")
        else:
            mlflow.sklearn.log_model(lr_model, "model")
    mlflow.end_run()

