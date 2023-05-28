import flask
import pandas as pd
import json
from .preprocess import cleaning

import requests

import xgboost as xgb
import pickle

def home() -> str:
    return "<h1>API</h1><p>Bienvenue sur l'api flask du projet de \
            prédiction de risque de défaut sur crédits immobiliers.</p>"

def predire_credit(ID: int, df: pd.DataFrame, model: xgb.XGBClassifier) -> flask.Response:
    '''
        retourne le risque de défaut d'un client selon son id
    '''
    data = cleaning(df)

    prediction_data = data[data['ID'] == ID].iloc[:,:-2]
    prediction_array = prediction_data.values.reshape(1,-1)

    print('Nouvelle Prédiction : \n', prediction_data)
    print('array : \n', prediction_array)

    prediction = model.predict(prediction_array)
    proba = model.predict_proba(prediction_array)

    dict_pred = {
            'individual_data': prediction_data.to_json(),
            'prediction': int(prediction),
            'proba': float(proba[0][0])
            }

    print('Nouvelle Prédiction : \n', dict_pred)

    return flask.jsonify(dict_pred)


