"""
    Controller de l'API: gestion des routes et des requêtes
"""
import flask
import pandas as pd
import xgboost as xgb
from .preprocess import cleaning


def home() -> str:
    """
    retourne la page d'accueil de l'API
    """
    return "<h1>API</h1><p>Bienvenue sur l'api flask du projet de \
            prédiction de risque de défaut sur crédits immobiliers.</p>"


def predire_credit(
    identifiant: int, dataframe: pd.DataFrame, model: xgb.XGBClassifier
) -> flask.Response:
    """
    retourne le risque de défaut d'un client selon son id
    """
    data = cleaning(dataframe)

    prediction_data = data[data["ID"] == identifiant].iloc[:, :-2]
    prediction_array = prediction_data.values.reshape(1, -1)

    print("Nouvelle Prédiction : \n", prediction_data)
    print("array : \n", prediction_array)

    prediction = model.predict(prediction_array)
    proba = model.predict_proba(prediction_array)

    dict_pred = {
        "individual_data": prediction_data.to_json(),
        "prediction": int(prediction),
        "proba": float(proba[0][0]),
    }

    print("Nouvelle Prédiction : \n", dict_pred)

    return flask.jsonify(dict_pred)
