"""
fichier principal de l'app flask (à lancer pour lancer l'app)
définit les routes et appelle les fonctions associées

api qui renvoie une prédiction de défaut de crédit pour un client de la
base de données (ID) et en fonction de ses caractéristiques
"""
import pickle
import flask
from utils import home, predire_credit

import pandas as pd

# Load model
PATH_MODEL = "model/model_final.pickle.dat"
with open(PATH_MODEL, "rb") as file:
    model = pickle.load(file)

# Load Dataframe
# (temp restricted dataset en attendant psql database)
PATH_DF = "model/restricted_dataset"
df = pd.read_csv(PATH_DF)
df = df.drop(["Unnamed: 0"], axis=1)

# setup app et routes
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route("/", methods=["GET"])
def index():
    """
    route de la page d'accueil
    """
    return home()


@app.route("/client/<int:ID>")
def client(identifiant):
    """
    route prediction pour un client
    """
    return predire_credit(identifiant, df, model)


if __name__ == "__main__":
    app.run()
