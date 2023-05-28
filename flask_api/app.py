import flask
import json
import requests
from utils import cleaning, home, predire_credit

import pandas as pd
import xgboost as xgb
import pickle

# Load model
path_model = 'model/model_final.pickle.dat'
model = pickle.load(open(path_model,'rb'))

# Load Dataframe 
# (temp restricted dataset en attendant psql database)
path_df = 'model/restricted_dataset'
df = pd.read_csv(path_df)
df = df.drop(['Unnamed: 0'], axis = 1)

# setup app et routes
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def index():
 return home()

@app.route('/client/<int:ID>')
def client(ID):
    return predire_credit(ID, df, model)

if __name__ == '__main__':
    app.run()
