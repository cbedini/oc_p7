#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:54:48 2022

@author: mcBedini
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np


def readdf(debug=False):
    savedataframe = "debugdataframe.csv" if debug else "apidataframe.csv"
    df = pd.read_csv(savedataframe)
    print("Reading",savedataframe)
    print("Shape",df.shape)
    return df

def gender(X):
    X['CODE_GENDER'] = X['CODE_GENDER'].astype(str)

    X=X.replace({
        'CODE_GENDER': {
            '0': 'homme',
            '1': 'femme',        
        }
    })
    return X

def age(X):
    X['DAYS_BIRTH'] = X['DAYS_BIRTH']//365*-1
    return X

def cosmetics(df):
    df=gender(df)
    df=age(df)
    return df

def prepare_train_df(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    feats = [f for f in train_df.columns if f not in ['SK_ID_BUREAU','SK_ID_PREV','index']]
    test_feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    X, target, apptest = train_df[feats], train_df[['SK_ID_CURR','TARGET']], test_df[test_feats]
    print('Training dataframe shape:', X.shape)
    print('Test shape:', apptest.shape)
    cosmeticX = cosmetics(X)
    cosmetic_apptest= cosmetics(apptest)
    return X, cosmeticX, target, apptest, cosmetic_apptest

df=readdf()
X, cosmeticX, target, apptest, cosmetic_apptest = prepare_train_df(df)


# inputs




app = Flask(__name__)

@app.route('/predict_proba',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = clf.predict_proba([[np.array(data)]])
    output = prediction[0]
    return jsonify(output)

@app.route('/data')
def data():
     return X.iloc[:10,:10].to_dict('records')

    
 
    
 
@app.route("/")
def hello():
    return "Alive and kicking"    
 
    
if __name__ == '__main__':
     clf = joblib.load('lightgbmodel.joblib')
     app.run(port=8080)



# https://towardsdatascience.com/a-flask-api-for-serving-scikit-learn-models-c8bcdaa41daa
# https://towardsdatascience.com/creating-restful-apis-using-flask-and-python-655bad51b24
# JSON https://www.fernandomc.com/posts/your-first-flask-api/

     