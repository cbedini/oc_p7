#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:54:48 2022

@author: mcBedini
"""

from flask import Flask, request, jsonify
import joblib
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import json
import shap





app = Flask(__name__)
#heroku
clf = joblib.load('lightgbmodel.joblib')
explainer = shap.Explainer(clf)

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    data = request.get_json(force=True)
    reshaped_data=np.array(data).reshape(1, -1)
    targeted=clf.predict_proba(reshaped_data)
    response=json.dumps(targeted.tolist()[0])
    return response
   

@app.route('/shap', methods=['POST'])
def shap():
    data = request.get_json(force=True)
    reshaped_data=np.array(data).reshape(1, -1)
    shap_val = explainer.shap_values(reshaped_data)
    # valeurs pour le forceplot
    a_float=explainer.expected_value[0]
    an_array=shap_val[0]
    response=json.dumps([a_float,an_array.tolist()[0]])
    return response
    

@app.route("/")
def hello():
    return "Alive and kicking"    
 
    
if __name__ == '__main__':
     #clf = joblib.load('lightgbmodel.joblib')
     app.run(port=8080)



# https://towardsdatascience.com/a-flask-api-for-serving-scikit-learn-models-c8bcdaa41daa
# https://towardsdatascience.com/creating-restful-apis-using-flask-and-python-655bad51b24
# JSON https://www.fernandomc.com/posts/your-first-flask-api/

     