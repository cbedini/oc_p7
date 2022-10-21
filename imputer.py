#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:06:04 2022

@author: mcBedini
"""

import pandas as pd
import time
from contextlib import contextmanager

from sklearn.impute import KNNImputer


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


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




imputer = KNNImputer()

Xnbs = pd.DataFrame(imputer.fit_transform(X),
                           columns=X.columns,
                           index=X.index)
print('Saving')
Xnbs.to_csv('imputed.csv')


