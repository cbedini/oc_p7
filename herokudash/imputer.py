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
    savedataframe = "dashsample.csv" if debug else "apidataframe.csv"
    df = pd.read_csv(savedataframe)
    print("Reading",savedataframe)
    print("Shape",df.shape)
    return df


def prepare_train_df(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print('Test', test_df.columns)
    test_feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    apptest, X = train_df[test_feats], train_df[test_feats]
    print('Training dataframe shape:', X.shape)
    print('Test shape:', apptest.shape)
    print('X', X.columns)
    return X,  apptest

df=readdf(True)
X, apptest = prepare_train_df(df)



print('Imputing')
imputer = KNNImputer()

Xnbs = pd.DataFrame(imputer.fit_transform(X),
                           columns=X.columns,
                           index=X.index)
print('Saving')
print('Xnbs',Xnbs.shape)
print('Xnbs',Xnbs.columns)

Xnbs.to_csv('imputed.csv')


