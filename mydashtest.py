#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 20:25:33 2022

@author: mcBedini
"""

import numpy as np
import pandas as pd
import time
import re
from contextlib import contextmanager

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq

from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import matplotlib.pyplot as plt, mpld3


import joblib
#import mlflow.sklearn

import shap

from textwrap import wrap
from flask import Flask
from operator import itemgetter
import os
import requests
import json

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#heroku
server=app.server

DATA='http://127.0.0.1:8080/data'


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

print('Loading model')
#Loading the saved model with joblib
reloaded = joblib.load('lightgbmodel.joblib')
print('Loaded')

# Outils

# Load explainer
#quick and dirty pour eviter Error https://stackoverflow.com/questions/71187653/the-passed-model-is-not-callable-and-cannot-be-analyzed-directly-with-the-given
def explain():   
    return shap.Explainer(reloaded)
explainer = explain()


# cosmetic



###### app





# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '22%',
    'margin-right': '2%',
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9',
}

color_discrete_map={ # replaces default color mapping by value
                0: "#26A65B", 1: "red"}

controls = dbc.Row(
    [
        html.P('Plus de graphes...', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(X.columns[2:],
            id='dropdown_s1',
            multi=False
        )
        ,
        html.P(),
        html.Hr(),
        html.P(),
        html.P('Client ID (SK_ID_CURR)', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(apptest['SK_ID_CURR'],
            id='dropdown',
            multi=False
        ),
        html.Br(),
        html.P('Nombre de paramètres à afficher', style={
            'textAlign': 'center'
        }),
        dcc.Slider(
            id='slider',
            min=10,
            max=30,
            step=2,
            value=10
        ),
        html.Br(),
        html.P('Proches voisins', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.RadioItems(
            id='radio_items',
            options=[{
                'label': '5',
                'value': '5'
                },
                {
                    'label': '10',
                    'value': '10'
                },
            ],
            value='5',
            style={
                'margin': 'auto'
            }
        )]),
        html.Br(),
        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
        ),
    ]
)

sidebar = html.Div(
    [
        html.H2('Paramètres', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

content_first_row_tab3 = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_1', children=['Select Client ID'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4('#Features', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id='card_text_2', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Solvabilité', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id='card_text_3', children=['Sample text.'], style=CARD_TEXT_STYLE),

                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Risque', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id='card_text_4', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                ),
            ]
        ),
        md=3
    )
])


content_first_row_tab1 = dbc.Row(
    [
        dbc.Col(
            html.Div([
                html.H2('Pourcentage de solvables et non solvables'),
            dcc.Graph(id='pie_graph')
            ]))
    ]
)




content_second_row_tab1 = dbc.Row(
    [

        dbc.Col(
            dcc.Graph(id='graph_1'), md=6
        ),
        dbc.Col(
            dcc.Graph(id='graph_2'), md=6
        ),
        dbc.Col( #TODO créer autre row
            html.Div([
                html.H2('Paramètres de scoring de l\'emprunteur'),
        # image crée dans un autre notebook,
                html.Img(src=app.get_asset_url("lgbm_importances01.png"))]
            )
        ),       
    ]
)


content_first_row_tab2 = dbc.Row(
    [   dbc.Col(
            dcc.Graph(id='graph_4'), md=12,
        )
    ]
)

content_second_row_tab3 = dbc.Row(
    [
        dbc.Col(
            [   html.P(style={'margin-top': '40px'}),
                html.Div(id='table_1')
    ]
         #dcc.Graph(id='graph_6'), md=6
        ),
        dbc.Col(
            daq.Gauge(id='my-gauge-1',
                label="          ",
                color={"gradient":True,"ranges":{"red":[0,0.4],"yellow":[0.4,0.5],"#26A65B":[0.5,1]}},
                value=0.1,
                max=1,
                min=0,
                showCurrentValue=True,
            ), md=6
        ),

    ]
)



    
content_fourth_row_tab3 = dbc.Row(
    [   html.Div(id='output-shap')
    ]
)

content_fifth_row_tab3 = dbc.Row(
    [   dbc.Col(
            dcc.Graph(id='shap_waterfall'), md=12,
        )
    ]
)

content_first_row_tab4 = dbc.Row(
    [   html.Div(id='table_2')
    ]
)

content_second_row_tab4 = dbc.Row(
    [   dbc.Col(
            dcc.Graph(id='graph_5'), md=12,
        )
    ]
)


    
### content    
    
    
content = html.Div(
    [
        dcc.Tabs([
            dcc.Tab(label='Statistiques', children=[
                html.H2('Prêt à dépenser', style=TEXT_STYLE),
                html.Hr(),
                content_first_row_tab1,
                content_second_row_tab1,

            ]),
            dcc.Tab(label='Plus de Statistiques', children=[
                html.H2('Prêt à dépenser', style=TEXT_STYLE),
                html.Hr(),
                content_first_row_tab2,

            ]),

            dcc.Tab(label='Paramètres clients', children=[
                html.H2('Prêt à dépenser', style=TEXT_STYLE),
                html.Hr(),
                content_first_row_tab3,
                content_second_row_tab3,
#                content_third_row_tab3,
                content_fourth_row_tab3,
                content_fifth_row_tab3
            ]),

            dcc.Tab(label='Comparaisons', children=[
                html.H2('Prêt à dépenser', style=TEXT_STYLE),
                html.Hr(),
                content_first_row_tab4,
                content_second_row_tab4,
            ])
        ])
    ],
    style=CONTENT_STYLE
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])



#tab1
@app.callback(
    Output('graph_1', 'figure'),
    [Input('submit_button', 'n_clicks')],
    )
def update_graph_1(n_clicks):
    print("graph_1",n_clicks)
    df = cosmeticX
    fig = px.histogram(
        df, x='DAYS_BIRTH', color="TARGET",
        title="Solvabilité par age",
                labels={ # replaces default labels by column name
                "DAYS_BIRTH": "Age",  "TARGET": "Solvabilité"
            },
            color_discrete_map=color_discrete_map
    )
    return fig


#tab1
@app.callback(
    Output('graph_2', 'figure'), 
    [Input('submit_button', 'n_clicks')])
def update_graph_2(n_clicks):
    print("graph_2",n_clicks)
    df=cosmeticX
    fig = px.histogram(df, x="CODE_GENDER", color='TARGET',
                title="Solvabilité par genre",
                labels={ # replaces default labels by column name
                "CODE_GENDER": "Gender",  "TARGET": "Solvabilité"
            },
            color_discrete_map=color_discrete_map
                    )
    return fig


#tab3
# multiple output shap
@app.callback(Output('output-shap', 'children'), Output('shap_waterfall', 'figure'),#Output('shap_waterfall', 'figure'),#Output('shap_waterfall', 'srcDoc'), #Output('shap_waterfall', 'src'), 
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), 
     State('slider', 'value')],
    prevent_initial_call=True)
def update_shap_figures(n_clicks, dropdown_value, slider_value):  
    print("shap js et fig",n_clicks)
    print("shap SLIDER",slider_value)
    data_for_prediction=apptest[apptest['SK_ID_CURR']==dropdown_value]
    data_for_prediction=data_for_prediction.iloc[:,1:]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    #explainer = shap.Explainer(reloaded) # global
        # Calculate Shap values
    shap_val = explainer.shap_values(data_for_prediction_array)
        # visualize the first prediction's explanation
    forceplot=shap.force_plot(explainer.expected_value[0], shap_val[0], data_for_prediction)
    shap_html = f"<head>{shap.getjs()}</head><body>{forceplot.html()}</body>"
    # end forceplot
    # plot waterfall
    #tentatives avortees d'utiliser shap.plot
    #waterfall='waterfall'+str(n_clicks)+'.png'
    #print('waterfallpath',waterfall)
    #shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
    #                                   shap_val[0][0],
    #                                   feature_names = apptest.columns[1:],
    #                                   show=False)
    #plt.tight_layout()
    #plt.savefig('assets/'+waterfall, bbox_inches='tight')
    #plt.close('assets/'+waterfall)
    #htmlfig=mpld3.fig_to_html(fig) 
    # if fig.canvas is None: AttributeError: 'NoneType' object has no attribute 'canvas'        

    # waterfall avec go.figure
    wf = pd.DataFrame(data=[shap_val[0][0],np.absolute(shap_val[0][0])], columns=apptest.columns[1:])
    wfs = wf.sort_values(wf.last_valid_index(), ascending=False, axis=1)
    nwf=wfs.iloc[:, : slider_value]
    nwf['Autres']=wfs.iloc[:, slider_value:].sum(axis=1)
    fig = go.Figure(go.Waterfall(
        name = slider_value, orientation = "h",
        y = nwf.columns,
        textposition = "outside",
        x = nwf.iloc[0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
    fig.layout.plot_bgcolor = 'whitesmoke'
    fig.update_layout(
    autosize=False,
    width=800,
    height=500,) 

    
    return html.Iframe(srcDoc=shap_html,
     style={"width": "100%", "height": "200px", "border": 0}), fig
    #, fig #app.get_asset_url(waterfall)

    
#tab2    
@app.callback(
    Output('graph_4', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown_s1', 'value')])
def update_graph_4(n_clicks, dropdown_s1_value):
    print("graph_4",n_clicks)
    print(dropdown_s1_value)
    if dropdown_s1_value:
        df=cosmeticX
        fig = px.histogram(df, x=dropdown_s1_value, color='TARGET',
                    labels={ # replaces default labels by column name
                    "TARGET": "Solvabilité"},
                    color_discrete_map=color_discrete_map)
        return fig
    else:
        return {"layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                "text": "Choisir un feature dans le menu",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }}]}}



#tab3 card available
@app.callback(
    Output('pie_graph', 'figure'),
    [Input('submit_button', 'n_clicks')])
def update_pie_graph(n_clicks):
    print("pie",n_clicks)
    target_values = X['TARGET'].value_counts()
    colors = ["#26A65B", "red"]
    fig = go.Figure(data=[go.Pie(labels=["Solvable", "Non Solvable"],
                             textinfo='label+percent',
                             values=target_values,
                            )])
    fig.update_traces(textfont_size=20,
                  marker=dict(colors=colors))
    return fig

#tab3 card
@app.callback(
    Output('card_title_1', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value')
     ])
def update_card_title_1(n_clicks, dropdown_value):
    print(n_clicks)
    print(dropdown_value)
    return 'Client ID'

#tab3 card
@app.callback(
    Output('card_text_1', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value')
     ])
def update_card_text_1(n_clicks, dropdown_value):
    print(n_clicks)
    print(dropdown_value)
    return dropdown_value

#tab3 card
@app.callback(
    Output('card_text_2', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('slider', 'value')
     ])
def update_card_text_2(n_clicks, slider_value):
    print(n_clicks)
    print(slider_value)
    return slider_value



#tab3
# Output pour gauge e card
@app.callback(Output('my-gauge-1', 'value'), 
              Output('card_text_3', 'children'), 
              Output('card_text_4', 'children'),
            [Input('submit_button', 'n_clicks')],
            [State('dropdown', 'value')
             ])
def update_gauge_card(n_clicks, dropdown_value):
    print("gauge et card",n_clicks)
    print(dropdown_value)
    if dropdown_value:
        data_for_prediction=apptest[apptest['SK_ID_CURR']==dropdown_value]
        data_for_prediction=data_for_prediction.iloc[:,1:]
        data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
        predicted=reloaded.predict_proba(data_for_prediction_array)
        return predicted[0][0], predicted[0][0], predicted[0][1]
    else: 
        return 0.5, "", ""


#tab3
@app.callback(
    Output('table_1', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('slider', 'value')
     ])
def update_table_1(n_clicks, dropdown_value, slider_value):
    print('Table',n_clicks, dropdown_value, slider_value )    
    table_data=cosmetic_apptest[cosmetic_apptest['SK_ID_CURR']==dropdown_value].iloc[: , :slider_value].T.reset_index()
    if len(table_data.columns)>1:
        table_data.rename(columns={table_data.columns[1]: "value" }, inplace = True)

    return html.Div(
        [   html.H3("Paramètres Client"),
            dash_table.DataTable(
                data=table_data.to_dict("records"),
                style_as_list_view=True,
                style_cell={'padding': '5px','textAlign': 'left', 'backgroundColor': 'whitesmoke',},
                style_header={
                    'backgroundColor': 'white',
                    'fontWeight': 'bold'
                },
                
            )
        ])
    
#tab4
@app.callback(
    Output('table_2', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('radio_items', 'value')
     ])
    
def update_table_2(n_clicks, dropdown_value, radio_items_value):
    print('Table dummy',n_clicks, dropdown_value, radio_items_value )
    response=requests.get(DATA)
    content = json.loads(response.content.decode('utf-8'))
    return html.Div(
        [   html.H3("Paramètres Client"),
            dash_table.DataTable(
                data=content,
                style_as_list_view=True,
                style_cell={'padding': '5px','textAlign': 'left', 'backgroundColor': 'whitesmoke',},
                style_header={
                    'backgroundColor': 'white',
                    'fontWeight': 'bold'
                },
                
            )
        ])


#tab4    
@app.callback(
    Output('graph_5', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('radio_items', 'value')
     ])
def update_graph_5(n_clicks, dropdown_value, radio_items_value):
    print('Neighbours graph',n_clicks, dropdown_value, radio_items_value )
    response=requests.get(DATA)
    content = json.loads(response.content.decode('utf-8'))
    if dropdown_value:
        df=pd.json_normalize(content)
        fig = go.Figure()
        # Use x instead of y argument for horizontal plot
        for col in df.columns:
            fig.add_trace(go.Box(x=df[col], name=col,
                          boxpoints='all', # can also be outliers, or suspectedoutliers, or False
                          jitter=0.3, # add some jitter for a better separation between points
                          pointpos=-1.8 # relative position of points wrt box
                          ))
            fig.update_layout(
                autosize=False,
                width=1000,
                height=800)

        return fig
    else:
        return {"layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                "text": "Choisir un client ID",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }}]}}



def kvoisins(nn,Xnbs,X):
    nnbs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(Xnbs)
    indexes = nnbs.kneighbors(Xnbs[0:1])[1].flatten()
    return X.iloc[indexes]

if __name__ == '__main__':
    app.run_server(port='8085')

