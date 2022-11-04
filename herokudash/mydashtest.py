#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 20:25:33 2022

@author: mcBedini
"""

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq

from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px


from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors

import shap

from flask import Flask
import requests
import json


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

#heroku
server=app.server


#url endpoints
#PREDICT_PROBA='http://127.0.0.1:8080/predict_proba'

PREDICT_PROBA='https://firsttestwith.herokuapp.com/predict_proba'

#SHAP='http://127.0.0.1:8080/shap'

SHAP='https://firsttestwith.herokuapp.com/shap'

def readdf(debug=False):
    savedataframe = "dashsample.csv" if debug else "apidataframe.csv"
    df = pd.read_csv(savedataframe)
    print("Reading",savedataframe)
    print("Shape",df.shape)
    return df


def prepare_train_df(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    feats = [f for f in train_df.columns if f not in ['SK_ID_BUREAU','SK_ID_PREV','index']]
    test_feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    apptrain, apptest = train_df[feats], test_df[test_feats]
    print('Training dataframe shape:', apptrain.shape)
    print('Test shape:', apptest.shape)
    return apptrain, apptest

df=readdf(True)
apptrain, apptest = prepare_train_df(df)
imputed= pd.read_csv("imputed.csv")
print("Shape IMPUTED", imputed.shape)

importance=['PAYMENT_RATE', 'EXT_SOURCES_WEIGHTED', 'CREDIT_PRICE_DIFF', 'Inst_DPD_MEAN',
            'Inst_AMT_PAYMENT_SUM', 'APPROVED_CNT_PAYMENT_SUM', 'Buro_ACTIVE_DEBT_CREDIT_DIFF_SUM',
            'DAYS_BIRTH', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'Cc_AMT_BALANCE_MEAN',
            'Buro_ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN', 'DAYS_ID_PUBLISH',
            'POS_COUNT', 'Cc_CNT_DRAWINGS_CURRENT_MEAN', 'OWN_CAR_AGE',
            'Prev_APPLICATION_CREDIT_DIFF_SUM', 'Buro_ACTIVE_CREDIT_DURATION_MEAN',
            'Pos_MONTHS_BALANCE_MEAN', 'Buro_CLOSED_CREDIT_DURATION_MEAN',
            'Pos_SK_DPD_DEF_MEAN', 'Prev_CNT_PAYMENT_SUM', 'ANNUITY_INCOME_RATIO',
            'Buro_ACTIVE_DAYS_CREDIT_UPDATE_MEAN', 'REGION_POPULATION_RELATIVE',
            'APARTMENTS_AVG', 'DAYS_REGISTRATION', 'Prev_DAYS_DECISION_MEAN',
            'APPROVED_DAYS_DECISION_MEAN', 'Buro_CLOSED_DAYS_CREDIT_UPDATE_MEAN']


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
        dcc.Dropdown(apptrain.columns[2:],
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
                        html.H4('Recommandation', className='card-title', style=CARD_TEXT_STYLE),
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
        ),
        dbc.Col(
            daq.Gauge(id='my-gauge-1',
                label="          ",
                scale={'start': 0, 'interval': 0.05, 'labelInterval': 2},
                color={"gradient":True,"ranges":{"red":[0,0.75],"yellow":[0.75,0.85],"#26A65B":[0.85,1]}},
                value=0.5,
                max=1,
                min=0,
                showCurrentValue=True,
            ), md=6
        ),

    ]
)



    
# content_fourth_row_tab3 = dbc.Row(
#     [   html.Div(id='output-shap')
#     ]
# )

content_fifth_row_tab3 = dbc.Row(
    [   html.Div(id='api-shap')
    ]
)
    
#     [   dbc.Col(
#             dcc.Graph(id='shap_waterfall'), md=12,
#         )
#     ]
# )

#    dbc.Col(
#         dbc.Card(
#             [

#                 dbc.CardBody(
#                     [
                        
#                         html.P(id='apicall', children=['Sample text.'], style=CARD_TEXT_STYLE),
#                     ]
#                 )
#             ]
#         ),
#         md=3
#     )
# )
    


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
#                content_fourth_row_tab3,
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

app.layout = html.Div([sidebar, content])
layout = html.Div([sidebar, content])



#tab1
@app.callback(
    Output('graph_1', 'figure'),
    [Input('submit_button', 'n_clicks')],
    )
def update_graph_1(n_clicks):
    print("graph_1",n_clicks)
    df = apptrain
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
    df=apptrain
    fig = px.histogram(df, x="CODE_GENDER", color='TARGET',
                title="Solvabilité par genre",
                labels={ # replaces default labels by column name
                "CODE_GENDER": "Gender",  "TARGET": "Solvabilité"
            },
            color_discrete_map=color_discrete_map
                    )
    return fig


#tab3
# @app.callback(Output('output-shap', 'children'), #Output('shap_waterfall', 'figure'),#Output('shap_waterfall', 'figure'),#Output('shap_waterfall', 'srcDoc'), #Output('shap_waterfall', 'src'), 
#     [Input('submit_button', 'n_clicks')],
#     [State('dropdown', 'value'), 
#      State('slider', 'value')],
#     prevent_initial_call=True)
# def update_shap_figures(n_clicks, dropdown_value, slider_value):  
#     print("shap js et fig",n_clicks)
#     print("shap SLIDER",slider_value)
#     data_for_prediction=apptest[apptest['SK_ID_CURR']==dropdown_value]
#     data_for_prediction=data_for_prediction.iloc[:,1:]
#     data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
#     shap_val = explainer.shap_values(data_for_prediction_array)
#         # visualize the first prediction's explanation
#     forceplot=shap.force_plot(explainer.expected_value[0], shap_val[0], data_for_prediction)
#     shap_html = f"<head>{shap.getjs()}</head><body>{forceplot.html()}</body>"
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

    # waterfall avec go.figure Supprimé sur conseil de mon tutor. Information equivalente au precedent graphe
    #wf = pd.DataFrame(data=[shap_val[0][0],np.absolute(shap_val[0][0])], columns=apptest.columns[1:])
    #wfs = wf.sort_values(wf.last_valid_index(), ascending=False, axis=1)
    #nwf=wfs.iloc[:, : slider_value]
    #nwf['Autres']=wfs.iloc[:, slider_value:].sum(axis=1)
    #fig = go.Figure(go.Waterfall(
    #    name = slider_value, orientation = "h",
    #    y = nwf.columns,
    #    textposition = "outside",
    #    x = nwf.iloc[0],
    #    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    #    ))
    #fig.layout.plot_bgcolor = 'whitesmoke'
    #fig.update_layout(
    #autosize=False,
    #width=800,
    #height=500,) 

    
    # return html.Iframe(srcDoc=shap_html,
    #  style={"width": "100%", "height": "200px", "border": 0})#, fig
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
        df=apptrain
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



#tab3 
@app.callback(
    Output('pie_graph', 'figure'),
    [Input('submit_button', 'n_clicks')])
def update_pie_graph(n_clicks):
    print("pie",n_clicks)
    target_values = apptrain['TARGET'].value_counts()
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
#@app.callback(
#    Output('card_text_2', 'children'),
#    [Input('submit_button', 'n_clicks')],
#    [State('slider', 'value')
#     ])
#def update_card_text_2(n_clicks, slider_value):
#    print(n_clicks)
#    print(slider_value)
#    return slider_value



#tab3
# Multiple Output pour gauge e card
@app.callback(Output('my-gauge-1', 'value'), 
              Output('card_text_3', 'children'), 
              Output('card_text_4', 'children'),
              Output('card_text_2', 'children'),
            [Input('submit_button', 'n_clicks')],
            [State('dropdown', 'value')
             ])
def update_gauge_card(n_clicks, dropdown_value):
    print("gauge et card",n_clicks)
    print(dropdown_value)
    if dropdown_value:
        data_for_prediction=apptest[apptest['SK_ID_CURR']==dropdown_value].iloc[:,1:]
        data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
        sending=json.dumps(data_for_prediction_array.tolist()[0])
        url = PREDICT_PROBA
        response = requests.post(url,sending)
        content = json.loads(response.content.decode('utf-8'))
        solvable= 'Insolvable' if content[1]>=0.15 else 'Solvable'
        return content[0], content[0], content[1], solvable
    else: 
        return 0.5, "", "", ""




#tab3 card
# @app.callback(
#     Output('apicall', 'children'),
#     [Input('submit_button', 'n_clicks')],
#     [State('dropdown', 'value')
#      ])
# def update_apicall(n_clicks, dropdown_value):
#     print(n_clicks)
#     print(dropdown_value)
#     if dropdown_value:
#         data_for_prediction=apptest[apptest['SK_ID_CURR']==dropdown_value].iloc[:,1:]
#         data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
#         sending=json.dumps(data_for_prediction_array.tolist()[0])
#         url = SHAP
#         response = requests.post(url,sending)
#         content = json.loads(response.content.decode('utf-8'))
#         return content[0]
#     else: 
#         return  "Nothing to see here"


@app.callback(Output('api-shap', 'children'), #Output('shap_waterfall', 'figure'),#Output('shap_waterfall', 'figure'),#Output('shap_waterfall', 'srcDoc'), #Output('shap_waterfall', 'src'), 
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), 
     State('slider', 'value')],
    prevent_initial_call=True)
def update_shap_api(n_clicks, dropdown_value, slider_value):  
    print("shap js ",n_clicks)
    print("shap SLIDER",slider_value)
    if dropdown_value:
        data_for_prediction=apptest[apptest['SK_ID_CURR']==dropdown_value].iloc[:,1:]
        data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
        sending=json.dumps(data_for_prediction_array.tolist()[0])
        url = SHAP
        response = requests.post(url,sending)
        content = json.loads(response.content.decode('utf-8')) 
        explainer_expected_value_0=content[0]
        shap_val_0=np.array(content[1]).reshape(1, -1)

        forceplot=shap.force_plot(explainer_expected_value_0, shap_val_0, data_for_prediction, plot_cmap=["#26A65B","#FF0000"])
        shap_html = f"<head>{shap.getjs()}</head><body>{forceplot.html()}</body>"   
        return html.Iframe(srcDoc=shap_html,
                           style={"width": "100%", "height": "200px", "border": 0})#, fig
    








#tab3
@app.callback(
    Output('table_1', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('slider', 'value')
     ])
def update_table_1(n_clicks, dropdown_value, slider_value):
    print('Table',n_clicks, dropdown_value, slider_value )
    table_data=apptest[apptest['SK_ID_CURR']==dropdown_value]
    table_data=table_data[importance].iloc[: , :slider_value].T.reset_index()
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
# @app.callback(
#     Output('table_2', 'children'),
#     [Input('submit_button', 'n_clicks')],
#     [State('dropdown', 'value'), State('radio_items', 'value'),
#      State('slider', 'value')],
#        prevent_initial_call=True)
# def update_table_2(n_clicks, dropdown_value, radio_items_value, slider_value):
#     riv=int(radio_items_value)
#     content=X.iloc[:riv,:slider_value]
#     print('client',)
#     return html.Div(
#         [   html.H3("Paramètres Client"),
#             dash_table.DataTable(
#                 data=content.to_dict('records'),
#                 style_as_list_view=True,
#                 style_cell={'padding': '5px','textAlign': 'left', 'backgroundColor': 'whitesmoke',},
#                 style_header={
#                     'backgroundColor': 'white',
#                     'fontWeight': 'bold'
#                 },
                
#             )
#         ])


#tab4    Multiple Neighbours
@app.callback(
    Output('graph_5', 'figure'), Output('table_2', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('radio_items', 'value'),
     State('slider', 'value')])
def update_graph_5(n_clicks, dropdown_value, radio_items_value, slider_value):
    riv=int(radio_items_value)+1
    if dropdown_value:
        #Nearest neighbors
        imp=imputed.iloc[:,2:] #imputed dataframe
        client=apptest[apptest['SK_ID_CURR']==dropdown_value].iloc[:,1:] #nouveau client
        voisins=pd.concat([client,imp]) # tous les voisins
        imp_voisins = pd.DataFrame(KNNImputer().fit_transform(voisins),
                           columns=voisins.columns,
                           index=voisins.index) #imputation pour le nouveau voisin
        nnbs = NearestNeighbors(n_neighbors=riv, algorithm='ball_tree').fit(imp_voisins)
        distances, indices = nnbs.kneighbors(imp_voisins[:1]) # les voisins du nouveau client
        voisins=imp_voisins.iloc[indices.flatten()]
        #nombre de feature a montrer
        fvoisins=voisins[importance].iloc[:,:slider_value]
        #table 2
        table=html.Div(
            [   html.H3("Paramètres Client"),
                dash_table.DataTable(
                    data=fvoisins.to_dict('records'),
                    style_as_list_view=True,
                    style_cell={'padding': '5px','textAlign': 'left', 'backgroundColor': 'whitesmoke',},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {
                                'row_index': 0,
                                 },
                            'backgroundColor': '#C9E9FD',
                            'color': 'MediumPurple'
                            }]
                    
                )
            ])
        # graph
        fig = go.Figure()
        for col in fvoisins.columns:
            fig.add_trace(go.Box(x=fvoisins[col], name=col,
                          boxpoints='all', # can also be outliers, or suspectedoutliers, or False
                          jitter=0.3, # add some jitter for a better separation between points
                          pointpos=-1.8 # relative position of points wrt box
                          ))
            fig.add_trace(go.Scatter(x=fvoisins[col].iloc[:1], y=[col]*len(fvoisins[col]), name=col, mode='markers', 
                                     marker=dict(
                                         color='LightSkyBlue',
                                         size=20,
                                         opacity=0.5,
                                         line=dict(
                                             color='MediumPurple',
                                             width=2
                                             )
                                         )

                                     ))

            fig.update_layout(
                autosize=False,
                width=1000,
                height=800)

        
        return fig, table
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
                }}]}}, ""



if __name__ == '__main__':
    app.run_server(port='8085')

