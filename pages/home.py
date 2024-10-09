import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

# Layout definition
layout = html.Div([
    # Header Section
    html.Div([
        html.H1('Welcome to DBP Predictor: A prediction tool for byproducts formation during oxidation powered by Machine Learning', 
                style={'text-align': 'center', 'margin-top': '20px'}),
    ], style={'padding': '20px'}),

    # Welcome Text Section
    html.Div([        
        html.P(
            '''
            DBP predictor is a website tool to help predict byproducts during wastewater chlorination.
            Our predictions are made using various Machine Learning algorithms and real world collected data. 
            Our best models perform with R2=0.968 when compared to real world measurements.
            Click on "visualisation" to start exploring our predictions.
            ''',
            style={'font-size': '16px', 'line-height': '1.6', 'text-align': 'justify', 'padding': '0px 50px'}
        ),
    ], style={'padding': '30px'}),

    # How to Use Section
    html.Div([
        html.H3('How to Use', style={'text-align': 'center'}),
        html.P(
            '''
            Select a set of features used for prediction and a machine learning model from the drop down menus.
            Select a target variable you wish to predict, and two axes on which you wish to see variations.
            You can adjust the remaining sliders to input your desired water matrix. The graphic is fully interactive 
            with Zoom, rotation, value highlighting, and real-time updates.
            ''',
            style={'font-size': '16px', 'line-height': '1.6', 'text-align': 'justify', 'padding': '0px 50px'}
        ),
    ], style={'padding': '30px'}),

    # Construction Notice Section
    html.Div([
        html.H3('Note:', style={'text-align': 'center', 'color': '#d9534f'}),
        html.P(
            '''
            This website is still under construction, please feel free to share your feedback via mail 
            in the "About" section!
            ''',
            style={'font-size': '16px', 'line-height': '1.6', 'text-align': 'center', 'color': '#d9534f'}
        ),
    ], style={'padding': '20px'}),
])
