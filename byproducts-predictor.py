import os
import pandas as pd
import numpy as np
import joblib
from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from pathlib import Path

# Define folders and their corresponding features
folders_and_features = {
    "model_multi_output_SUVA_results": ['temps', 'Cl2_i', 'NH4i', 'TOCi', 'UV254i'],
    "model_multi_output_SUVA_pH_results": ['temps', 'Cl2_i', 'NH4i', 'TOCi', 'UV254i', 'pHi'],
    "model_multi_output_SUVA_Br_results": ['temps', 'Cl2_i', 'NH4i', 'TOCi', 'UV254i', 'Br_i'],
    "model_multi_output_SUVA_Br_pH_results": ['temps', 'Cl2_i', 'NH4i', 'TOCi', 'UV254i', 'Br_i', 'pHi'],

    "model_multi_output_PARAFAC_results": ['temps', 'Cl2_i', 'NH4i', 'C1i', 'C2i', 'C3i', 'C4i', 'C5i', 'C6i'],
    "model_multi_output_PARAFAC_pH_results": ['temps', 'Cl2_i', 'NH4i', 'C1i', 'C2i', 'C3i', 'C4i', 'C5i', 'C6i', 'pHi'],
    "model_multi_output_PARAFAC_Br_results": ['temps', 'Cl2_i', 'NH4i', 'C1i', 'C2i', 'C3i', 'C4i', 'C5i', 'C6i', 'Br_i'],
    "model_multi_output_PARAFAC_Br_pH_results": ['temps', 'Cl2_i', 'NH4i', 'C1i', 'C2i', 'C3i', 'C4i', 'C5i', 'C6i', 'Br_i', 'pHi'],

    "model_multi_output_breakpoint_results": ['temps', 'Cl2_i', 'NH4i'],
    "model_multi_output_cl2_results": ['temps', 'Cl2_i']
}

# List of models
models = ["Linear Regression",
         "Lasso Regression",
         "Ridge Regression",
         "ElasticNet Regression",
         "Support Vector Regressor",
         "XGBoost",
         "Gradient Boosting",
         "Neural Network",
         "Random Forest"]

# Target variables
target_variables = [ 'AOX', 'TCM', 'BDCM', 'DBCM', 'TBM' ]

# Function to find the model file
def find_model_file(folder: str, model_name: str, folder_name: str) -> str:
    # Construct the expected filename
    if folder_name.startswith("model_"):
        folder_name = folder_name[len("model_"):]
    filename = f"{model_name}_{folder_name}.csv"
    file_path = Path(folder) / filename

    if file_path.exists():
        return str(file_path)  # Convert Path object to string if necessary
    else:
        raise FileNotFoundError(f"File {file_path} not found.")









# Reload and preprocess the actual_data
file_path = 'dependencies/data/data.xlsx'
actual_data = pd.read_excel(file_path)

columns = ['eau', 'temps', 'UV_system', 'UV_fluence', 'replicat', 'Cl2_initial', 'NH4', 'UV254', 'TOC', 'SUVA', 'pH', 'Conductivite', 'TAC', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Br_i', 'Cl2_libre', 'Cl2_total', 'Cl2_combine', 'TCM', 'BDCM', 'DBCM', 'TBM', 'THM4', 'AOX']
new_columns = ['eau', 'temps', 'UV_system', 'UV_fluence', 'replicat', 'Cl2_i', 'NH4i', 'UV254i', 'TOCi', 'SUVAi', 'pHi', 'Conductivite_i', 'TACi', 'C1i', 'C2i', 'C3i', 'C4i', 'C5i', 'C6i', 'Br_i', 'Cl2_libre', 'Cl2_total', 'Cl2_combine', 'TCM', 'BDCM', 'DBCM', 'TBM', 'THM4', 'AOX']
actual_data = actual_data[columns].rename(columns=dict(zip(columns, new_columns)))
actual_data.fillna(0, inplace=True)

# Set 'AOX' to 0 where 'temps' != 60
actual_data.loc[actual_data['temps'] != 60, 'AOX'] = 0

# Function to replace values with temps=0 values within each group
def replace_with_temps0(df):
    temps0_values = df[df['temps'] == 0].iloc[0][['NH4i', 'UV254i', 'TOCi', 'SUVAi', 'pHi', 'Conductivite_i', 'TACi', 'C1i', 'C2i', 'C3i', 'C4i', 'C5i', 'C6i', 'Br_i']]
    for col in temps0_values.index:
        df[col] = temps0_values[col]
    return df

actual_data = actual_data.groupby('eau', group_keys=False).apply(replace_with_temps0).reset_index(drop=True)
actual_data = actual_data[actual_data['UV_system'] == 0]
actual_data = pd.get_dummies(actual_data, columns=['eau'])








# Define reasonable default ranges for the features
feature_ranges = {
    'temps': (0, 60),
    'Cl2_i': (0, 20),
    'NH4i': (0, 17),
    'C1i': (0, 4.3),
    'C2i': (0, 2.3),
    'C3i': (0, 2.4),
    'C4i': (0, 1.5),
    'C5i': (0, 2.3),
    'C6i': (0, 1.1),
    'pHi': (7, 8.5),
    'Br_i': (0, 380),
    'TOCi': (0, 22),
    'UV254i': (0, 0.260)
}









# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

# Define the app layout
app.layout = dbc.Container([
    html.H1(
        "Chlorine Byproducts Prediction by Machine Learning",
        style={'textAlign': 'center', 'font-size': '2rem'}
    ),
    html.H2(
        "By Mounir AYADI",
        style={'textAlign': 'center', 'font-size': '1rem'}
    ),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Select prediction parameters:", style={'font-size': '0.85rem'}),
            dcc.Dropdown(
                id='equation-dropdown',
                options=[{'label': f"{idx+1} - \"{eq}\"", 'value': eq} for idx, eq in enumerate(folders_and_features.keys())],
                value=list(folders_and_features.keys())[0],
                style={'font-size': '0.85rem', 'height': '25px', 'padding': '5px'}
            ),
            html.Br(),
            html.Label("Select Model:", style={'font-size': '0.85rem'}),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': f"{idx+1} - \"{model}\"", 'value': model} for idx, model in enumerate(models)],
                value=models[0],
                style={'font-size': '0.85rem', 'height': '25px', 'padding': '5px'}
            ),
            html.Br(),
            html.Label("Select Target Variable:", style={'font-size': '0.85rem'}),
            dcc.Dropdown(
                id='target-dropdown',
                options=[{'label': f"{idx+1} - \"{target}\"", 'value': target} for idx, target in enumerate(target_variables)],
                value=target_variables[0],
                style={'font-size': '0.85rem', 'height': '25px', 'padding': '5px'}
            ),
            html.Br(),
            html.Label("Select X-axis Feature:", style={'font-size': '0.85rem'}),
            dcc.Dropdown(
                id='x-feature-dropdown',
                style={'font-size': '0.85rem', 'height': '25px', 'padding': '5px'}
            ),
            html.Br(),
            html.Label("Select Y-axis Feature:", style={'font-size': '0.85rem'}),
            dcc.Dropdown(
                id='y-feature-dropdown',
                style={'font-size': '0.85rem', 'height': '25px', 'padding': '5px'}
            ),
            html.Br(),
            html.Div(id='feature-inputs'),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='prediction-graph', style={'height': '125vh'}),
            dcc.Store(id='camera-store')  # Store to keep camera state
        ], width=9)
    ], align='start')
], fluid=True)

# Callback to update feature dropdowns based on selected equation
@app.callback(
    [Output('x-feature-dropdown', 'options'),
     Output('x-feature-dropdown', 'value'),
     Output('y-feature-dropdown', 'options'),
     Output('y-feature-dropdown', 'value')],
    Input('equation-dropdown', 'value')
)
def update_feature_dropdowns(selected_equation):
    features = folders_and_features[selected_equation]
    options = [{'label': f"{idx+1} - \"{feature}\"", 'value': feature} for idx, feature in enumerate(features)]
    default_value_x = features[0]
    default_value_y = features[1] if len(features) > 1 else features[0]
    return options, default_value_x, options, default_value_y

# Callback to generate sliders for features
@app.callback(
    Output('feature-inputs', 'children'),
    [Input('equation-dropdown', 'value'),
     Input('x-feature-dropdown', 'value'),
     Input('y-feature-dropdown', 'value')]
)
def update_feature_inputs(selected_equation, x_feature, y_feature):
    features = folders_and_features[selected_equation]
    inputs = []
    for feature in features:
        min_val, max_val = feature_ranges.get(feature, (0, 30))
        if feature == x_feature or feature == y_feature:
            inputs.append(html.Label(f"Select range for {feature}:", style={'font-size': '0.85rem'}))
            inputs.append(
                dcc.RangeSlider(
                    id={'type': 'feature-slider', 'index': feature},
                    min=min_val,
                    max=max_val,
                    step=(max_val - min_val) / 30,
                    value=[min_val, max_val],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag',
                    persistence=True,
                    persistence_type='session',
                )
            )
            inputs.append(html.Br())
        else:
            default_val = (min_val + max_val) / 2
            inputs.append(html.Label(f"Select value for {feature}:", style={'font-size': '0.85rem'}))
            inputs.append(
                dcc.Slider(
                    id={'type': 'feature-slider', 'index': feature},
                    min=min_val,
                    max=max_val,
                    step=(max_val - min_val) / 30,
                    value=default_val,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag',
                    persistence=True,
                    persistence_type='session',
                )
            )
            inputs.append(html.Br())
    return inputs

# Callback to perform prediction and update the output and graph in real-time
@app.callback(
    Output('prediction-graph', 'figure'),
    Output('camera-store', 'data'),
    [Input('equation-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('target-dropdown', 'value'),
     Input('x-feature-dropdown', 'value'),
     Input('y-feature-dropdown', 'value'),
     Input({'type': 'feature-slider', 'index': ALL}, 'value'),
     Input('prediction-graph', 'relayoutData')],
    State('camera-store', 'data')
)
def perform_prediction(selected_equation, selected_model_name, selected_target_variable, x_feature, y_feature, feature_values, relayoutData, camera_data):
    features = folders_and_features[selected_equation]
    feature_value_map = {}
    x_range = None
    y_range = None

    # Map feature values
    for feature, value in zip(features, feature_values):
        feature_value_map[feature] = value

    # Extract ranges for X and Y features
    x_range = feature_value_map.get(x_feature)
    y_range = feature_value_map.get(y_feature)

    if x_range is None or y_range is None:
        return go.Figure(), camera_data

    num_steps = 30  # You can adjust the resolution here
    X_values = np.linspace(x_range[0], x_range[1], num_steps)
    Y_values = np.linspace(y_range[0], y_range[1], num_steps)
    X_mesh, Y_mesh = np.meshgrid(X_values, Y_values)
    grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])

    # Prepare DataFrame for prediction
    df_list = []
    for x_val, y_val in grid_points:
        feature_dict = {}
        for feature in features:
            if feature == x_feature:
                feature_dict[feature] = x_val
            elif feature == y_feature:
                feature_dict[feature] = y_val
            else:
                feature_dict[feature] = feature_value_map[feature]
        df_list.append(feature_dict)
    grid_df = pd.DataFrame(df_list)

    # Load the model
    model_folder_path = Path("./dependencies") / selected_equation
    try:
        csv_path = find_model_file(model_folder_path, selected_model_name, selected_equation)
    except FileNotFoundError:
        return go.Figure(), camera_data

    # Load the results CSV
    model_results_df = pd.read_csv(csv_path)

    # Find the best model
    best_model_row = model_results_df.loc[model_results_df['r2_test'].idxmax()]
    best_model_path = best_model_row['model_path']
    best_model_seed = int(best_model_row['seed'])
    best_model_fold_id = int(best_model_row['fold_id'])

    parent_dir = os.path.dirname(model_folder_path)
    best_model_full_path = Path(parent_dir) / best_model_path

    if not best_model_full_path.exists():
     return go.Figure(), camera_data

    best_model = joblib.load(best_model_full_path)

    if hasattr(best_model, 'named_steps') and 'standardscaler' in best_model.named_steps:
        model_features = best_model.named_steps['standardscaler'].feature_names_in_
    else:
        model_features = features

    missing_features = [feat for feat in model_features if feat not in grid_df.columns]
    for feat in missing_features:
        grid_df[feat] = 0  # Fill missing features with 0

    grid_df = grid_df[model_features]

    try:
        y_pred = best_model.predict(grid_df)
    except Exception:
        return go.Figure(), camera_data

    target_idx = target_variables.index(selected_target_variable)
    Z_values = y_pred[:, target_idx].reshape(X_mesh.shape)

    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(x=X_mesh, y=Y_mesh, z=Z_values)])

    # Maintain the camera state
    if relayoutData and 'scene.camera' in relayoutData:
        camera_data = relayoutData['scene.camera']

    fig.update_layout(
        scene=dict(
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            zaxis_title=selected_target_variable,
            camera=camera_data  # Apply the stored camera view
        ),
        autosize=True,
        height=650,
        margin=dict(l=50, r=50, b=65, t=10)
    )

    return fig, camera_data

if __name__ == "__main__":
    app.run_server(debug=True)

