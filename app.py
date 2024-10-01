import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

server = app.server

# Create navigation links to other pages
navigation_links = html.Div(
    [
        dcc.Link(
            f"{page['name']}", href=page["relative_path"], style={'margin': '10px'}
        )
        for page in dash.page_registry.values()
    ],
    style={'textAlign': 'center', 'padding': '20px'}
)

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
    navigation_links,  # Include navigation links here
    dash.page_container  # This will render the pages
])

if __name__ == "__main__":
    app.run_server(debug=True)
