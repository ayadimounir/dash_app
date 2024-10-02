import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap"], use_pages=True)

server = app.server

# Define a modern navigation bar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Chlorine Byproducts Prediction", style={"fontWeight": "bold", "color": "#ffffff", "fontSize": "1.5rem"}),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dcc.Link(
                            f"{page['name']}", href=page["relative_path"],
                            className="nav-link", style={'margin': '0 10px'}
                        )
                        for page in dash.page_registry.values()
                    ],
                    className="ml-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
    ),
    color="dark",  # Navbar color
    dark=True,  # Invert text to light for dark background
    className="mb-5"
)

# Define a modern footer
footer = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.P("© 2024 by Mounir AYADI", className="text-center", style={"margin": "10px 0"})),
            ],
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#333333", "color": "#ffffff", "padding": "10px 0", "marginTop": "30px"}
)

# Create the layout of the app
app.layout = dbc.Container(
    [
        navbar,  # call the defined navbar
        dbc.Container(
            [
                dash.page_container  # Render the pages dynamically here
            ],
            fluid=True,
            style={ 'backgroundColor': '#f8f9fa'}  # Light background for main content area
        ),
        footer  # Add footer at the bottom
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)
