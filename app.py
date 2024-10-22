import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Initialize the Dash app with the DARKLY theme
app = Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,  # Theme for styling
    "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap"  # Font
], use_pages=True)

# Load the figure template for the DARKLY theme
load_figure_template('DARKLY')

server = app.server

# Define the navbar (will be displayed inline across all devices, no mobile hamburger menu)
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                "Chlorine Byproducts Prediction",
                style={
                    "fontWeight": "bold",
                    "color": "#ffffff",
                    "fontSize": "1.5rem"
                }
            ),
            dbc.Nav(
                [
                    dcc.Link(
                        f"{page['name']}",
                        href=page["relative_path"],
                        className="nav-link",
                        style={'margin': '0 5%', 'color': 'white'}
                    )
                    for page in dash.page_registry.values()
                ],
                className="d-flex justify-content-center",  # Align items horizontally
                navbar=True
            )
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-5"
)

# Define a simple footer
footer = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.P("© 2024 by Mounir AYADI", className="text-center", style={"margin": "10px 0"})),
            ],
        ),
    ],
    fluid=True,
    style={"color": "#ffffff", "padding": "10px 0", "marginTop": "30px"}
)

# Define the layout of the app
app.layout = dbc.Container(
    [
        # Meta tag to disable responsiveness and force desktop layout on all devices
        html.Meta(name="viewport", content="width=1024"),

        # Navbar
        navbar,

        # Main content section that renders pages dynamically
        dbc.Container(
            dash.page_container,  # Where multipage content will appear
            fluid=True,
        ),

        # Footer
        footer
    ],
    fluid=True
)

if __name__ == "__main__":
    app.run_server(debug=False)
