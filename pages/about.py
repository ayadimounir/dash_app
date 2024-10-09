import dash
from dash import html

dash.register_page(__name__)

layout = html.Div(
    [
        html.H1(
            'About us', 
            style={
                'textAlign': 'center', 
                'fontSize': '36px',  # Increased font size for the title
                'fontWeight': 'bold'
            }
        ),
        
        html.P(
            [
                "This website is based on a scientific work under publication, authored by ",

                html.B("Mounir AYADI."),  # Highlight the name for emphasis
                html.Br(),
                "Mounir AYADI is a PhD candidate working at the University of Poitiers under the ",
                html.Span("IC2MP EBicom team.", style={'fontStyle': 'italic'}),  # Italicize team name
                html.Br(), html.Br(),
                "Email: ",
                html.A("mounir.ayadi@univ-poitiers.fr", href="mailto:mounir.ayadi@univ-poitiers.fr", style={'color': '#1E90FF'}),  # Make the email clickable and styled
            ],
            style={
                'fontSize': '18px',
                'lineHeight': '1.6',  # Increase line spacing for readability
                'marginTop': '15px'
            }
        ),
    ],
    style={
        'margin': '20px',
        'fontFamily': 'Arial, sans-serif'
    }
)
