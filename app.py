from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Initialize the Dash app with custom stylesheet
app = Dash(__name__)

# Color scheme
COLORS = {
    'background': '#f8f9fa',
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'text': '#2c3e50',
    'light': '#ecf0f1'
}
#Sample comment to check the git changes
# Load and prepare data
df = pd.read_csv('Customer-Churn-Records.csv')

# Calculate key metrics
churn_rate = df['Exited'].mean() * 100
active_rate = df['IsActiveMember'].mean() * 100
avg_satisfaction = df['Satisfaction Score'].mean()
complaint_rate = df['Complain'].mean() * 100

# Create layout with enhanced styling
app.layout = html.Div([
    # Header with gradient background
    html.Div([
        html.H1('Bank Customer Retention Analysis',
                style={
                    'textAlign': 'center',
                    'color': 'white',
                    'padding': '20px',
                    'marginBottom': 0
                })
    ], style={
        'background': 'linear-gradient(120deg, #2c3e50, #3498db)',
        'marginBottom': '30px',
        'borderRadius': '0 0 20px 20px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    }),
    
    # Quick Guide Section (New)
    html.Div([
        html.Div([
            html.H3('📊 How to Use This Dashboard', 
                   style={'color': 'white', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.I(className='fas fa-chart-bar', style={'marginRight': '10px'}),
                    html.Span('View different analyses using the dropdown menu below')
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.I(className='fas fa-mouse-pointer', style={'marginRight': '10px'}),
                    html.Span('Hover over charts for detailed information')
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.I(className='fas fa-search-plus', style={'marginRight': '10px'}),
                    html.Span('Click and drag on charts to zoom, double-click to reset')
                ]),
            ], style={'fontSize': '0.9em'})
        ], style={
            'backgroundColor': 'rgba(52, 152, 219, 0.9)',
            'padding': '20px',
            'borderRadius': '15px',
            'color': 'white',
            'marginBottom': '20px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'transform': 'translateY(0)',
            'transition': 'transform 0.3s ease',
            ':hover': {
                'transform': 'translateY(-5px)'
            }
        })
    ], style={'padding': '0 40px'}),
    
    # Main container
    html.Div([
        # Key Metrics Row
        html.Div([
            html.Div([
                html.H4('Churn Rate', style={'color': COLORS['text']}),
                html.H2(f'{churn_rate:.1f}%', style={'color': COLORS['accent']})
            ], className='metric-card'),
            html.Div([
                html.H4('Active Members', style={'color': COLORS['text']}),
                html.H2(f'{active_rate:.1f}%', style={'color': COLORS['secondary']})
            ], className='metric-card'),
            html.Div([
                html.H4('Avg Satisfaction', style={'color': COLORS['text']}),
                html.H2(f'{avg_satisfaction:.1f}/5', style={'color': COLORS['secondary']})
            ], className='metric-card'),
            html.Div([
                html.H4('Complaint Rate', style={'color': COLORS['text']}),
                html.H2(f'{complaint_rate:.1f}%', style={'color': COLORS['accent']})
            ], className='metric-card'),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}),
        
        # Analysis Selection with styled dropdown
        html.Div([
            html.H3('Select Analysis View:', style={'color': COLORS['primary'], 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='analysis-selector',
                options=[
                    {'label': 'Geographic Analysis', 'value': 'geo'},
                    {'label': 'Age Distribution', 'value': 'age'},
                    {'label': 'Financial Analysis', 'value': 'financial'},
                    {'label': 'Satisfaction Analysis', 'value': 'satisfaction'}
                ],
                value='geo',
                style={'borderRadius': '10px'}
            ),
        ], style={'marginBottom': '20px'}),
        
        # Graph Display with reduced size
        html.Div([
            dcc.Graph(
                id='main-graph',
                style={'height': '450px'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'modeBarButtonsToRemove': ['lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'chart_export',
                        'height': 600,
                        'width': 1000,
                        'scale': 2
                    },
                    'scrollZoom': True,
                    'showTips': True
                }
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '15px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'marginBottom': '30px',
            'transition': 'transform 0.3s ease'
        }),
        
        # Insights section with enhanced styling
        html.Div([
            html.H3('Key Insights and Recommendations', 
                   style={'color': COLORS['primary'], 'marginBottom': '20px', 'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H4('Geographic Focus', style={'color': COLORS['secondary']}),
                    html.P('German market shows significantly higher churn (32.4%). Requires immediate attention.')
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4('Age-Specific Strategy', style={'color': COLORS['secondary']}),
                    html.P('Older customers (avg. 44.8 years) are more likely to churn. Need targeted retention programs.')
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4('High-Value Customer Program', style={'color': COLORS['secondary']}),
                    html.P('Customers with higher balances show increased churn risk. Implement VIP services.')
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4('Product Strategy', style={'color': COLORS['secondary']}),
                    html.P('Multi-product relationships improve retention. Focus on product diversification.')
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4('Complaint Resolution', style={'color': COLORS['secondary']}),
                    html.P('Implement proactive complaint resolution system to improve satisfaction.')
                ])
            ], style={
                'backgroundColor': COLORS['light'],
                'padding': '30px',
                'borderRadius': '15px',
                'boxShadow': 'inset 0 2px 4px rgba(0,0,0,0.05)'
            })
        ])
    ], style={'padding': '0 40px 40px 40px'})
])

# Callback for interactive graphs with enhanced styling
@app.callback(
    Output('main-graph', 'figure'),
    Input('analysis-selector', 'value')
)
def update_graph(analysis_type):
    if analysis_type == 'geo':
        # Geographic Analysis with enhanced colors
        churn_by_geo = df.groupby('Geography')['Exited'].mean() * 100
        fig = px.bar(
            x=churn_by_geo.index,
            y=churn_by_geo.values,
            title='Churn Rate by Geography',
            labels={'x': 'Country', 'y': 'Churn Rate (%)'},
            color_discrete_sequence=[COLORS['secondary']]
        )
        
    elif analysis_type == 'age':
        # Age Distribution with custom colors
        fig = px.histogram(
            df,
            x='Age',
            color='Exited',
            marginal='box',
            title='Age Distribution by Churn Status',
            labels={'Exited': 'Customer Status'},
            color_discrete_map={0: COLORS['secondary'], 1: COLORS['accent']},
            opacity=0.7
        )
        
    elif analysis_type == 'financial':
        # Financial Analysis with custom colors
        fig = px.box(
            df,
            x='Exited',
            y='Balance',
            color='Exited',
            title='Balance Distribution by Churn Status',
            labels={'Balance': 'Account Balance ($)', 'Exited': 'Customer Status'},
            color_discrete_map={0: COLORS['secondary'], 1: COLORS['accent']}
        )
        
    else:
        # Satisfaction Analysis with custom colors
        satisfaction_data = pd.crosstab(df['Satisfaction Score'], df['Exited'], normalize='index') * 100
        fig = px.bar(
            satisfaction_data,
            title='Churn Rate by Satisfaction Score',
            labels={'index': 'Satisfaction Score', 'value': 'Percentage (%)'},
            barmode='group',
            color_discrete_sequence=[COLORS['secondary'], COLORS['accent']]
        )
    
    # Update layout for all graphs
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': COLORS['text'], 'size': 12},
        title={'font': {'size': 24, 'color': COLORS['primary']}},
        margin=dict(t=60, l=40, r=40, b=40),
        showlegend=True,
        legend={'bgcolor': 'rgba(0,0,0,0)'},
        hovermode='closest',
        transition_duration=500,
        modebar=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            color=COLORS['secondary'],
            activecolor=COLORS['accent'],
            orientation='v'
        )
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig

# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Bank Customer Retention Analysis</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
            body {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                margin: 0;
                background-color: #f8f9fa;
            }
            .metric-card {
                background-color: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                width: 22%;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .metric-card h4 {
                margin: 0;
                font-size: 1.1em;
                transition: color 0.3s ease;
            }
            .metric-card h2 {
                margin: 10px 0 0 0;
                font-size: 2em;
                transition: all 0.3s ease;
            }
            .metric-card:hover h2 {
                transform: scale(1.1);
            }
            /* Custom dropdown styling */
            .Select-control {
                border-radius: 10px !important;
                border: 1px solid #e0e0e0 !important;
                transition: all 0.3s ease !important;
            }
            .Select-control:hover {
                border-color: #3498db !important;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
            }
            /* Animate graph transitions */
            .js-plotly-plot {
                transition: all 0.3s ease;
            }
            /* Animate insights section */
            .insight-card {
                transition: all 0.3s ease;
            }
            .insight-card:hover {
                transform: translateX(10px);
            }
            /* Enhanced modebar styling */
            .modebar-btn {
                font-size: 14px !important;
                padding: 6px !important;
                transition: all 0.3s ease !important;
            }
            .modebar-btn:hover {
                background-color: rgba(52, 152, 219, 0.1) !important;
                transform: translateY(-2px) !important;
            }
            .modebar {
                border-radius: 10px !important;
                padding: 5px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            
            /* Tooltip styling */
            .plotly-notifier {
                font-family: 'Helvetica Neue', Arial, sans-serif !important;
                padding: 8px 12px !important;
                border-radius: 8px !important;
                background-color: rgba(44, 62, 80, 0.9) !important;
                border: none !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
                color: white !important;
                font-size: 14px !important;
            }
            
            /* Enhanced zoom and pan controls */
            .zoom-plot .nsewdrag {
                cursor: move !important;
            }
            
            .zoombox {
                fill: rgba(52, 152, 219, 0.1) !important;
                stroke: #3498db !important;
            }
            
            .select-outline {
                stroke: #e74c3c !important;
                stroke-width: 2px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 