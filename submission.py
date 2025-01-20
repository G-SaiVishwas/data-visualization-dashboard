import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Bank Customer Retention Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stMetric {
        background-color: #2D2D2D;
        color: #FFFFFF !important;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        background-color: #3D3D3D;
    }
    .stPlotlyChart {
        background-color: white !important;
        padding: 1rem;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        margin: 1rem 0 !important;
        overflow: hidden !important;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    h1, h2, h3, h4, h5, h6, .metric-label {
        color: #FFFFFF !important;
    }
    .metric-value {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    .stMarkdown {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    .modebar {
        background-color: white !important;
        border-radius: 8px;
        margin-top: 10px !important;
    }
    .modebar-btn {
        color: #000000 !important;
    }
    .js-plotly-plot .plotly .modebar {
        right: 10px !important;
    }
    .gtitle, .xtitle, .ytitle {
        fill: #000000 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    .xtick text, .ytick text {
        fill: #000000 !important;
        font-size: 12px !important;
    }
    .js-plotly-plot .plotly .main-svg {
        border-radius: 8px !important;
    }
    .js-plotly-plot {
        border-radius: 8px !important;
    }
    .plot-container {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def download_plot_as_html(fig, filename="plot.html"):
    """Save plot as interactive HTML file."""
    fig.write_html(filename)
    with open(filename, "rb") as file:
        btn = st.download_button(
            label="Download Plot",
            data=file,
            file_name=filename,
            mime="text/html"
        )
    if os.path.exists(filename):
        os.remove(filename)

# Update the get_plot_config function to include more options
def get_plot_config():
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape',
            'hoverClosestGeo',
            'hoverCompareCartesian',
            'toggleSpikelines'
        ],
        'modeBarButtonsToRemove': [],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart_export',
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'scrollZoom': True
    }

# Add missing helper function for consistent styling
def apply_chart_style(fig):
    """Apply consistent styling to charts."""
    fig.update_traces(
        marker=dict(line=dict(color='white', width=1)),
        selector=dict(mode='markers')
    )
    fig.update_layout(
        font=dict(color='#000000'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, l=50, r=50, b=50)
    )
    return fig

# Model Functions
def prepare_features(df):
    """Prepare features for model training or prediction."""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Initialize label encoders
    le_geography = LabelEncoder()
    le_gender = LabelEncoder()
    
    # Fit and transform categorical variables
    df['Geography'] = le_geography.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    # Save label encoders
    joblib.dump(le_geography, 'models/le_geography.joblib')
    joblib.dump(le_gender, 'models/le_gender.joblib')
    
    # Select features for model
    feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                   'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                   'EstimatedSalary', 'Complain', 'Satisfaction Score']
    
    return df[feature_cols]

def train_model():
    """Train and save the churn prediction model."""
    # Load data
    df = pd.read_csv('Customer-Churn-Records.csv')
    
    # Prepare features
    X = prepare_features(df)
    y = df['Exited']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LGBMClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'models/churn_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    return model, scaler, X_test, y_test

def predict_churn(data):
    """Make churn predictions for new data."""
    # Check if model exists, if not train it
    if not os.path.exists('models/churn_model.joblib'):
        train_model()
    
    # Load model and preprocessing objects
    model = joblib.load('models/churn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    le_geography = joblib.load('models/le_geography.joblib')
    le_gender = joblib.load('models/le_gender.joblib')
    
    # Prepare input data
    data['Geography'] = le_geography.transform(data['Geography'])
    data['Gender'] = le_gender.transform(data['Gender'])
    
    # Scale features
    data_scaled = scaler.transform(data)
    
    # Make prediction
    pred_proba = model.predict_proba(data_scaled)
    return pred_proba[:, 1]

def get_feature_importance():
    """Get feature importance from the trained model."""
    # Load model and feature names
    model = joblib.load('models/churn_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame with feature importance
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    feature_imp = feature_imp.sort_values('importance', ascending=True)
    
    return feature_imp

# Dashboard Helper Functions
def create_figure_layout(title, height=None):
    """Create a standardized layout for figures."""
    layout = dict(
        title=dict(
            text=title,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(
                color='#000000',
                size=20
            )
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            color='#000000',
            size=14
        ),
        margin=dict(t=80, l=50, r=50, b=50),
        xaxis=dict(
            title_font=dict(color='#000000', size=16),
            tickfont=dict(color='#000000', size=12),
            gridcolor='rgba(128, 128, 128, 0.2)',
            linecolor='rgba(0, 0, 0, 0.3)'
        ),
        yaxis=dict(
            title_font=dict(color='#000000', size=16),
            tickfont=dict(color='#000000', size=12),
            gridcolor='rgba(128, 128, 128, 0.2)',
            linecolor='rgba(0, 0, 0, 0.3)'
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family="Arial"
        ),
        modebar=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            color='#000000',
            activecolor='#4CAF50'
        )
    )
    if height:
        layout['height'] = height
    return layout

def update_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#000000', size=14),
        title=dict(font=dict(color='#000000', size=20)),
        margin=dict(t=80, l=50, r=50, b=50)
    )
    fig.update_xaxes(gridcolor='#eee', title_font=dict(color='#000000', size=14))
    fig.update_yaxes(gridcolor='#eee', title_font=dict(color='#000000', size=14))
    return fig

# Custom color scheme
colors = {
    'primary': '#1f77b4',
    'secondary': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'background': '#ffffff'
}

# Load data
@st.cache_data
def load_data():
    """Load and cache the customer data."""
    return pd.read_csv('Customer-Churn-Records.csv')

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Check if model needs to be trained
if not os.path.exists('models/churn_model.joblib'):
    with st.spinner('Training model... Please wait.'):
        train_model()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["📊 Overview",
     "👥 Customer Demographics",
     "💰 Financial Analysis",
     "⭐ Satisfaction Analysis",
     "🔮 Churn Prediction",
     "📈 Interactive Analysis",
     "❓ Help & Guide"]
)

# Sidebar with additional information
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown("""
This dashboard provides comprehensive analysis of customer churn patterns 
and predictive insights for bank customer retention.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")
st.sidebar.write(f"Total Customers: {len(df):,}")
st.sidebar.write(f"Churned Customers: {df['Exited'].sum():,}")
st.sidebar.write(f"Countries: {', '.join(df['Geography'].unique())}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
📊 **Overview**: Key metrics and geographic analysis  
👥 **Demographics**: Age and gender distribution  
💰 **Financial**: Balance and credit score analysis  
⭐ **Satisfaction**: Customer satisfaction insights  
🔮 **Prediction**: Churn probability calculator  
📈 **Interactive**: Custom analysis tools  
❓ **Help**: User guide and documentation
""")

# Add tooltip information
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
💡 **Tips**:
- Click and drag to zoom into charts
- Double-click to reset zoom
- Use the drawing tools to annotate
- Download charts for presentations
</small>
""", unsafe_allow_html=True)

# Main content
if page == "📊 Overview":
    st.title("Bank Customer Retention Analysis")
    
    # Key Metrics
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = df['Exited'].mean() * 100
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%",
            f"{churn_rate - 20:.1f}%" if churn_rate > 20 else f"{20 - churn_rate:.1f}%"
        )
    
    with col2:
        active_members = df['IsActiveMember'].mean() * 100
        st.metric(
            "Active Members",
            f"{active_members:.1f}%",
            f"{active_members - 50:.1f}%" if active_members > 50 else f"{50 - active_members:.1f}%"
        )
    
    with col3:
        avg_satisfaction = df['Satisfaction Score'].mean()
        st.metric(
            "Avg Satisfaction",
            f"{avg_satisfaction:.2f}",
            f"{avg_satisfaction - 3:.2f}" if avg_satisfaction > 3 else f"{3 - avg_satisfaction:.2f}"
        )
    
    with col4:
        complaint_rate = df['Complain'].mean() * 100
        st.metric(
            "Complaint Rate",
            f"{complaint_rate:.1f}%",
            f"{complaint_rate - 20:.1f}%" if complaint_rate > 20 else f"{20 - complaint_rate:.1f}%"
        )
    
    # Geographic Analysis
    st.subheader("Geographic Distribution")
    
    # Aggregate data by geography
    geo_data = df.groupby('Geography').agg({
        'Exited': ['mean', 'count'],
        'CreditScore': 'mean',
        'Balance': 'mean'
    }).round(2)
    
    # Create scatter geo plot
    fig_map = go.Figure()
    
    # Add scatter geo trace
    fig_map.add_trace(go.Scattergeo(
        locations=['FRA', 'DEU', 'ESP'],  # ISO-3 country codes
        text=geo_data.index,
        mode='markers+text',
        marker=dict(
            size=geo_data[('Exited', 'count')] / 50,
            color=geo_data[('Exited', 'mean')],
            colorscale=[[0, colors['primary']], [1, colors['danger']]],
            showscale=True,
            colorbar_title="Churn Rate",
            line=dict(color='white', width=1)
        ),
        textposition='top center',
        hovertemplate="<b>%{text}</b><br>" +
                     "Churn Rate: %{marker.color:.1%}<br>" +
                     "Customers: %{marker.size*50:,.0f}<br>" +
                     "<extra></extra>"
    ))
    
    # Update layout
    fig_map.update_layout(
        title=dict(
            text="Customer Distribution and Churn Rate by Country",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(color='#000000', size=20)
        ),
        geo=dict(
            scope='europe',
            showland=True,
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            projection_type='mercator',
            bgcolor='white',
            showcoastlines=True,
            coastlinecolor='rgb(150, 150, 150)',
            showframe=True,
            framecolor='rgb(150, 150, 150)'
        ),
        height=500
    )
    
    fig_map = update_plotly_layout(fig_map)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(fig_map, use_container_width=True, config=get_plot_config())
    with col2:
        st.markdown("""
        **Chart Guide**:
        - Bubble size: Customer count
        - Color: Churn rate
        - Hover for details
        - Use toolbar to:
            - Pan
            - Zoom
            - Draw
            - Download
        """)
        download_plot_as_html(fig_map, "geographic_distribution.html")

    # Churn Trends
    st.subheader("Churn Analysis by Key Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution with Density Plot
        fig_age = go.Figure()
        
        for exited, color in zip([0, 1], [colors['primary'], colors['danger']]):
            age_data = df[df['Exited'] == exited]['Age']
            fig_age.add_trace(go.Violin(
                x=['Retained' if exited == 0 else 'Churned'] * len(age_data),
                y=age_data,
                name='Retained' if exited == 0 else 'Churned',
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                fillcolor=color
            ))
        
        fig_age.update_layout(**create_figure_layout("Age Distribution by Churn Status"))
        
        fig_age.update_traces(
            box=dict(
                fillcolor='white',
                line=dict(color='#000000', width=1)
            ),
            meanline=dict(color='#000000', width=1),
            line=dict(color='#000000', width=1)
        )
        
        fig_age = update_plotly_layout(fig_age)
        st.plotly_chart(fig_age, use_container_width=True, config=get_plot_config())
    
    with col2:
        # Product Usage Sunburst
        product_churn = df.groupby(['NumOfProducts', 'Exited']).size().reset_index(name='count')
        fig_products = px.sunburst(
            product_churn,
            path=['NumOfProducts', 'Exited'],
            values='count',
            title="Product Usage and Churn Distribution",
            color='Exited',
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
        
        fig_products.update_layout(
            title=dict(
                text="Product Usage and Churn Distribution",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(color='#000000', size=20)
            ),
            font=dict(color='#000000', size=14)
        )
        
        fig_products = update_plotly_layout(fig_products)
        st.plotly_chart(fig_products, use_container_width=True, config=get_plot_config())

elif page == "👥 Customer Demographics":
    st.title("Customer Demographics Analysis")
    
    # Age and Gender Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive Age Histogram
        age_bins = st.slider("Select Age Range", 
                           min_value=int(df['Age'].min()),
                           max_value=int(df['Age'].max()),
                           value=(30, 50))
        
        filtered_df = df[(df['Age'] >= age_bins[0]) & (df['Age'] <= age_bins[1])]
        
        fig_age = px.histogram(
            filtered_df,
            x="Age",
            color="Exited",
            marginal="box",
            nbins=30,
            title=f"Age Distribution ({age_bins[0]}-{age_bins[1]} years)",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
        
        fig_age = update_plotly_layout(fig_age)
        st.plotly_chart(fig_age, use_container_width=True, config=get_plot_config())
    
    with col2:
        # Gender Distribution with Donut Chart
        gender_data = df.groupby(['Gender', 'Exited']).size().unstack()
        
        fig_gender = go.Figure(data=[
            go.Pie(
                labels=gender_data.index,
                values=gender_data[0],
                name="Retained",
                hole=0.6,
                domain={'x': [0, 0.45]},
                marker_colors=[colors['primary'], colors['secondary']]
            ),
            go.Pie(
                labels=gender_data.index,
                values=gender_data[1],
                name="Churned",
                hole=0.6,
                domain={'x': [0.55, 1]},
                marker_colors=[colors['warning'], colors['danger']]
            )
        ])
        
        fig_gender.update_layout(
            title="Gender Distribution: Retained vs Churned",
            annotations=[
                dict(text="Retained", x=0.20, y=0.5, font_size=12, showarrow=False),
                dict(text="Churned", x=0.80, y=0.5, font_size=12, showarrow=False)
            ]
        )
        
        fig_gender = update_plotly_layout(fig_gender)
        st.plotly_chart(fig_gender, use_container_width=True, config=get_plot_config())

elif page == "💰 Financial Analysis":
    st.title("Financial Metrics Analysis")
    
    # Balance Distribution
    st.subheader("Account Balance Analysis")
    
    # Interactive Balance Range Selection
    balance_range = st.slider(
        "Select Balance Range",
        min_value=float(df['Balance'].min()),
        max_value=float(df['Balance'].max()),
        value=(0.0, 250000.0),
        format="%.2f"
    )
    
    filtered_df = df[(df['Balance'] >= balance_range[0]) & (df['Balance'] <= balance_range[1])]
    
    # Create violin plot with individual points
    fig_balance = go.Figure()
    
    for exited, color in zip([0, 1], [colors['primary'], colors['danger']]):
        balance_data = filtered_df[filtered_df['Exited'] == exited]['Balance']
        
        fig_balance.add_trace(go.Violin(
            x=['Retained' if exited == 0 else 'Churned'] * len(balance_data),
            y=balance_data,
            name='Retained' if exited == 0 else 'Churned',
            box_visible=True,
            meanline_visible=True,
            points="all",
            line_color=color,
            fillcolor=color
        ))
    
    fig_balance = update_plotly_layout(fig_balance)
    st.plotly_chart(fig_balance, use_container_width=True, config=get_plot_config())
    
    # After balance plot
    st.markdown("""
    **Chart Guide**:
    - Violin plot shows distribution shape
    - Box plot shows statistical summary
    - Points show individual customers
    - Use slider to filter balance range
    """)
    download_plot_as_html(fig_balance, "balance_distribution.html")
    
    # Credit Score Analysis
    st.subheader("Credit Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score Distribution
        fig_credit = px.histogram(
            df,
            x="CreditScore",
            color="Exited",
            marginal="box",
            nbins=50,
            title="Credit Score Distribution",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
        
        fig_credit.update_traces(
            opacity=0.75,
            marker=dict(line=dict(color='white', width=1))
        )
        
        fig_credit = update_plotly_layout(fig_credit)
        st.plotly_chart(fig_credit, use_container_width=True, config=get_plot_config())
    
    # After credit score plots
    with col1:
        st.markdown("""
        **Chart Guide**:
        - Histogram shows score distribution
        - Box plot shows quartiles
        - Color indicates churn status
        - Use toolbar to zoom/analyze
        """)
        download_plot_as_html(fig_credit, "credit_score_distribution.html")
    
    with col2:
        # Credit Score vs Balance Scatter
        fig_scatter = px.scatter(
            df,
            x="CreditScore",
            y="Balance",
            color="Exited",
            size="Age",
            hover_data=["Geography", "Gender"],
            title="Credit Score vs Balance",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
        
        fig_scatter.update_traces(
            marker=dict(
                line=dict(color='white', width=1),
                opacity=0.7
            )
        )
        
        fig_scatter = update_plotly_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True, config=get_plot_config())
    
    # After scatter plot
    with col2:
        st.markdown("""
        **Chart Guide**:
        - Point size shows customer age
        - Color shows churn status
        - Hover for detailed information
        - Use toolbar to select regions
        """)
        download_plot_as_html(fig_scatter, "credit_vs_balance.html")

elif page == "⭐ Satisfaction Analysis":
    st.title("Customer Satisfaction Analysis")
    
    # Satisfaction Score Distribution
    st.subheader("Satisfaction Score Distribution")
    
    # Interactive Satisfaction Analysis
    satisfaction_selection = st.multiselect(
        "Select Satisfaction Scores to Analyze",
        options=sorted(df['Satisfaction Score'].unique()),
        default=sorted(df['Satisfaction Score'].unique())
    )
    
    filtered_df = df[df['Satisfaction Score'].isin(satisfaction_selection)]
    
    # Create an interactive sunburst chart
    satisfaction_data = filtered_df.groupby(['Satisfaction Score', 'Complain', 'Exited']).size().reset_index(name='count')
    
    fig_satisfaction = px.sunburst(
        satisfaction_data,
        path=['Satisfaction Score', 'Complain', 'Exited'],
        values='count',
        title="Satisfaction Score, Complaints, and Churn Relationship",
        color='Exited',
        color_discrete_map={0: colors['primary'], 1: colors['danger']}
    )
    
    fig_satisfaction = update_plotly_layout(fig_satisfaction)
    st.plotly_chart(fig_satisfaction, use_container_width=True, config=get_plot_config())
    
    # After satisfaction sunburst chart
    st.markdown("""
    **Chart Guide**:
    - Click segments to explore deeper levels
    - Outer ring shows churn status
    - Middle ring shows complaints
    - Inner ring shows satisfaction
    - Double-click to zoom out
    """)
    download_plot_as_html(fig_satisfaction, "satisfaction_analysis.html")
    
    # Complaint Analysis
    st.subheader("Complaint Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Complaints by Geography
        complaint_geo = df.groupby('Geography')['Complain'].mean() * 100
        
        fig_complaint_geo = go.Figure(data=[
            go.Bar(
                x=complaint_geo.index,
                y=complaint_geo.values,
                text=complaint_geo.values.round(1),
                textposition='auto',
                marker_color=colors['warning']
            )
        ])
        
        fig_complaint_geo.update_layout(
            title="Complaint Rate by Geography",
            xaxis_title="Country",
            yaxis_title="Complaint Rate (%)"
        )
        
        fig_complaint_geo = update_plotly_layout(fig_complaint_geo)
        st.plotly_chart(fig_complaint_geo, use_container_width=True, config=get_plot_config())
    
    # After complaint analysis charts
    with col1:
        st.markdown("""
        **Chart Guide**:
        - Bar height shows complaint rate
        - Hover for exact percentages
        - Compare across countries
        - Download for reporting
        """)
        download_plot_as_html(fig_complaint_geo, "complaint_by_geography.html")
    
    with col2:
        # Satisfaction Score Trends
        satisfaction_trend = df.groupby('Satisfaction Score')['Exited'].mean() * 100
        
        fig_satisfaction_trend = go.Figure(data=[
            go.Scatter(
                x=satisfaction_trend.index,
                y=satisfaction_trend.values,
                mode='lines+markers',
                line=dict(width=3, color=colors['primary']),
                marker=dict(size=10, color=colors['secondary'])
            )
        ])
        
        fig_satisfaction_trend.update_layout(
            title="Churn Rate by Satisfaction Score",
            xaxis_title="Satisfaction Score",
            yaxis_title="Churn Rate (%)"
        )
        
        fig_satisfaction_trend = update_plotly_layout(fig_satisfaction_trend)
        st.plotly_chart(fig_satisfaction_trend, use_container_width=True, config=get_plot_config())
    
    # After satisfaction trend chart
    with col2:
        st.markdown("""
        **Chart Guide**:
        - Line shows churn trend
        - Points show exact values
        - Hover for details
        - Analyze satisfaction impact
        """)
        download_plot_as_html(fig_satisfaction_trend, "satisfaction_trend.html")

elif page == "🔮 Churn Prediction":
    st.title("Customer Churn Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
    
    with col2:
        geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        has_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        complain = st.checkbox("Has Complained", value=False)
    
    # Create a dataframe with the input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [int(has_card)],
        'IsActiveMember': [int(is_active)],
        'EstimatedSalary': [salary],
        'Complain': [int(complain)],
        'Satisfaction Score': [satisfaction]
    })
    
    # Make prediction
    if st.button("Predict Churn Probability", key="predict_button"):
        try:
            churn_prob = predict_churn(input_data)[0]
            
            # Create a gauge chart for the prediction
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': colors['primary']},
                    'steps': [
                        {'range': [0, 30], 'color': colors['primary']},
                        {'range': [30, 70], 'color': colors['warning']},
                        {'range': [70, 100], 'color': colors['danger']}
                    ],
                    'threshold': {
                        'line': {'color': colors['danger'], 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig = update_plotly_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=get_plot_config())
            
            # Display risk factors
            st.subheader("Risk Factors")
            feature_imp = get_feature_importance()
            
            fig_importance = px.bar(
                feature_imp.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Factors Influencing Churn",
                color_discrete_sequence=[colors['primary']]
            )
            
            fig_importance = update_plotly_layout(fig_importance)
            st.plotly_chart(fig_importance, use_container_width=True, config=get_plot_config())
            
            # Provide recommendations based on the prediction
            st.subheader("Recommendations")
            if churn_prob > 0.7:
                st.error("⚠️ High Risk of Churn")
                st.markdown("""
                ### Immediate Actions Required:
                1. Schedule a personal consultation
                2. Review pricing and fees
                3. Offer loyalty rewards
                4. Conduct satisfaction survey
                5. Consider premium service upgrades
                """)
            elif churn_prob > 0.3:
                st.warning("⚠️ Moderate Risk of Churn")
                st.markdown("""
                ### Recommended Actions:
                1. Send customer satisfaction survey
                2. Review product usage patterns
                3. Consider promotional offers
                4. Monitor engagement levels
                """)
            else:
                st.success("✅ Low Risk of Churn")
                st.markdown("""
                ### Retention Strategy:
                1. Continue regular engagement
                2. Offer product upgrades
                3. Consider loyalty rewards
                4. Maintain service quality
                """)
            
            # After prediction gauge
            st.markdown("""
            **Chart Guide**:
            - Green zone: Low risk (0-30%)
            - Yellow zone: Medium risk (30-70%)
            - Red zone: High risk (70-100%)
            - Value shows exact probability
            """)
            download_plot_as_html(fig, "churn_prediction.html")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

elif page == "📈 Interactive Analysis":
    st.title("Interactive Analysis")
    
    # Feature Selection
    st.subheader("Custom Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            "Select X-axis feature",
            options=['Age', 'CreditScore', 'Balance', 'Tenure', 'NumOfProducts', 'Satisfaction Score']
        )
    
    with col2:
        y_axis = st.selectbox(
            "Select Y-axis feature",
            options=['Balance', 'CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'Satisfaction Score']
        )
    
    # Plot Type Selection
    plot_type = st.radio(
        "Select Plot Type",
        options=['Scatter', 'Box', 'Violin', 'Histogram']
    )
    
    # Create the selected plot
    if plot_type == 'Scatter':
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color='Exited',
            size='Balance',
            hover_data=['Geography', 'Gender'],
            title=f"{x_axis} vs {y_axis}",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
    
    elif plot_type == 'Box':
        fig = px.box(
            df,
            x='Exited',
            y=x_axis if x_axis != y_axis else y_axis,
            color='Exited',
            title=f"{x_axis if x_axis != y_axis else y_axis} Distribution by Churn Status",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
    
    elif plot_type == 'Violin':
        fig = px.violin(
            df,
            x='Exited',
            y=x_axis if x_axis != y_axis else y_axis,
            color='Exited',
            box=True,
            title=f"{x_axis if x_axis != y_axis else y_axis} Distribution by Churn Status",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
    
    else:  # Histogram
        fig = px.histogram(
            df,
            x=x_axis if x_axis != y_axis else y_axis,
            color='Exited',
            marginal='box',
            title=f"{x_axis if x_axis != y_axis else y_axis} Distribution",
            color_discrete_map={0: colors['primary'], 1: colors['danger']}
        )
    
    fig = update_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plot_config())
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                   'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited',
                   'Complain', 'Satisfaction Score']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale=[[0, colors['danger']], [0.5, colors['warning']], [1, colors['primary']]],
        aspect='auto'
    )
    
    fig_corr = update_plotly_layout(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True, config=get_plot_config())
    
    # After correlation matrix
    st.markdown("""
    **Chart Guide**:
    - Red shows negative correlation
    - Blue shows positive correlation
    - Darker colors indicate stronger relationships
    - Hover for exact correlation values
    """)
    download_plot_as_html(fig_corr, "correlation_matrix.html")

elif page == "❓ Help & Guide":
    st.title("How to Use This Dashboard")
    
    # Add tabs for different help sections
    help_tab = st.tabs(["Navigation", "Features", "Interactions", "Tips", "FAQ"])
    
    with help_tab[0]:
        st.header("🗺️ Navigation Guide")
        st.markdown("""
        ### Using the Dashboard
        1. Use the sidebar menu to navigate between different sections
        2. Each section provides unique insights and analysis capabilities
        3. Hover over charts for detailed information
        4. Use the interactive features to customize your analysis
        """)
    
    with help_tab[1]:
        st.header("🎯 Key Features")
        st.markdown("""
        ### Available Analysis Tools
        - **Overview Dashboard**: Key metrics and geographic insights
        - **Customer Demographics**: Age and gender analysis
        - **Financial Analysis**: Balance and credit score patterns
        - **Satisfaction Analysis**: Customer satisfaction metrics
        - **Churn Prediction**: ML-powered prediction tool
        - **Interactive Analysis**: Custom visualization builder
        """)
    
    with help_tab[2]:
        st.header("🔄 Interactive Features")
        st.markdown("""
        ### Chart Interactions
        - Click and drag to zoom
        - Double-click to reset view
        - Click legend items to filter
        - Use the modebar for additional tools
        - Download charts as PNG or HTML
        """)
    
    with help_tab[3]:
        st.header("💡 Pro Tips")
        st.markdown("""
        ### Making the Most of the Dashboard
        1. Use filters to focus on specific customer segments
        2. Compare multiple metrics for deeper insights
        3. Export visualizations for presentations
        4. Check the correlation matrix for relationships
        5. Use the prediction tool for risk assessment
        """)
    
    with help_tab[4]:
        st.header("❓ Frequently Asked Questions")
        st.markdown("""
        ### Common Questions
        **Q: How is churn rate calculated?**  
        A: Churn rate is the percentage of customers who have exited (Exited = 1).
        
        **Q: What does the satisfaction score mean?**  
        A: Satisfaction score ranges from 1-5, with 5 being the highest satisfaction level.
        
        **Q: How accurate is the prediction model?**  
        A: The model is regularly trained on historical data and provides probability estimates.
        
        **Q: Can I download the visualizations?**  
        A: Yes, use the download button or the modebar camera icon.
        """)

# Error handling for missing data
if 'Satisfaction Score' not in df.columns:
    st.error("Error: 'Satisfaction Score' column not found in the dataset. Please ensure the data file contains all required columns.")
    st.stop()

if 'Complain' not in df.columns:
    st.error("Error: 'Complain' column not found in the dataset. Please ensure the data file contains all required columns.")
    st.stop()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <p>🏦 Bank Customer Retention Analysis Dashboard | Created with ❤️ using Streamlit and Plotly</p>
    <p>Version 1.0.0 | © 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True) 