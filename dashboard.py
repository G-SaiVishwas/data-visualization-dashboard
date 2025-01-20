import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import predict_churn, get_feature_importance
import joblib
import os

# Create models directory and train model if needed
if not os.path.exists('models'):
    os.makedirs('models')
    from model import train_model
    train_model()
elif not os.path.exists('models/churn_model.joblib') or \
     not os.path.exists('models/le_geography.joblib') or \
     not os.path.exists('models/le_gender.joblib'):
    from model import train_model
    train_model()

# Define helper functions first
def create_figure_layout(title, height=None):
    layout = dict(
        title=dict(
            text=title,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(
                color='#FFFFFF',
                size=20
            )
        ),
        paper_bgcolor='#2D2D2D',
        plot_bgcolor='#2D2D2D',
        font=dict(
            color='#FFFFFF',
            size=14
        ),
        margin=dict(t=80, l=50, r=50, b=50),
        xaxis=dict(
            title_font=dict(color='#FFFFFF', size=16),
            tickfont=dict(color='#FFFFFF', size=12),
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            title_font=dict(color='#FFFFFF', size=16),
            tickfont=dict(color='#FFFFFF', size=12),
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    if height:
        layout['height'] = height
    return layout

def get_plot_config():
    return {
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': [],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart_export',
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'displayModeBar': True,
        'scrollZoom': True
    }

def update_plotly_layout(fig):
    fig.update_layout(
        paper_bgcolor='#2D2D2D',
        plot_bgcolor='#2D2D2D',
        font=dict(
            color='#FFFFFF',
            size=14
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(255, 255, 255, 0.2)',
            title_font=dict(color='#FFFFFF', size=16),
            tickfont=dict(color='#FFFFFF', size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(255, 255, 255, 0.2)',
            title_font=dict(color='#FFFFFF', size=16),
            tickfont=dict(color='#FFFFFF', size=12)
        ),
        margin=dict(t=80, l=50, r=50, b=50),
        modebar=dict(
            bgcolor='rgba(45, 45, 45, 0.9)',
            color='#FFFFFF',
            activecolor='#4CAF50'
        )
    )
    return fig

# Custom color scheme
colors = {
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'warning': '#FFC107',
    'danger': '#F44336',
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'plot_colors': ['#4CAF50', '#2196F3', '#FFC107', '#F44336', '#9C27B0']
}

# Page config
st.set_page_config(
    page_title="Bank Customer Retention Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main container and background */
    .main {
        background-color: #1E1E1E;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #252525;
    }
    
    /* Metric cards */
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
    
    /* Plot containers */
    .stPlotlyChart {
        background-color: #2D2D2D !important;
        padding: 1rem;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        margin: 1rem 0 !important;
    }
    
    /* Buttons */
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
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    .stMarkdown {
        color: #FFFFFF !important;
    }
    p {
        color: #FFFFFF !important;
    }
    
    /* Metric values and labels */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    
    /* Input widgets */
    .stSelectbox select {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #404040;
    }
    .stSlider {
        color: #FFFFFF;
    }
    .stMultiSelect {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #FFFFFF !important;
    }
    
    /* Plotly modebar */
    .modebar {
        background-color: rgba(45, 45, 45, 0.9) !important;
        border-radius: 4px !important;
        margin-top: 10px !important;
        border: 1px solid #404040 !important;
    }
    .modebar-btn {
        color: #FFFFFF !important;
    }
    .modebar-btn:hover {
        color: #4CAF50 !important;
    }
    .modebar-btn.active {
        color: #4CAF50 !important;
    }
    
    /* Plot elements */
    .gtitle, .xtitle, .ytitle {
        fill: #FFFFFF !important;
    }
    .xtick text, .ytick text {
        fill: #FFFFFF !important;
    }
    
    /* Sidebar navigation */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Slider input */
    .stSlider input {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
        border: 1px solid #404040 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
    }
    
    /* MultiSelect */
    .stMultiSelect > div > div {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Customer-Churn-Records.csv')
    # Map Exited values to readable labels
    df['Churn_Status'] = df['Exited'].map({0: 'Non-Churned', 1: 'Churned'})
    return df

df = load_data()

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Overview",
    "👥 Customer Demographics",
    "💰 Financial Analysis",
    "⭐ Satisfaction Analysis",
    "🔮 Churn Prediction",
    "📈 Interactive Analysis",
    "❓ Help & Guide"
])

# Overview Page
if page == "📊 Overview":
    st.title("🏦 Bank Customer Retention Analysis")
    
    # Key Metrics in a single row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = df['Exited'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%",
                 delta=f"{churn_rate - 20:.1f}%" if churn_rate > 20 else f"{20 - churn_rate:.1f}%",
                 delta_color="inverse")
    
    with col2:
        active_rate = df['IsActiveMember'].mean() * 100
        st.metric("Active Members", f"{active_rate:.1f}%",
                 delta=f"{active_rate - 50:.1f}%",
                 delta_color="normal")
    
    with col3:
        avg_satisfaction = df['Satisfaction Score'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5",
                 delta=f"{avg_satisfaction - 3:.1f}",
                 delta_color="normal")
    
    with col4:
        complaint_rate = df['Complain'].mean() * 100
        st.metric("Complaint Rate", f"{complaint_rate:.1f}%",
                 delta=f"{20 - complaint_rate:.1f}%",
                 delta_color="inverse")
    
    # Geographic Analysis with Interactive Map
    st.subheader("Geographic Distribution of Churn")
    geo_data = df.groupby('Geography').agg({
        'Exited': ['count', 'mean'],
        'CreditScore': 'mean',
        'Balance': 'mean'
    }).round(2)
    
    fig = go.Figure(data=[
        go.Scattergeo(
            locations=['DEU', 'FRA', 'ESP'],
            text=geo_data.index,
            mode='markers+text',
            marker=dict(
                size=geo_data[('Exited', 'count')] / 50,
                color=geo_data[('Exited', 'mean')],
                colorscale=[[0, colors['primary']], [1, colors['danger']]],
                showscale=True,
                colorbar_title="Churn Rate"
            ),
            hovertemplate="<b>%{text}</b><br>" +
                         "Churn Rate: %{marker.color:.1%}<br>" +
                         "Customers: %{marker.size*50:,.0f}<br>" +
                         "<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Customer Distribution and Churn Rate by Country",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(color='#FFFFFF', size=20)
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
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#FFFFFF', size=14),
        height=500,
        margin=dict(t=80, l=50, r=50, b=50)
    )
    
    fig = update_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plot_config())
    
    # Churn Trends
    st.subheader("Churn Analysis by Key Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution with Density Plot
        fig_age = go.Figure()
        
        for status, color in zip(['Non-Churned', 'Churned'], [colors['primary'], colors['danger']]):
            age_data = df[df['Churn_Status'] == status]['Age']
            fig_age.add_trace(go.Violin(
                x=[status] * len(age_data),
                y=age_data,
                name=status,
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                fillcolor=color
            ))
        
        fig_age.update_layout(**create_figure_layout("Age Distribution by Churn Status"))
        
        fig_age.update_traces(
            box=dict(
                fillcolor='white',
                line=dict(color='#FFFFFF', width=1)
            ),
            meanline=dict(color='#FFFFFF', width=1),
            line=dict(color='#FFFFFF', width=1)
        )
        
        fig_age = update_plotly_layout(fig_age)
        st.plotly_chart(fig_age, use_container_width=True, config=get_plot_config())
    
    with col2:
        # Product Usage Sunburst
        product_churn = df.groupby(['NumOfProducts', 'Churn_Status']).size().reset_index(name='count')
        fig_products = px.sunburst(
            product_churn,
            path=['NumOfProducts', 'Churn_Status'],
            values='count',
            title="Product Usage and Churn Distribution",
            color='Churn_Status',
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
        
        fig_products.update_layout(
            title=dict(
                text="Product Usage and Churn Distribution",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(color='#FFFFFF', size=20)
            ),
            font=dict(color='#FFFFFF', size=14)
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
            color="Churn_Status",
            marginal="box",
            nbins=30,
            title=f"Age Distribution ({age_bins[0]}-{age_bins[1]} years)",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
        
        fig_age.update_layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
        
        fig_age = update_plotly_layout(fig_age)
        st.plotly_chart(fig_age, use_container_width=True, config=get_plot_config())
    
    with col2:
        # Gender Distribution with Donut Chart
        gender_data = df.groupby(['Gender', 'Churn_Status']).size().unstack()
        
        fig_gender = go.Figure(data=[
            go.Pie(
                labels=gender_data.index,
                values=gender_data['Non-Churned'],
                name="Non-Churned",
                hole=0.6,
                domain={'x': [0, 0.45]},
                marker_colors=[colors['primary'], colors['secondary']]
            ),
            go.Pie(
                labels=gender_data.index,
                values=gender_data['Churned'],
                name="Churned",
                hole=0.6,
                domain={'x': [0.55, 1]},
                marker_colors=[colors['warning'], colors['danger']]
            )
        ])
        
        fig_gender.update_layout(
            title="Gender Distribution: Non-Churned vs Churned",
            annotations=[
                dict(text="Non-Churned", x=0.20, y=0.5, font_size=12, showarrow=False, font_color=colors['text']),
                dict(text="Churned", x=0.80, y=0.5, font_size=12, showarrow=False, font_color=colors['text'])
            ],
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text'])
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
    
    for status, color in zip(['Non-Churned', 'Churned'], [colors['primary'], colors['danger']]):
        balance_data = filtered_df[filtered_df['Churn_Status'] == status]['Balance']
        
        fig_balance.add_trace(go.Violin(
            x=[status] * len(balance_data),
            y=balance_data,
            name=status,
            box_visible=True,
            meanline_visible=True,
            points="all",
            line_color=color,
            fillcolor=color
        ))
    
    fig_balance.update_layout(
        title="Balance Distribution by Churn Status",
        xaxis_title="Customer Status",
        yaxis_title="Balance",
        showlegend=True,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'])
    )
    
    fig_balance = update_plotly_layout(fig_balance)
    st.plotly_chart(fig_balance, use_container_width=True, config=get_plot_config())
    
    # Credit Score Analysis
    st.subheader("Credit Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score Distribution
        fig_credit = px.histogram(
            df,
            x="CreditScore",
            color="Churn_Status",
            marginal="box",
            nbins=50,
            title="Credit Score Distribution",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
        
        fig_credit.update_layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
        
        fig_credit.update_traces(
            opacity=0.75,
            marker=dict(line=dict(color='white', width=1))
        )
        
        fig_credit = update_plotly_layout(fig_credit)
        st.plotly_chart(fig_credit, use_container_width=True, config=get_plot_config())
    
    with col2:
        # Credit Score vs Balance Scatter
        fig_scatter = px.scatter(
            df,
            x="CreditScore",
            y="Balance",
            color="Churn_Status",
            size="Age",
            hover_data=["Geography", "Gender"],
            title="Credit Score vs Balance",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
        
        fig_scatter.update_layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
        
        fig_scatter.update_traces(
            marker=dict(
                line=dict(color='white', width=1),
                opacity=0.7
            )
        )
        
        fig_scatter = update_plotly_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True, config=get_plot_config())

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
    satisfaction_data = filtered_df.groupby(['Satisfaction Score', 'Complain', 'Churn_Status']).size().reset_index(name='count')
    
    fig_satisfaction = px.sunburst(
        satisfaction_data,
        path=['Satisfaction Score', 'Complain', 'Churn_Status'],
        values='count',
        title="Satisfaction Score, Complaints, and Churn Relationship",
        color='Churn_Status',
        color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
    )
    
    fig_satisfaction.update_layout(
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'])
    )
    
    fig_satisfaction = update_plotly_layout(fig_satisfaction)
    st.plotly_chart(fig_satisfaction, use_container_width=True, config=get_plot_config())
    
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
            yaxis_title="Complaint Rate (%)",
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
        
        fig_complaint_geo = update_plotly_layout(fig_complaint_geo)
        st.plotly_chart(fig_complaint_geo, use_container_width=True, config=get_plot_config())
    
    with col2:
        # Satisfaction Score Trends
        satisfaction_trend = df.groupby('Satisfaction Score')['Churn_Status'].mean() * 100
        
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
            yaxis_title="Churn Rate (%)",
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
        
        fig_satisfaction_trend = update_plotly_layout(fig_satisfaction_trend)
        st.plotly_chart(fig_satisfaction_trend, use_container_width=True, config=get_plot_config())

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
            
            fig.update_layout(
                paper_bgcolor=colors['background'],
                plot_bgcolor=colors['background'],
                font=dict(color=colors['text'])
            )
            
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
            
            fig_importance.update_layout(
                paper_bgcolor=colors['background'],
                plot_bgcolor=colors['background'],
                font=dict(color=colors['text'])
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
            color='Churn_Status',
            size='Balance',
            hover_data=['Geography', 'Gender'],
            title=f"{x_axis} vs {y_axis}",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
    
    elif plot_type == 'Box':
        fig = px.box(
            df,
            x='Churn_Status',
            y=x_axis if x_axis != y_axis else y_axis,
            color='Churn_Status',
            title=f"{x_axis if x_axis != y_axis else y_axis} Distribution by Churn Status",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
    
    elif plot_type == 'Violin':
        fig = px.violin(
            df,
            x='Churn_Status',
            y=x_axis if x_axis != y_axis else y_axis,
            color='Churn_Status',
            box=True,
            title=f"{x_axis if x_axis != y_axis else y_axis} Distribution by Churn Status",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
    
    else:  # Histogram
        fig = px.histogram(
            df,
            x=x_axis if x_axis != y_axis else y_axis,
            color='Churn_Status',
            marginal='box',
            title=f"{x_axis if x_axis != y_axis else y_axis} Distribution",
            color_discrete_map={'Non-Churned': colors['primary'], 'Churned': colors['danger']}
        )
    
    fig.update_layout(
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'])
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
    
    fig_corr.update_layout(
        title=dict(
            text="Feature Correlation Matrix",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(color='#FFFFFF', size=20)
        ),
        font=dict(color='#FFFFFF', size=14),
        coloraxis_colorbar=dict(
            title="Correlation",
            titlefont=dict(color='#FFFFFF', size=14),
            tickfont=dict(color='#FFFFFF', size=12)
        )
    )
    
    fig_corr = update_plotly_layout(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True, config=get_plot_config())

elif page == "❓ Help & Guide":
    st.title("How to Use This Dashboard")
    
    st.markdown("""
    ## Welcome to the Bank Customer Retention Analysis Dashboard! 👋
    
    This guide will help you navigate and make the most of our interactive dashboard.
    
    ### 📍 Navigation
    - Use the sidebar menu (🎯) to switch between different sections
    - Each section provides unique insights and analysis capabilities
    
    ### 🔍 Key Features by Section
    
    #### 1. Overview Dashboard (📊)
    - View key performance metrics at a glance
    - Explore geographic distribution of churn on an interactive map
    - Analyze age distribution and product usage patterns
    - **Tip**: Hover over charts for detailed information
    
    #### 2. Customer Demographics (👥)
    - Use sliders to filter age ranges
    - Compare gender distribution between churned and retained customers
    - **Tip**: Click on legend items to show/hide specific categories
    
    #### 3. Financial Analysis (💰)
    - Adjust balance range using the slider
    - View credit score distribution
    - Analyze relationships between financial metrics
    - **Tip**: Use the zoom tools to focus on specific areas
    
    #### 4. Satisfaction Analysis (⭐)
    - Select specific satisfaction scores to analyze
    - View complaint rates by geography
    - Track satisfaction trends
    - **Tip**: Click through the sunburst chart layers for detailed views
    
    #### 5. Churn Prediction (🔮)
    - Input customer details using sliders and dropdowns
    - Get real-time churn probability predictions
    - View feature importance analysis
    - **Tip**: Try different combinations to understand risk factors
    
    #### 6. Interactive Analysis (📈)
    - Choose different features for X and Y axes
    - Select from multiple plot types
    - Explore correlation between variables
    - **Tip**: Use the drawing tools to annotate interesting patterns
    
    ### 🛠️ Interactive Features
    
    #### Drawing Tools
    1. Click the pencil icon in the chart toolbar
    2. Choose from lines, shapes, or text annotations
    3. Draw or write on the charts
    4. Use the eraser tool to remove annotations
    
    #### Chart Controls
    - 📱 Pan: Click and drag
    - 🔍 Zoom: Use the zoom buttons or scroll wheel
    - 📸 Download: Click the camera icon
    - ↩️ Reset: Double click or use the home button
    
    #### Filtering Data
    - Use sliders to adjust ranges
    - Click on legend items to filter categories
    - Select multiple items in dropdowns
    
    ### 💡 Pro Tips
    1. **Hover** over any chart element for detailed information
    2. **Double-click** charts to reset the view
    3. Use the **drawing tools** to highlight important findings
    4. **Export** charts using the camera icon
    5. **Combine filters** for deeper analysis
    
    ### 🎨 Customization
    - Charts are interactive and can be:
        - Zoomed
        - Panned
        - Downloaded
        - Annotated
        - Reset
    
    ### 🤔 Need Help?
    - Look for tooltips and hover information
    - Check the documentation in our repository
    - Open an issue for support
    
    ### 🔄 Regular Updates
    We continuously improve the dashboard with:
    - New features
    - Enhanced visualizations
    - Performance optimizations
    - User experience improvements
    """)

# Footer
st.markdown("---")
st.markdown("Dashboard created for Bank Customer Retention Analysis | © 2025") 