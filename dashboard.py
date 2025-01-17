import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Bank Customer Retention Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Customer-Churn-Records.csv')
    return df

df = load_data()

# Title and introduction
st.title("🏦 Bank Customer Retention Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive insights into customer retention metrics and behaviors.
Use the sidebar to navigate through different analyses and explore interactive visualizations.
""")

# Sidebar
st.sidebar.header("Dashboard Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis",
    ["Overview", "Customer Demographics", "Financial Metrics", "Satisfaction Analysis", "Churn Prediction Factors"]
)

# Overview Section
if analysis_type == "Overview":
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = (df['Exited'].mean() * 100)
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    
    with col2:
        active_rate = (df['IsActiveMember'].mean() * 100)
        st.metric("Active Members", f"{active_rate:.1f}%")
    
    with col3:
        avg_satisfaction = df['Satisfaction Score'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    with col4:
        complaint_rate = (df['Complain'].mean() * 100)
        st.metric("Complaint Rate", f"{complaint_rate:.1f}%")

    # Churn by Geography
    st.subheader("Geographic Distribution of Churn")
    fig_geo = px.bar(
        df.groupby('Geography')['Exited'].agg(['count', 'mean']).reset_index(),
        x='Geography',
        y='mean',
        color='Geography',
        text=df.groupby('Geography')['Exited'].mean().apply(lambda x: f'{x:.1%}'),
        title="Churn Rate by Country"
    )
    fig_geo.update_traces(textposition='outside')
    st.plotly_chart(fig_geo, use_container_width=True)

# Customer Demographics Section
elif analysis_type == "Customer Demographics":
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution
        st.subheader("Age Distribution by Churn Status")
        fig_age = px.histogram(
            df,
            x="Age",
            color="Exited",
            marginal="box",
            nbins=30,
            title="Age Distribution"
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender Distribution
        st.subheader("Gender Distribution")
        gender_churn = pd.crosstab(df['Gender'], df['Exited'], normalize='index') * 100
        fig_gender = px.bar(
            gender_churn,
            title="Churn Rate by Gender",
            barmode='group'
        )
        st.plotly_chart(fig_gender, use_container_width=True)

# Financial Metrics Section
elif analysis_type == "Financial Metrics":
    col1, col2 = st.columns(2)
    
    with col1:
        # Balance Distribution
        st.subheader("Account Balance Distribution")
        fig_balance = px.box(
            df,
            x="Exited",
            y="Balance",
            color="Exited",
            title="Balance Distribution by Churn Status"
        )
        st.plotly_chart(fig_balance, use_container_width=True)
    
    with col2:
        # Credit Score Analysis
        st.subheader("Credit Score Analysis")
        fig_credit = px.violin(
            df,
            x="Exited",
            y="CreditScore",
            color="Exited",
            box=True,
            title="Credit Score Distribution by Churn Status"
        )
        st.plotly_chart(fig_credit, use_container_width=True)

    # Product Analysis
    st.subheader("Product Usage Analysis")
    product_churn = pd.crosstab(df['NumOfProducts'], df['Exited'], normalize='index') * 100
    fig_products = px.bar(
        product_churn,
        title="Churn Rate by Number of Products",
        barmode='group'
    )
    st.plotly_chart(fig_products, use_container_width=True)

# Satisfaction Analysis Section
elif analysis_type == "Satisfaction Analysis":
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction Score Distribution
        st.subheader("Satisfaction Score Distribution")
        fig_satisfaction = px.histogram(
            df,
            x="Satisfaction Score",
            color="Exited",
            barmode="group",
            title="Satisfaction Score Distribution by Churn Status"
        )
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        # Complaints Analysis
        st.subheader("Complaints Impact")
        complaint_churn = pd.crosstab(df['Complain'], df['Exited'], normalize='index') * 100
        fig_complaints = px.bar(
            complaint_churn,
            title="Churn Rate by Complaint Status",
            barmode='group'
        )
        st.plotly_chart(fig_complaints, use_container_width=True)

# Churn Prediction Factors
else:
    # Correlation Matrix
    st.subheader("Correlation Analysis")
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                   'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 
                   'Complain', 'Satisfaction Score']
    corr_matrix = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Key Factors",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Key Findings
    st.subheader("Key Findings and Recommendations")
    st.markdown("""
    ### 🎯 Top Churn Risk Factors:
    1. **Geographic Location**: German market shows significantly higher churn (32.4%)
    2. **Age**: Older customers (avg. 44.8 years) are more likely to churn
    3. **Balance**: Higher balance customers show increased churn risk
    
    ### 💡 Recommendations:
    1. **Targeted Retention Program for German Market**
       - Investigate specific issues in the German market
       - Develop country-specific retention strategies
    
    2. **Age-Specific Customer Engagement**
       - Create specialized products for older customers
       - Enhance digital banking support for senior customers
    
    3. **High-Value Customer Program**
       - Implement VIP service for high-balance accounts
       - Develop personalized relationship management
    
    4. **Product Diversification**
       - Encourage multi-product relationships
       - Create bundled offerings with better benefits
    
    5. **Proactive Complaint Resolution**
       - Implement early warning system for customer dissatisfaction
       - Enhance complaint resolution processes
    """)

# Footer
st.markdown("---")
st.markdown("Dashboard created for Statistella Round 2 | Bank Customer Retention Analysis") 