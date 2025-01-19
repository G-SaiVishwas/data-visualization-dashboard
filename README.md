# Bank Customer Churn Analysis Dashboard

A comprehensive, interactive dashboard for analyzing bank customer churn data with predictive capabilities and advanced visualizations.

## Features

### 1. Interactive Dashboards

- **Overview Dashboard**: Key metrics and geographic distribution
- **Customer Demographics**: Age and gender analysis
- **Financial Analysis**: Balance and credit score insights
- **Satisfaction Analysis**: Customer satisfaction trends
- **Churn Prediction**: ML-powered churn probability prediction
- **Interactive Analysis**: Custom visualization builder
- **Help & Guide**: Comprehensive user guide and tips

### 2. Advanced Visualizations

- Interactive geographic maps
- Dynamic sunburst charts
- Violin plots with individual points
- Customizable histograms
- Correlation matrices
- Real-time gauge charts
- Drawing and annotation tools

### 3. Key Features

- Dark theme with modern UI
- Interactive filtering and data exploration
- Real-time churn predictions
- Drawing and annotation capabilities
- Custom analysis builder
- Comprehensive metric cards
- Data-driven recommendations

### 4. Technical Features

- Machine learning-powered predictions
- LightGBM model integration
- Standardized data preprocessing
- Feature importance analysis
- Interactive plot configurations
- Responsive design
- Optimized performance

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd data-visualization-dashboard
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the dashboard:

```bash
streamlit run dashboard.py
```

2. Access the dashboard at: http://localhost:8501

## Dashboard Sections

### 1. Overview (📊)

- Key performance metrics
- Geographic distribution of churn
- Age distribution analysis
- Product usage patterns

### 2. Customer Demographics (👥)

- Age distribution analysis
- Gender distribution
- Interactive filtering
- Demographic trends

### 3. Financial Analysis (💰)

- Balance distribution
- Credit score analysis
- Financial metrics correlation
- Risk assessment

### 4. Satisfaction Analysis (⭐)

- Satisfaction score distribution
- Complaint analysis
- Geographic satisfaction trends
- Customer feedback insights

### 5. Churn Prediction (🔮)

- Real-time churn probability
- Risk factor analysis
- Personalized recommendations
- Feature importance visualization

### 6. Interactive Analysis (📈)

- Custom visualization builder
- Correlation analysis
- Multi-variable comparison
- Trend discovery tools

### 7. Help & Guide (❓)

- Comprehensive user guide
- Interactive features tutorial
- Navigation instructions
- Pro tips and best practices
- Feature-by-feature walkthrough
- Chart interaction guide
- Customization options

## Technical Stack

- **Python 3.9+**
- **Streamlit**: Main dashboard framework
- **Plotly**: Interactive visualizations
- **LightGBM**: Machine learning model
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Data preprocessing

## Data Features

The analysis uses comprehensive customer data including:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary
- Complaints
- Satisfaction Score
- Churn Status

## Project Structure

```
├── dashboard.py        # Main dashboard application
├── model.py           # Machine learning model
├── requirements.txt   # Project dependencies
├── models/           # Saved model files
│   ├── churn_model.joblib
│   ├── scaler.joblib
│   ├── le_geography.joblib
│   └── le_gender.joblib
└── README.md         # Documentation
```

## Key Insights

- Analyze customer churn patterns
- Identify high-risk customer segments
- Monitor satisfaction trends
- Track complaint rates by region
- Evaluate product performance
- Assess financial indicators
- Predict customer churn probability

## Recommendations

The dashboard provides actionable insights for:

1. **Customer Retention**:

   - Targeted intervention strategies
   - Personalized engagement plans
   - Risk mitigation approaches

2. **Service Improvement**:

   - Product optimization
   - Service quality enhancement
   - Customer experience improvement

3. **Business Strategy**:
   - Data-driven decision making
   - Resource allocation
   - Market segmentation

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue in the repository.

## Acknowledgments

- Built with Streamlit and Plotly
- Powered by LightGBM
- Designed for optimal user experience
