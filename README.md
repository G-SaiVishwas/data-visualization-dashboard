# Bank Customer Churn Analysis Dashboard

A comprehensive analysis and visualization tool for bank customer churn data, featuring multiple interactive dashboards and analysis methods.

## Project Overview

This project provides multiple ways to analyze and visualize bank customer churn data:
1. **Interactive Dash Dashboard**: Modern, responsive dashboard with real-time filtering and interactive visualizations
2. **Static Analysis Script**: Generates detailed visualizations and insights saved as image files
3. **Streamlit Dashboard**: Alternative interactive dashboard with additional features

## Key Features

- Geographic distribution of customer churn
- Age and demographic analysis
- Financial metrics visualization
- Customer satisfaction correlation
- Product usage patterns
- Interactive filtering and data exploration
- Comprehensive metric cards
- Detailed recommendations based on analysis

## Technologies Used

- **Python 3.13+**
- **Dash**: For the main interactive dashboard
- **Plotly**: For interactive visualizations
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Streamlit**: For alternative dashboard
- **Matplotlib & Seaborn**: For static visualizations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Dash Dashboard (Primary)
Run the interactive Dash dashboard:
```bash
python app.py
```
Access the dashboard at: http://127.0.0.1:8050

### 2. Static Analysis
Generate static visualizations and analysis:
```bash
python analysis.py
```
The visualizations will be saved in the `visualizations` directory.

### 3. Streamlit Dashboard (Alternative)
Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```
Access the dashboard at: http://localhost:8501

## Data Description

The analysis uses the following customer data points:
- Credit Score
- Geography
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary
- Churn Status
- Complaints
- Satisfaction Score

## Key Insights

1. **Geographic Patterns**: 
   - Germany: 32.4% churn rate
   - Spain: 16.7% churn rate
   - France: 16.2% churn rate

2. **Age Insights**:
   - Average age of churned customers: 44.8 years
   - Average age of retained customers: 37.4 years

3. **Financial Patterns**:
   - Average balance of churned customers: $91,109.48
   - Average balance of retained customers: $72,742.75

4. **Product Usage**:
   - Average products for churned customers: 1.48
   - Average products for retained customers: 1.54

## Recommendations

1. Implement targeted retention programs for the German market
2. Develop age-specific engagement strategies
3. Create high-value customer programs
4. Focus on product diversification
5. Implement proactive complaint resolution

## Project Structure

```
├── app.py              # Main Dash dashboard
├── analysis.py         # Static analysis script
├── dashboard.py        # Streamlit dashboard
├── requirements.txt    # Project dependencies
└── visualizations/     # Generated visualizations
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or suggestions, please open an issue in the repository. 
