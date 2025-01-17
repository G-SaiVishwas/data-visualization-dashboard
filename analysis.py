import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set style for all plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Read data
df = pd.read_csv('Customer-Churn-Records.csv')

# Create visualizations directory
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Overview Metrics
print("\n=== Overview Metrics ===")
print(f"Overall Churn Rate: {df['Exited'].mean()*100:.1f}%")
print(f"Active Members: {df['IsActiveMember'].mean()*100:.1f}%")
print(f"Average Satisfaction: {df['Satisfaction Score'].mean():.1f}/5")
print(f"Complaint Rate: {df['Complain'].mean()*100:.1f}%")

# 2. Geographic Analysis
plt.figure(figsize=(10, 6))
churn_by_geography = df.groupby('Geography')['Exited'].mean() * 100
sns.barplot(x=churn_by_geography.index, y=churn_by_geography.values)
plt.title('Customer Churn Rate by Geography')
plt.ylabel('Churn Rate (%)')
plt.savefig('visualizations/churn_by_geography.png', bbox_inches='tight')
plt.close()

print("\n=== Geographic Distribution of Churn ===")
for country in df['Geography'].unique():
    print(f"{country}: {df[df['Geography'] == country]['Exited'].mean()*100:.1f}% churn rate")

# 3. Age Analysis
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Age', hue='Exited', common_norm=False)
plt.title('Age Distribution: Churned vs Non-Churned Customers')
plt.savefig('visualizations/age_distribution.png', bbox_inches='tight')
plt.close()

print("\n=== Age Analysis ===")
print(f"Average age of churned customers: {df[df['Exited']==1]['Age'].mean():.1f}")
print(f"Average age of retained customers: {df[df['Exited']==0]['Age'].mean():.1f}")

# 4. Financial Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title('Account Balance Distribution by Churn Status')
plt.savefig('visualizations/balance_distribution.png', bbox_inches='tight')
plt.close()

print("\n=== Financial Analysis ===")
print(f"Average balance of churned customers: ${df[df['Exited']==1]['Balance'].mean():,.2f}")
print(f"Average balance of retained customers: ${df[df['Exited']==0]['Balance'].mean():,.2f}")

# 5. Product Usage
plt.figure(figsize=(10, 6))
product_data = pd.crosstab(df['NumOfProducts'], df['Exited'], normalize='index') * 100
product_data.plot(kind='bar')
plt.title('Churn Rate by Number of Products')
plt.xlabel('Number of Products')
plt.ylabel('Percentage')
plt.savefig('visualizations/product_analysis.png', bbox_inches='tight')
plt.close()

print("\n=== Product Usage Analysis ===")
print(f"Average products (churned customers): {df[df['Exited']==1]['NumOfProducts'].mean():.2f}")
print(f"Average products (retained customers): {df[df['Exited']==0]['NumOfProducts'].mean():.2f}")

# 6. Satisfaction Analysis
plt.figure(figsize=(10, 6))
satisfaction_data = pd.crosstab(df['Satisfaction Score'], df['Exited'], normalize='index') * 100
satisfaction_data.plot(kind='bar')
plt.title('Churn Rate by Satisfaction Score')
plt.xlabel('Satisfaction Score')
plt.ylabel('Percentage')
plt.savefig('visualizations/satisfaction_analysis.png', bbox_inches='tight')
plt.close()

print("\n=== Satisfaction Analysis ===")
for score in sorted(df['Satisfaction Score'].unique()):
    print(f"Score {score}: {df[df['Satisfaction Score']==score]['Exited'].mean()*100:.1f}% churn rate")

# 7. Correlation Analysis
plt.figure(figsize=(12, 10))
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited',
                'Complain', 'Satisfaction Score']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0)
plt.title('Correlation Matrix of Key Factors')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png', bbox_inches='tight')
plt.close()

print("\n=== Key Recommendations ===")
print("""
1. Targeted Retention Program for German Market
   - Investigate specific issues in the German market
   - Develop country-specific retention strategies

2. Age-Specific Customer Engagement
   - Create specialized products for older customers
   - Enhance digital banking support for senior customers

3. High-Value Customer Program
   - Implement VIP service for high-balance accounts
   - Develop personalized relationship management

4. Product Diversification
   - Encourage multi-product relationships
   - Create bundled offerings with better benefits

5. Proactive Complaint Resolution
   - Implement early warning system for customer dissatisfaction
   - Enhance complaint resolution processes
""") 