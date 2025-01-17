import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('Customer-Churn-Records.csv')

# Set the style for better visualizations
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Create a directory for saving plots
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Customer Churn Rate by Geography
plt.figure(figsize=(10, 6))
churn_by_geography = df.groupby('Geography')['Exited'].mean() * 100
sns.barplot(x=churn_by_geography.index, y=churn_by_geography.values)
plt.title('Customer Churn Rate by Geography')
plt.ylabel('Churn Rate (%)')
plt.savefig('visualizations/churn_by_geography.png')
plt.close()

# 2. Age Distribution of Churned vs Non-Churned Customers
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Age', hue='Exited', common_norm=False)
plt.title('Age Distribution: Churned vs Non-Churned Customers')
plt.savefig('visualizations/age_distribution.png')
plt.close()

# 3. Correlation between Credit Score and Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Credit Score Distribution by Churn Status')
plt.savefig('visualizations/creditscore_churn.png')
plt.close()

# 4. Satisfaction Score Analysis
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Satisfaction Score', hue='Exited')
plt.title('Customer Satisfaction Score Distribution')
plt.savefig('visualizations/satisfaction_distribution.png')
plt.close()

# 5. Balance Distribution by Churn Status
plt.figure(figsize=(12, 6))
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title('Account Balance Distribution by Churn Status')
plt.savefig('visualizations/balance_distribution.png')
plt.close()

# Generate insights
print("\nKey Insights:")
print("-" * 50)

# Churn rate by geography
print("\n1. Geographic Distribution of Churn:")
for country in df['Geography'].unique():
    churn_rate = df[df['Geography'] == country]['Exited'].mean() * 100
    print(f"{country}: {churn_rate:.1f}% churn rate")

# Age-related insights
print("\n2. Age-related Insights:")
avg_age_churned = df[df['Exited'] == 1]['Age'].mean()
avg_age_stayed = df[df['Exited'] == 0]['Age'].mean()
print(f"Average age of churned customers: {avg_age_churned:.1f}")
print(f"Average age of retained customers: {avg_age_stayed:.1f}")

# Credit Score insights
print("\n3. Credit Score Insights:")
avg_credit_churned = df[df['Exited'] == 1]['CreditScore'].mean()
avg_credit_stayed = df[df['Exited'] == 0]['CreditScore'].mean()
print(f"Average credit score of churned customers: {avg_credit_churned:.1f}")
print(f"Average credit score of retained customers: {avg_credit_stayed:.1f}")

# Satisfaction Score analysis
print("\n4. Satisfaction Score Analysis:")
for score in sorted(df['Satisfaction Score'].unique()):
    churn_rate = df[df['Satisfaction Score'] == score]['Exited'].mean() * 100
    print(f"Score {score}: {churn_rate:.1f}% churn rate")

# Balance-related insights
print("\n5. Balance-related Insights:")
avg_balance_churned = df[df['Exited'] == 1]['Balance'].mean()
avg_balance_stayed = df[df['Exited'] == 0]['Balance'].mean()
print(f"Average balance of churned customers: ${avg_balance_churned:,.2f}")
print(f"Average balance of retained customers: ${avg_balance_stayed:,.2f}")

# Additional correlation analysis
print("\n6. Additional Insights:")
correlation = df['Exited'].corr(df['IsActiveMember'])
print(f"Correlation between Active Membership and Churn: {correlation:.3f}")

# Product-related insights
print("\n7. Product Usage Insights:")
avg_products_churned = df[df['Exited'] == 1]['NumOfProducts'].mean()
avg_products_stayed = df[df['Exited'] == 0]['NumOfProducts'].mean()
print(f"Average number of products (churned customers): {avg_products_churned:.2f}")
print(f"Average number of products (retained customers): {avg_products_stayed:.2f}")

# Card Type analysis
if 'Card Type' in df.columns:
    print("\n8. Card Type Analysis:")
    for card_type in df['Card Type'].unique():
        churn_rate = df[df['Card Type'] == card_type]['Exited'].mean() * 100
        print(f"{card_type}: {churn_rate:.1f}% churn rate") 